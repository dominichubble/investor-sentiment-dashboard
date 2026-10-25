"""Single-condition FinBERT fine-tuning runner (DSFE revision, plan.md T7).

One (arm, seed) run: stratified 90/10 train/val split of the arm, AdamW with
weight decay + linear warm-up schedule, max_length=128, CPU, N epochs with
best-epoch selection on validation macro-F1 (3-label macro, zero_division=0),
then evaluation of the selected epoch on the frozen 338-item benchmark.
Benchmark predictions/logits are also stored for every epoch (transparency
only; selection uses validation exclusively).

Also provides stock (un-fine-tuned) FinBERT evaluation on the benchmark.
New file; does not modify finetune_cv.py.

keep_best_model=True (added 24 Oct, default off) also returns the val-selected model for extra evaluation.
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]
ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "analysis" / "benchmark_v2_frozen.json"
ARMS_DIR = ROOT / "analysis" / "training_sets_v1"

DEFAULTS = dict(epochs=3, lr=2e-5, batch_size=16, weight_decay=0.01, warmup_ratio=0.1,
                max_length=128, val_frac=0.10, eval_batch_size=64)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(LABELS)},
        "confusion_matrix": {"labels": LABELS,
                             "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()},
        "pred_counts": {l: int(sum(1 for x in y_pred if x == l)) for l in LABELS},
    }


def load_model(device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    assert [id2label[i] for i in range(3)] == LABELS, f"unexpected id2label {id2label}"
    return tok, model


@torch.no_grad()
def predict(model, tok, texts: List[str], max_length: int, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def load_benchmark():
    items = json.loads(BENCHMARK.read_text())["items"]
    assert len(items) == 338, len(items)
    return items


def pred_records(items, logits, text_key="text"):
    preds = [LABELS[i] for i in logits.argmax(1)]
    return preds, [{"id": it.get("id"), "gold": it["label"], "pred": pr,
                    "logits": [round(float(x), 6) for x in lg]}
                   for it, pr, lg in zip(items, preds, logits)]


def run_stock(out_path: Path, max_length: int = 128, threads: int | None = None) -> Dict:
    if threads: torch.set_num_threads(threads)
    t0 = time.time()
    tok, model = load_model()
    bench = load_benchmark()
    res = {"condition": "stock", "model": MODEL_NAME, "max_length": max_length, "torch_threads": torch.get_num_threads()}
    for ml in sorted({max_length, 512}):
        lg = predict(model, tok, [b["text"] for b in bench], ml, DEFAULTS["eval_batch_size"])
        preds, recs = pred_records(bench, lg)
        res[f"benchmark_max_length_{ml}"] = {"metrics": metrics([b["label"] for b in bench], preds), "predictions": recs}
    res["benchmark"] = res[f"benchmark_max_length_{max_length}"]["metrics"]
    res["wall_seconds"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1))
    return res


def run_condition(arm: str, seed: int, out_path: Path, epochs: int = 3, threads: int | None = None,
                  arms_dir: Path | None = None, keep_best_model: bool = False, **kw):
    cfg = {**DEFAULTS, **kw, "epochs": epochs}
    if threads: torch.set_num_threads(threads)
    t0 = time.time()
    set_seed(seed)
    arms_dir = Path(arms_dir).resolve() if arms_dir else ARMS_DIR
    arm_file = arms_dir / f"{arm}.json"
    rows = json.loads(arm_file.read_text())
    texts = [r["text"] for r in rows]; labels = [r["label"] for r in rows]
    idx = np.arange(len(rows))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg["val_frac"], random_state=seed, stratify=labels)
    tr_idx, va_idx = sorted(tr_idx.tolist()), sorted(va_idx.tolist())

    tok, model = load_model()
    l2i = {l: i for i, l in enumerate(LABELS)}
    enc = tok([texts[i] for i in tr_idx], padding=True, truncation=True, max_length=cfg["max_length"], return_tensors="pt")
    y = torch.tensor([l2i[labels[i]] for i in tr_idx])
    ds = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], y)
    g = torch.Generator(); g.manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, generator=g)

    no_decay = ("bias", "LayerNorm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(round(cfg["warmup_ratio"] * total)), total)

    bench = load_benchmark(); btexts = [b["text"] for b in bench]; bgold = [b["label"] for b in bench]
    vtexts = [texts[i] for i in va_idx]; vgold = [labels[i] for i in va_idx]
    epochs_log, best, best_state = [], None, None
    for ep in range(1, epochs + 1):
        te = time.time(); model.train(); run_loss = 0.0
        for ii, am, yy in loader:
            opt.zero_grad()
            out = model(input_ids=ii, attention_mask=am, labels=yy)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run_loss += float(out.loss.item())
        train_s = time.time() - te
        vlg = predict(model, tok, vtexts, cfg["max_length"], cfg["eval_batch_size"])
        vpred = [LABELS[i] for i in vlg.argmax(1)]
        vm = metrics(vgold, vpred)
        blg = predict(model, tok, btexts, cfg["max_length"], cfg["eval_batch_size"])
        bpred, brecs = pred_records(bench, blg)
        bm = metrics(bgold, bpred)
        e = {"epoch": ep, "train_loss": run_loss / max(len(loader), 1), "train_seconds": round(train_s, 1),
             "val": vm, "benchmark": bm, "benchmark_predictions": brecs,
             "val_predictions": [{"arm_index": i, "gold": g_, "pred": p_, "logits": [round(float(x), 6) for x in lg]}
                                 for i, g_, p_, lg in zip(va_idx, vgold, vpred, vlg)]}
        epochs_log.append(e)
        print(f"[{arm} s{seed}] epoch {ep}/{epochs} loss={e['train_loss']:.4f} val_macroF1={vm['macro_f1']:.4f} "
              f"bench_macroF1={bm['macro_f1']:.4f} ({train_s:.0f}s train)", flush=True)
        # selection on validation only; ties -> earlier epoch
        if best is None or vm["macro_f1"] > epochs_log[best - 1]["val"]["macro_f1"]:
            best = ep
            if keep_best_model:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sel = epochs_log[best - 1]
    import hashlib
    res = {"condition": "finetune", "arm": arm, "seed": seed, "arm_file": str(arm_file.relative_to(ROOT)),
           "arm_sha256": hashlib.sha256(arm_file.read_bytes()).hexdigest(), "n_arm": len(rows), "model": MODEL_NAME, "config": cfg,
           "torch_threads": torch.get_num_threads(), "n_train": len(tr_idx), "n_val": len(va_idx),
           "train_label_counts": {l: sum(1 for i in tr_idx if labels[i] == l) for l in LABELS},
           "val_label_counts": {l: sum(1 for i in va_idx if labels[i] == l) for l in LABELS},
           "val_arm_indices": va_idx, "selection": "max val macro_f1 (3-label, zero_division=0); ties->earlier epoch",
           "best_epoch": best, "val": sel["val"], "benchmark": sel["benchmark"],
           "benchmark_predictions": sel["benchmark_predictions"],
           "epochs": [{k: v for k, v in e.items()} for e in epochs_log],
           "wall_seconds": round(time.time() - t0, 1)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); tmp.replace(out_path)
    if keep_best_model:
        model.load_state_dict(best_state)
        return res, model, tok
    return res
