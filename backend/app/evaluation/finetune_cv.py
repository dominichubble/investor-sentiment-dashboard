"""
Multi-seed / cross-validated fine-tuning of FinBERT on the neutral-loaded corpus.

Addresses the reviewer concern that the headline fine-tuning result in the
manuscript rests on a SINGLE seed and a SINGLE benchmark, without
cross-validation. This script repeats the fine-tuning -> benchmark-evaluation
loop across multiple random seeds (default) or across stratified folds of the
fine-tuning corpus (--kfold), and reports the mean +/- standard deviation of
every headline metric, plus a paired McNemar test of the fine-tuned model
against the stock model on the unchanged 200-sample benchmark.

It does NOT fabricate numbers: the corpus must be supplied and the metrics are
computed when the script is run on a machine with the model weights.

The fine-tuning corpus is a JSON list of objects:
    [{"text": "...", "label": "neutral"}, ...]
with labels in {positive, negative, neutral}. A template is provided at
data/evaluation/finetune_corpus.EXAMPLE.json -- replace it with the real
300-sample neutral-but-lexically-loaded corpus described in the manuscript.

Usage (from the backend directory):
    python -m app.evaluation.finetune_cv \
        --corpus data/evaluation/finetune_corpus.json \
        --seeds 13 21 42 84 168 \
        --epochs 3 --lr 2e-5 --batch-size 16 \
        --output data/evaluation/finetune_cv_results.json

    # k-fold sensitivity over the fine-tuning corpus instead of seeds:
    python -m app.evaluation.finetune_cv --corpus ... --kfold 5 --output ...

Dependencies: torch, transformers, numpy, scipy.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from . import stats as st
from .labeled_dataset import get_labeled_dataset
from .metrics import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
# Label <-> id mapping aligned with the FinBERTModel wrapper (positive=0,
# negative=1, neutral=2) so the fine-tuned head decodes consistently.
LABELS = ["positive", "negative", "neutral"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
def load_corpus(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Fine-tuning corpus not found at {p}.\n"
            "Supply the 300-sample neutral-but-lexically-loaded corpus described "
            "in the manuscript (Section 3.3). A schema template is at "
            "data/evaluation/finetune_corpus.EXAMPLE.json."
        )
    data = json.loads(p.read_text())
    for row in data:
        if row.get("label") not in LABEL2ID:
            raise ValueError(f"Bad label in corpus: {row!r}")
    logger.info("Loaded %d fine-tuning samples from %s", len(data), p)
    return data


# ---------------------------------------------------------------------------
# Training / inference (manual loop -- stable across transformers versions)
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one(
    corpus: Sequence[Dict[str, str]],
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
):
    """Fine-tune a fresh FinBERT on the given corpus; return (model, tokenizer, device)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, use_safetensors=True
    ).to(device)

    texts = [r["text"] for r in corpus]
    labels = [LABEL2ID[r["label"]] for r in corpus]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512,
                    return_tensors="pt")
    dataset = TensorDataset(
        enc["input_ids"], enc["attention_mask"], torch.tensor(labels)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for input_ids, attn, y in loader:
            input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, labels=y)
            out.loss.backward()
            optimizer.step()
            running += float(out.loss.item())
        logger.info("  seed %d epoch %d/%d  mean loss %.4f",
                    seed, epoch + 1, epochs, running / max(len(loader), 1))
    return model, tokenizer, device


def _predict(model, tokenizer, device, texts: Sequence[str],
             batch_size: int = 32) -> List[str]:
    import torch

    model.eval()
    out_labels: List[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to(device)
            logits = model(**enc).logits
            idx = torch.argmax(logits, dim=-1).tolist()
            out_labels.extend(LABELS[j] for j in idx)
    return out_labels


def _stock_predictions(texts: Sequence[str]) -> List[str]:
    """Stock (un-fine-tuned) FinBERT predictions on the benchmark, for McNemar."""
    from app.models.finbert_model import get_model

    model = get_model()
    return [p["label"] for p in model.predict_batch(list(texts), batch_size=32)]


def _metrics_row(true_labels, pred_labels) -> Dict[str, float]:
    ev = evaluate_predictions(true_labels, pred_labels)
    neu = ev["per_class"]["neutral"]
    return {
        "accuracy": ev["accuracy"],
        "macro_f1": ev["macro_f1"],
        "neutral_precision": neu["precision"],
        "neutral_recall": neu["recall"],
        "neutral_f1": neu["f1"],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _make_splits(corpus, seeds, kfold) -> List[Tuple[str, List[Dict[str, str]]]]:
    """Return a list of (run_label, training_subset)."""
    if kfold and kfold > 1:
        import numpy as np
        from sklearn.model_selection import StratifiedKFold

        y = [r["label"] for r in corpus]
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
        runs = []
        arr = np.array(corpus, dtype=object)
        for fold, (train_idx, _) in enumerate(skf.split(arr, y)):
            runs.append((f"fold{fold}", [corpus[i] for i in train_idx]))
        return runs
    return [(f"seed{s}", list(corpus)) for s in seeds]


def run(corpus_path: str, seeds: List[int], kfold: int, epochs: int,
        lr: float, batch_size: int) -> Dict:
    corpus = load_corpus(corpus_path)
    benchmark = get_labeled_dataset()
    bench_texts = [b["text"] for b in benchmark]
    bench_true = [b["label"] for b in benchmark]

    logger.info("Computing stock-FinBERT benchmark predictions (McNemar baseline)...")
    stock_preds = _stock_predictions(bench_texts)
    stock_row = _metrics_row(bench_true, stock_preds)
    logger.info("Stock FinBERT: %s", stock_row)

    splits = _make_splits(corpus, seeds, kfold)
    per_run: List[Dict] = []
    for run_label, train_subset in splits:
        seed = int(run_label.replace("seed", "")) if run_label.startswith("seed") else 42
        logger.info("=== Run %s (train n=%d) ===", run_label, len(train_subset))
        model, tok, device = _train_one(train_subset, seed, epochs, lr, batch_size)
        ft_preds = _predict(model, tok, device, bench_texts)
        row = _metrics_row(bench_true, ft_preds)
        row["run"] = run_label
        row["mcnemar_vs_stock"] = st.mcnemar_test(bench_true, stock_preds, ft_preds)
        per_run.append(row)
        logger.info("Run %s fine-tuned: %s", run_label, {k: row[k] for k in
                    ("accuracy", "macro_f1", "neutral_recall")})
        # free GPU memory between runs
        try:
            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # Aggregate mean +/- SD across runs.
    aggregate = {
        metric: st.summarise_runs([r[metric] for r in per_run])
        for metric in ("accuracy", "macro_f1", "neutral_precision",
                       "neutral_recall", "neutral_f1")
    }

    return {
        "config": {
            "corpus": corpus_path,
            "mode": "kfold" if kfold else "multi_seed",
            "kfold": kfold,
            "seeds": seeds if not kfold else None,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "n_runs": len(per_run),
        },
        "stock_finbert": stock_row,
        "per_run": per_run,
        "aggregate_mean_sd": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="data/evaluation/finetune_corpus.json")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[13, 21, 42, 84, 168],
                        help="Random seeds for the multi-seed protocol.")
    parser.add_argument("--kfold", type=int, default=0,
                        help="If >1, run stratified k-fold over the corpus "
                             "instead of multi-seed.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = run(args.corpus, args.seeds, args.kfold, args.epochs, args.lr,
                  args.batch_size)

    logger.info("\n=== AGGREGATE (mean +/- SD across %d runs) ===",
                results["config"]["n_runs"])
    for metric, s in results["aggregate_mean_sd"].items():
        logger.info("  %-18s %.4f +/- %.4f  (min %.4f, max %.4f)",
                    metric, s["mean"], s["sd"], s["min"], s["max"])

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()
