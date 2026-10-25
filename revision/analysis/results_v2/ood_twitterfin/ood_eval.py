"""OOD stress test: zeroshot/twitter-financial-news-sentiment validation split (0 Bearish=negative, 1 Bullish=positive, 2 Neutral).
Stock FinBERT + v2 arm_a s84, arm_b s84, arm_c s21 (checkpoints were not saved earlier, so these are retrained with
identical finetune_runner settings and saved under checkpoints/). max_length 128. Paired item bootstrap vs stock.
"""
import csv, json, re, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "analysis/finetune_v1"))
import finetune_runner as fr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
MAP = {"0": "negative", "1": "positive", "2": "neutral"}
MODELS = [("arm_a", 84), ("arm_b", 84), ("arm_c", 21)]
B, RNG = 10000, 20261024


def norm(t): return re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", t.lower())).strip()


def macro_f1(g, p):
    f = []
    for l in range(3):
        tp = np.sum((g == l) & (p == l), -1); fp = np.sum((g != l) & (p == l), -1); fn = np.sum((g == l) & (p != l), -1)
        d = 2 * tp + fp + fn; f.append(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0))
    return np.mean(f, 0)


def main():
    t0 = time.time()
    rows = [{"text": r["text"], "label": MAP[r["label"]]} for r in csv.DictReader(open(HERE / "data/sent_valid.csv", encoding="utf-8"))]
    # overlap check
    ours = {}
    for a in "abc":
        ours[f"arm_{a}"] = [x["text"] for x in json.loads((ROOT / f"analysis/training_sets_v2/arm_{a}.json").read_text())]
    ours["benchmark_frozen_338"] = [x["text"] for x in json.loads((ROOT / "analysis/benchmark_v2_frozen.json").read_text())["items"]]
    ours["benchmark_final_359"] = [x["text"] for x in (lambda d: d["items"] if isinstance(d, dict) else d)(json.loads((ROOT / "analysis/benchmark_v2_final.json").read_text()))]
    tw_exact = {r["text"].strip() for r in rows}; tw_norm = {norm(r["text"]) for r in rows} - {""}
    overlap = {k: {"n": len(v), "exact": sum(t.strip() in tw_exact for t in v), "normalised": sum(norm(t) in tw_norm for t in v)} for k, v in ours.items()}
    data = {"n": len(rows), "n_unique_text": len(tw_exact), "label_counts": {l: sum(r["label"] == l for r in rows) for l in fr.LABELS}}
    print("data", json.dumps(data), "overlap", json.dumps(overlap), flush=True)
    texts = [r["text"] for r in rows]; L = {l: i for i, l in enumerate(fr.LABELS)}; g = np.array([L[r["label"]] for r in rows])
    preds, summ = {}, {"data": data, "overlap": overlap, "max_length": 128, "models": {}}
    for name, seed in [("stock", None)] + MODELS:
        key = name if seed is None else f"{name}_seed{seed}"; outp = HERE / f"{key}.json"
        if outp.exists():
            j = json.loads(outp.read_text()); preds[key] = np.array([L[x["pred"]] for x in j["predictions"]]); summ["models"][key] = j["summary"]; continue
        ts = time.time(); extra = {}
        ck = HERE / "checkpoints" / key
        if seed is None:
            tok, model = fr.load_model()
        elif (ck / "config.json").exists():
            tok = AutoTokenizer.from_pretrained(ck); model = AutoModelForSequenceClassification.from_pretrained(ck); extra["checkpoint"] = "loaded"
        else:
            saved = json.loads((ROOT / f"analysis/results_v2/runs/{key}.json").read_text())
            res, model, tok = fr.run_condition(name, seed, HERE / "retrain" / f"{key}.json", epochs=3,
                                               arms_dir=ROOT / "analysis/training_sets_v2", keep_best_model=True)
            extra = {"best_epoch": res["best_epoch"], "retrain_seconds": res["wall_seconds"],
                     "retrain_reproduces_saved_benchmark_preds": [a["pred"] for a in res["benchmark_predictions"]] == [b["pred"] for b in saved["benchmark_predictions"]]}
            ck.mkdir(parents=True, exist_ok=True); model.save_pretrained(ck, safe_serialization=True); tok.save_pretrained(ck)
            extra["checkpoint"] = str(ck.relative_to(ROOT))
        lg = fr.predict(model, tok, texts, 128, 64); p = [fr.LABELS[i] for i in lg.argmax(1)]
        m = fr.metrics([r["label"] for r in rows], p)
        s = {**extra, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "recall": {l: v["recall"] for l, v in m["per_class"].items()},
             "pred_counts": m["pred_counts"], "wall_seconds": round(time.time() - ts, 1)}
        outp.write_text(json.dumps({"summary": s, "metrics": m, "predictions": [{"i": i, "gold": r["label"], "pred": q, "logits": [round(float(x), 6) for x in l]}
                                                                              for i, (r, q, l) in enumerate(zip(rows, p, lg))]}, indent=1))
        preds[key] = np.array([L[x] for x in p]); summ["models"][key] = s
        print("done", key, round(s["macro_f1"], 4), round(s["accuracy"], 4), extra, s["wall_seconds"], flush=True)
    idx = np.random.default_rng(RNG).integers(0, len(g), size=(B, len(g))); gb = g[idx]
    st = macro_f1(gb, preds["stock"][idx]); st_acc = np.mean(preds["stock"][idx] == gb, 1)
    summ["bootstrap_vs_stock"] = {}
    for k in preds:
        if k == "stock": continue
        d = macro_f1(gb, preds[k][idx]) - st; da = np.mean(preds[k][idx] == gb, 1) - st_acc
        summ["bootstrap_vs_stock"][k] = {"macro_f1_diff": float(macro_f1(g, preds[k]) - macro_f1(g, preds["stock"])),
            "macro_f1_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "macro_f1_p": float(min(1, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))),
            "acc_diff": float(np.mean(preds[k] == g) - np.mean(preds["stock"] == g)), "acc_ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))]}
    summ["settings"] = {"B": B, "rng_seed": RNG}; summ["wall_seconds_this_invocation"] = round(time.time() - t0, 1)
    (HERE / "ood_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ["bootstrap_vs_stock"], indent=1)); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
