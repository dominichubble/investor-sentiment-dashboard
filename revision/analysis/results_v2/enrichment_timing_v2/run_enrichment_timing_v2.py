"""Rerun of the finance-emotion enrichment ablation and FinBERT/keyword inference
timing on the frozen 338-item benchmark (benchmark_v2_frozen.json).
Uses unmodified copies of the repo modules (src/backend/app/...), stock ProsusAI/finbert, CPU.
Ablation logic mirrors scripts/emotion_ablation.py. Two variants are reported:
  corrected: FinBERT probabilities captured (return_all_scores=True), as intended.
  as_coded:  predict_batch() without return_all_scores, as in the original script,
             which leaves probabilities at 0 (reproduces the original pipeline exactly).
Timing mirrors backend/app/evaluation/benchmark.py (predict_batch, batch_size=32)."""
import json, math, platform, statistics, sys, time
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src" / "backend"))
sys.path.insert(0, str(HERE / "src" / "scripts"))
import torch
from app.models.finbert_model import get_model
from app.models.keyword_sentiment import calculate_financial_sentiment
from app.analysis import finance_emotion
import types
_m = types.ModuleType('app.evaluation.labeled_dataset'); _m.get_labeled_dataset = None  # unused by the ablation helpers
import app; sys.modules['app.evaluation'] = types.ModuleType('app.evaluation'); sys.modules['app.evaluation.labeled_dataset'] = _m
import emotion_ablation as ea

items = json.load(open(HERE / "benchmark_v2_frozen.json"))["items"]
texts = [it["text"] for it in items]
model = get_model()

def ablate(preds):
    rows = []
    for t, p in zip(texts, preds):
        probs = p.get("scores") or {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        ent = ea.shannon_entropy(probs); lab = p["label"]; sc = float(p.get("score") or max(probs.values()))
        rows.append({"full": ea.run_full(t, lab, sc, probs, ent)["label"],
                     "no_entropy": ea.run_no_entropy(t, lab, sc, probs, ent)["label"],
                     "no_lexicon": ea.run_no_lexicon(t, lab, sc, probs, ent)["label"]})
    n = len(rows)
    return {"n": n,
            "distribution": {v: dict(Counter(r[v] for r in rows)) for v in ["full", "no_entropy", "no_lexicon"]},
            "change_fraction_vs_full": {v: sum(r[v] != r["full"] for r in rows) / n for v in ["no_entropy", "no_lexicon"]},
            "change_count_vs_full": {v: sum(r[v] != r["full"] for r in rows) for v in ["no_entropy", "no_lexicon"]}}

out = {"ablation": {"corrected": ablate(model.predict_batch(texts, batch_size=32, return_all_scores=True)),
                    "as_coded": ablate(model.predict_batch(texts, batch_size=32))}}
fb = []
for _ in range(5):
    s = time.time(); model.predict_batch(texts, batch_size=32); fb.append(time.time() - s)
kw = []
for _ in range(5):
    s = time.time(); [calculate_financial_sentiment(t) for t in texts]; kw.append(time.time() - s)
out["timing"] = {"n": len(texts), "finbert_runs_s": fb, "finbert_median_s": statistics.median(fb),
                 "finbert_ms_per_text": 1000 * statistics.median(fb) / len(texts),
                 "keyword_runs_s": kw, "keyword_median_s": statistics.median(kw),
                 "hardware": {"platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
                              "device": model.device, "torch": torch.__version__}}
json.dump(out, open(HERE / "enrichment_timing_v2.json", "w"), indent=2)
print(json.dumps(out, indent=2))
