"""Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) predictions on the frozen 338-item benchmark (CPU),
model-card preprocessing (@user, http). Bootstrap vs stock FinBERT and vs seed-averaged arm macro-F1, mirroring stats_v2.py."""
import json, sys, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
HERE = Path(__file__).parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "analysis/results_v2/stats")); from stats_v2 import macro_f1, holm, B, RNG_SEED, LABELS, SEEDS, ARMS
MID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
def preprocess(text):
    return " ".join("@user" if t.startswith("@") and len(t) > 1 else "http" if t.startswith("http") else t for t in text.split(" "))
items = json.loads((ROOT / "analysis/benchmark_v2_frozen.json").read_text())["items"]
tok = AutoTokenizer.from_pretrained(MID); m = AutoModelForSequenceClassification.from_pretrained(MID).eval()
id2 = {int(k): v.lower() for k, v in m.config.id2label.items()}
L = {l: i for i, l in enumerate(LABELS)}; preds, probs = [], []
for i in range(0, len(items), 32):
    enc = tok([preprocess(x["text"]) for x in items[i:i+32]], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad(): p = torch.softmax(m(**enc).logits, -1).numpy()
    for row in p: preds.append(id2[int(row.argmax())]); probs.append({id2[j]: float(v) for j, v in enumerate(row)})
stock = json.loads((ROOT / "analysis/results_v1/runs/stock.json").read_text())["benchmark_max_length_128"]["predictions"]
g = np.array([L[r["gold"]] for r in stock]); ps = np.array([L[r["pred"]] for r in stock])
assert [L[x["label"]] for x in items] == list(g)
pr = np.array([L[p] for p in preds])
P = {(a, s): np.array([L[x["pred"]] for x in json.loads((ROOT / f"analysis/results_v2/runs/{a}_seed{s}.json").read_text())["benchmark_predictions"]]) for a in ARMS for s in SEEDS}
per = {}
for c, l in enumerate(LABELS):
    tp = int(((pr == c) & (g == c)).sum()); npred = int((pr == c).sum()); ngold = int((g == c).sum())
    prec = tp / npred if npred else 0.0; rec = tp / ngold; per[l] = dict(precision=prec, recall=rec, f1=2*prec*rec/(prec+rec) if prec+rec else 0.0, predicted=npred, support=ngold)
cm = [[int(((g == i) & (pr == j)).sum()) for j in range(3)] for i in range(3)]
rng = np.random.default_rng(RNG_SEED); idx = rng.integers(0, len(g), size=(B, len(g))); gb = g[idx]
mr = macro_f1(gb, pr[idx]); refs = {"stock": (macro_f1(gb, ps[idx]), float(macro_f1(g, ps)))}
for a in ARMS: refs[a] = (np.mean([macro_f1(gb, P[(a, s)][idx]) for s in SEEDS], axis=0), float(np.mean([macro_f1(g, P[(a, s)]) for s in SEEDS])))
obs = float(macro_f1(g, pr)); boot = []
for k, (bd, o) in refs.items():
    d = mr - bd; boot.append(dict(comparison=f"twitter_roberta vs {k}", observed_diff=obs - o, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                  p_two_sided=float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))), p_floor=1 / B))
for b, h in zip(boot, holm([b["p_two_sided"] for b in boot])): b["p_holm_4"] = float(h)
out = dict(model=MID, n_items=len(g), macro_f1=obs, accuracy=float(np.mean(pr == g)), per_class=per, confusion_rows_true_cols_pred=cm, label_order=LABELS,
           bootstrap=boot, settings=dict(B=B, seed=RNG_SEED, holm_family="4 comparisons (vs stock, arm_a, arm_b, arm_c)", max_length=512, device="cpu", torch=torch.__version__),
           predictions=[dict(id=x.get("id"), gold=x["label"], pred=p, probs=q) for x, p, q in zip(items, preds, probs)])
(HERE / "twitter_roberta_v2.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != "predictions"}, indent=1))
