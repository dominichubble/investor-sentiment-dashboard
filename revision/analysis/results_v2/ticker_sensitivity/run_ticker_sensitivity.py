"""Item 11 sensitivity: macro-F1 with vs without news items whose ticker starts with 'A'.
Existing per-item predictions only (no retraining). Arms = mean over 5 seeds.
Bootstrap: item-level, B=2000, seed 20261025; arms resample items, average macro-F1 over seeds."""
import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "analysis/results_v2/stats")); from stats_v2 import macro_f1, LABELS, SEEDS, ARMS
L = {l: i for i, l in enumerate(LABELS)}
items = json.loads((ROOT / "analysis/benchmark_v2_frozen.json").read_text())["items"]
ids = [x["id"] for x in items]; g = np.array([L[x["label"]] for x in items])
def load(recs):
    assert [r["id"] for r in recs] == ids; return np.array([L[r["pred"]] for r in recs])
M = {"stock": [load(json.loads((ROOT / "analysis/results_v1/runs/stock.json").read_text())["benchmark_max_length_128"]["predictions"])]}
for a in ARMS: M[a] = [load(json.loads((ROOT / f"analysis/results_v2/runs/{a}_seed{s}.json").read_text())["benchmark_predictions"]) for s in SEEDS]
M["twitter_roberta"] = [load(json.loads((ROOT / "analysis/results_v2/twitter_roberta_baseline/twitter_roberta_v2.json").read_text())["predictions"])]
kw = json.loads((ROOT / "analysis/results_v2/keyword_baseline/keyword_v2.json").read_text())
kp = kw.get("predictions") or kw.get("item_predictions"); M["keyword"] = [load(kp)]
ch = np.array([x["channel"] for x in items]); tk = np.array([str(x["ticker"] or "") for x in items])
anews = (ch == "news") & np.char.startswith(np.char.upper(tk), "A")
G = {"news_all": ch == "news", "news_A": anews, "news_nonA": (ch == "news") & ~anews,
     "all_338": np.ones(len(g), bool), "all_minus_A_news": ~anews}
rng = np.random.default_rng(20261025); B = 2000
def f1(P, idx): return float(np.mean([macro_f1(g[idx], p[idx]) for p in P]))
out = {"n": {k: int(v.sum()) for k, v in G.items()}, "macro_f1": {}, "ci95": {}, "diff_vs_stock": {}}
for k, m in G.items():
    idx0 = np.where(m)[0]; boots = [rng.choice(idx0, len(idx0)) for _ in range(B)]
    st = np.array([f1(M["stock"], b) for b in boots])
    for mod, P in M.items():
        out["macro_f1"].setdefault(mod, {})[k] = f1(P, idx0)
        bs = st if mod == "stock" else np.array([f1(P, b) for b in boots])
        out["ci95"].setdefault(mod, {})[k] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        if mod != "stock":
            d = bs - st
            out["diff_vs_stock"].setdefault(mod, {})[k] = {"diff": f1(P, idx0) - f1(M["stock"], idx0), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
(HERE / "ticker_sensitivity.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
