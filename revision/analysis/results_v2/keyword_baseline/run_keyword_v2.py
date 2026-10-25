"""Keyword-lexicon baseline (app/models/keyword_sentiment.py, unmodified) on the frozen 338-item v2 benchmark.
Read-only on inputs; writes keyword_v2.json here. Bootstrap vs stock mirrors stats_v2.py (B=10000, seed 20261024)."""
import json, sys, importlib.util, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("ks", ROOT/"investor-sentiment-dashboard/backend/app/models/keyword_sentiment.py")
ks = importlib.util.module_from_spec(spec); spec.loader.exec_module(ks)
sys.path.insert(0, str(ROOT/"analysis/results_v2/stats")); from stats_v2 import macro_f1 as _mf, B, RNG_SEED, LABELS
_I={l:i for i,l in enumerate(LABELS)}
macro_f1=lambda g,p: _mf(np.vectorize(_I.get)(g), np.vectorize(_I.get)(p))
items = json.loads((ROOT/"analysis/benchmark_v2_frozen.json").read_text())["items"]
g = np.array([i["label"] for i in items]); pk = np.array([ks.calculate_financial_sentiment(i["text"])[0] for i in items])
stock = json.loads((ROOT/"analysis/results_v1/runs/stock.json").read_text())["benchmark_max_length_128"]["predictions"]
ps = np.array([r["pred"] for r in stock]); assert list(g) == [r["gold"] for r in stock]
L = ["positive","negative","neutral"]
per = {c: dict(precision=float(np.mean(g[pk==c]==c)) if (pk==c).any() else 0.0, recall=float(np.mean(pk[g==c]==c)), predicted=int((pk==c).sum())) for c in L}
rng = np.random.default_rng(RNG_SEED); idx = rng.integers(0, len(g), size=(B, len(g))); gb = g[idx]
d = macro_f1(gb, pk[idx]) - macro_f1(gb, ps[idx]); obs = float(macro_f1(g, pk) - macro_f1(g, ps))
p = float(min(1.0, 2*min(np.mean(d <= 0), np.mean(d >= 0))))
out = dict(n_items=len(g), macro_f1=float(macro_f1(g, pk)), accuracy=float(np.mean(g==pk)), per_class=per,
           bootstrap_keyword_minus_stock=dict(diff=obs, ci95=[float(np.percentile(d,2.5)), float(np.percentile(d,97.5))], p_two_sided=p, B=B, seed=RNG_SEED),
           predictions=[dict(id=i.get("id"), gold=x, pred=y) for i,x,y in zip(items,g,pk)])
(Path(__file__).parent/"keyword_v2.json").write_text(json.dumps(out, indent=1))
print({k:v for k,v in out.items() if k!="predictions"})
