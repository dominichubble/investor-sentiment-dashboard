"""R1-1 / R2-7 / R2-1: per-channel benchmark description, per-channel macro-F1 and neutral-error flows. Existing data only."""
import json, sys, collections as C, numpy as np
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
kp = kw.get("predictions") or kw.get("item_predictions")
if kp and isinstance(kp, list) and isinstance(kp[0], dict) and "pred" in kp[0] and [r.get("id") for r in kp] == ids: M["keyword"] = [load(kp)]
ch = np.array([x["channel"] for x in items]); groups = ["all"] + sorted(set(ch))
def mask(gr): return np.ones(len(g), bool) if gr == "all" else ch == gr
desc = {}
for gr in groups:
    m = mask(gr); sub = [x for x, k in zip(items, m) if k]; ts = sorted(x["published_at"] for x in sub if x["published_at"])
    tk = [x["ticker"] for x in sub]; noticker = sum(1 for t in tk if not t or str(t).strip() in ("", "None", "null"))
    tc = C.Counter(t for t in tk if t and str(t).strip() not in ("", "None", "null"))
    desc[gr] = dict(n=len(sub), labels={l: int((g[m] == L[l]).sum()) for l in LABELS}, ts_min=ts[0] if ts else None, ts_max=ts[-1] if ts else None,
                    n_missing_ts=len(sub) - len(ts), n_distinct_tickers=len(tc), share_no_ticker=noticker / len(sub),
                    top5_tickers=tc.most_common(5), share_top5=sum(v for _, v in tc.most_common(5)) / len(sub), share_top1=(tc.most_common(1)[0][1] / len(sub)) if tc else 0,
                    sub_sources=C.Counter(x["source"] for x in sub).most_common(), low_information=sum(x["low_information"] for x in sub))
f1 = {mod: {gr: float(np.mean([macro_f1(g[mask(gr)], p[mask(gr)]) for p in P])) for gr in groups} for mod, P in M.items()}
f1_sd = {mod: {gr: float(np.std([macro_f1(g[mask(gr)], p[mask(gr)]) for p in P], ddof=1)) if len(P) > 1 else None for gr in groups} for mod, P in M.items()}
flows = {}
for mod, P in M.items():
    f = {}
    for gl in LABELS:
        for pl in LABELS:
            if gl != pl and "neutral" in (gl, pl): f[f"{gl}->{pl}"] = float(np.mean([((g == L[gl]) & (p == L[pl])).sum() for p in P]))
    f["neutral_correct"] = float(np.mean([((g == 2) & (p == 2)).sum() for p in P])); f["pos<->neg"] = float(np.mean([(((g == 0) & (p == 1)) | ((g == 1) & (p == 0))).sum() for p in P]))
    flows[mod] = f
out = dict(note="'source' grouped by 'channel' (3 values); 39 fine-grained 'source' values listed under sub_sources. Timestamps are 'published_at' as stored.",
           models_included=list(M), keyword_item_predictions_available="keyword" in M, description=desc, macro_f1_by_channel=f1, macro_f1_sd_over_seeds=f1_sd, error_flows_counts=flows,
           gold_counts={l: int((g == L[l]).sum()) for l in LABELS})
(HERE / "partial_items.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out, indent=1, default=str))
