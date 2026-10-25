"""Significance tests for the v2 fine-tuning runs (plan.md T11 subset). numpy/scipy only.
Inputs (read-only): analysis/results_v2/runs/arm_{a,b,c}_seed*.json, analysis/results_v1/runs/stock.json (max_length 128).
Outputs: stats_v2.json, stats_v2.md in this directory.
"""
import json, itertools
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "analysis/results_v2/runs"; OUTD = Path(__file__).resolve().parent
LABELS = ["positive", "negative", "neutral"]; SEEDS = [13, 21, 42, 84, 168]; ARMS = ["arm_a", "arm_b", "arm_c"]
B = 10000; RNG_SEED = 20261024


def macro_f1(g, p):
    f = []
    for l in range(3):
        tp = np.sum((g == l) & (p == l), axis=-1); fp = np.sum((g != l) & (p == l), axis=-1); fn = np.sum((g == l) & (p != l), axis=-1)
        d = 2 * tp + fp + fn; f.append(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0))
    return np.mean(f, axis=0)


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def ci_t(x):
    x = np.asarray(x); m = x.mean(); sd = x.std(ddof=1); h = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))
    return dict(mean=m, sd=sd, ci95=[m - h, m + h])


def mcnemar_exact(g, a, b):
    ca, cb = (a == g), (b == g); n01 = int(np.sum(ca & ~cb)); n10 = int(np.sum(~ca & cb))
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(arm_right_stock_wrong=n01, arm_wrong_stock_right=n10, p=float(p))


def main():
    L = {l: i for i, l in enumerate(LABELS)}
    stock = json.loads((ROOT / "analysis/results_v1/runs/stock.json").read_text())["benchmark_max_length_128"]["predictions"]
    ids = [r["id"] for r in stock]; g = np.array([L[r["gold"]] for r in stock]); ps = np.array([L[r["pred"]] for r in stock])
    runs, P = {}, {}
    for a in ARMS:
        for s in SEEDS:
            r = json.loads((RUNS / f"{a}_seed{s}.json").read_text())
            assert [x["id"] for x in r["benchmark_predictions"]] == ids
            assert np.array_equal(np.array([L[x["gold"]] for x in r["benchmark_predictions"]]), g)
            runs[(a, s)] = r; P[(a, s)] = np.array([L[x["pred"]] for x in r["benchmark_predictions"]])
    out = {"n_items": len(g), "stock": {"macro_f1": float(macro_f1(g, ps)), "accuracy": float(np.mean(g == ps)),
           "recall": {l: float(np.mean(ps[g == i] == i)) for i, l in enumerate(LABELS)}}, "per_run": [], "per_arm": {}}
    for (a, s), r in runs.items():
        out["per_run"].append(dict(arm=a, seed=s, best_epoch=r["best_epoch"], val_macro_f1=r["val"]["macro_f1"],
                                   macro_f1=r["benchmark"]["macro_f1"], accuracy=r["benchmark"]["accuracy"],
                                   recall={l: r["benchmark"]["per_class"][l]["recall"] for l in LABELS},
                                   pred_counts=r["benchmark"]["pred_counts"], mcnemar_vs_stock=mcnemar_exact(g, P[(a, s)], ps)))
    for a in ARMS:
        rr = [x for x in out["per_run"] if x["arm"] == a]
        out["per_arm"][a] = {"macro_f1": ci_t([x["macro_f1"] for x in rr]), "accuracy": ci_t([x["accuracy"] for x in rr]),
                             **{f"recall_{l}": ci_t([x["recall"][l] for x in rr]) for l in LABELS}}
    # arm vs arm by seed
    pair = []
    for a, b in itertools.combinations(ARMS, 2):
        for met in ("macro_f1", "accuracy"):
            xa = np.array([runs[(a, s)]["benchmark"][met] for s in SEEDS]); xb = np.array([runs[(b, s)]["benchmark"][met] for s in SEEDS])
            d = xa - xb
            pair.append(dict(comparison=f"{a} vs {b}", metric=met, mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)),
                             t_p=float(stats.ttest_rel(xa, xb).pvalue), wilcoxon_p=float(stats.wilcoxon(xa, xb, method="exact").pvalue)))
    # item-level paired bootstrap on seed-averaged macro-F1 (mean of the 5 per-seed macro-F1s on each resample)
    rng = np.random.default_rng(RNG_SEED); idx = rng.integers(0, len(g), size=(B, len(g)))
    gb = g[idx]
    mf = {a: np.mean([macro_f1(gb, P[(a, s)][idx]) for s in SEEDS], axis=0) for a in ARMS}
    mf["stock"] = macro_f1(gb, ps[idx])
    obs = {a: float(np.mean([macro_f1(g, P[(a, s)]) for s in SEEDS])) for a in ARMS}; obs["stock"] = out["stock"]["macro_f1"]
    boot = []
    for a, b in [(x, "stock") for x in ARMS] + list(itertools.combinations(ARMS, 2)):
        d = mf[a] - mf[b]; od = obs[a] - obs[b]
        p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
        boot.append(dict(comparison=f"{a} vs {b}", observed_diff=od, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                         p_two_sided=p, p_floor=1 / B))
    # Holm families
    fam_main = [x for x in pair if x["metric"] == "macro_f1"]
    for k, adj in (("t_p", "t_p_holm"), ("wilcoxon_p", "wilcoxon_p_holm")):
        for x, h in zip(fam_main, holm([x[k] for x in fam_main])): x[adj] = h
    fam_acc = [x for x in pair if x["metric"] == "accuracy"]
    for k, adj in (("t_p", "t_p_holm"), ("wilcoxon_p", "wilcoxon_p_holm")):
        for x, h in zip(fam_acc, holm([x[k] for x in fam_acc])): x[adj] = h
    for x, h in zip(boot, holm([x["p_two_sided"] for x in boot])): x["p_holm"] = h
    mc = [x["mcnemar_vs_stock"] for x in out["per_run"]]
    for x, h in zip(mc, holm([x["p"] for x in mc])): x["p_holm_15"] = h
    out.update(seed_paired=pair, bootstrap=boot, settings=dict(bootstrap_B=B, rng_seed=RNG_SEED,
               holm_families=["seed-paired macro-F1 (3 comparisons) per test", "seed-paired accuracy (3) per test",
                              "item bootstrap seed-averaged macro-F1 (6)", "per-run exact McNemar vs stock (15)"],
               note="n=5 seeds: exact two-sided Wilcoxon minimum p = 0.0625, so it cannot reach 0.05; t-test low power."))
    (OUTD / "stats_v2.json").write_text(json.dumps(out, indent=1, default=float))
    f = lambda v: f"{v:.3f}"
    md = ["# v2 fine-tuning: significance tests (benchmark_v2_frozen, n=338)", "",
          f"Stock FinBERT (max_length 128): macro-F1 {f(out['stock']['macro_f1'])}, accuracy {f(out['stock']['accuracy'])}, "
          f"recall pos/neg/neu {'/'.join(f(out['stock']['recall'][l]) for l in LABELS)}", "",
          "## Per arm (5 seeds; mean ± sd [95% t-CI])", "", "| arm | macro-F1 | accuracy | recall pos | recall neg | recall neu |", "|---|---|---|---|---|---|"]
    for a in ARMS:
        q = out["per_arm"][a]; c = lambda k: f"{f(q[k]['mean'])} ± {f(q[k]['sd'])} [{f(q[k]['ci95'][0])}, {f(q[k]['ci95'][1])}]"
        md.append(f"| {a} | {c('macro_f1')} | {c('accuracy')} | {c('recall_positive')} | {c('recall_negative')} | {c('recall_neutral')} |")
    md += ["", "## Per run", "", "| arm | seed | best ep | val mF1 | mF1 | acc | rec pos | rec neg | rec neu | McNemar vs stock (arm✓stock✗ / arm✗stock✓, p, Holm-15 p) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for x in out["per_run"]:
        m = x["mcnemar_vs_stock"]
        md.append(f"| {x['arm']} | {x['seed']} | {x['best_epoch']} | {f(x['val_macro_f1'])} | {f(x['macro_f1'])} | {f(x['accuracy'])} | "
                  + " | ".join(f(x["recall"][l]) for l in LABELS) + f" | {m['arm_right_stock_wrong']}/{m['arm_wrong_stock_right']}, {m['p']:.4f}, {m['p_holm_15']:.4f} |")
    md += ["", "## Seed-paired arm comparisons (n=5)", "", "| comparison | metric | mean diff | sd | t p | t p Holm | Wilcoxon p | Wilcoxon p Holm |", "|---|---|---|---|---|---|---|---|"]
    for x in pair:
        md.append(f"| {x['comparison']} | {x['metric']} | {x['mean_diff']:+.4f} | {f(x['sd_diff'])} | {x['t_p']:.4f} | {x['t_p_holm']:.4f} | {x['wilcoxon_p']:.4f} | {x['wilcoxon_p_holm']:.4f} |")
    md += ["", f"## Item-level paired bootstrap, seed-averaged macro-F1 (B={B}, seed {RNG_SEED})", "", "| comparison | diff | 95% CI | p | p Holm (6) |", "|---|---|---|---|---|"]
    for x in boot:
        md.append(f"| {x['comparison']} | {x['observed_diff']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | {x['p_two_sided']:.4f} | {x['p_holm']:.4f} |")
    md += ["", "Notes: " + out["settings"]["note"] + " p=0.0000 in bootstrap means < 1/B."]
    (OUTD / "stats_v2.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
