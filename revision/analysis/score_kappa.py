"""Agreement between a completed check sheet and the LLM benchmark labels (rubric v1 and v2).

usage: score_kappa.py CHECK_CSV OUT_JSON [--show-disagreements]
Kappa is computed on items where both sides gave positive/negative/neutral; excludes are counted separately.
CI = percentile bootstrap over items (2000 resamples, seed 20261020). Pure stdlib.
"""
import csv, json, sys, random, hashlib
from collections import Counter

CLASSES = ("positive", "neutral", "negative")
BENCH = "analysis/benchmark_v2_final.json"
V2 = "analysis/benchmark_v2_labels_v2.json"


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in CLASSES) / n ** 2
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_ci(pairs, reps=2000, seed=20261020):
    rng = random.Random(seed)
    ks = sorted(kappa([rng.choice(pairs) for _ in pairs])[1] for _ in range(reps))
    return ks[int(0.025 * reps)], ks[int(0.975 * reps) - 1]


def score(dom, llm, ids):
    both = [i for i in ids if dom[i] in CLASSES and llm[i] in CLASSES]
    pairs = [(dom[i], llm[i]) for i in both]
    po, k = kappa(pairs)
    lo, hi = boot_ci(pairs)
    cm = {d: {l: sum(1 for a, b in pairs if a == d and b == l) for l in CLASSES} for d in CLASSES}
    return {
        "n_scored": len(pairs), "agreement": round(po, 4), "kappa": round(k, 4),
        "kappa_ci95": [round(lo, 3), round(hi, 3)],
        "excluded_by_check": sum(dom[i] == "exclude" for i in ids),
        "excluded_by_llm": sum(llm[i] == "exclude" for i in ids),
        "confusion_rows_check_cols_llm": cm,
        "disagreements": [{"item": i, "check": dom[i], "llm": llm[i]} for i in both if dom[i] != llm[i]],
    }


def main():
    path, out = sys.argv[1], sys.argv[2]
    rows = list(csv.reader(open(path, newline="")))
    dom = {r[0]: r[-1].strip().lower() for r in rows[1:]}
    bad = {i: v for i, v in dom.items() if v not in CLASSES + ("exclude",)}
    assert not bad, f"unexpected labels: {bad}"
    v1 = {b["item"]: b["llm_label"] for b in json.load(open(BENCH))}
    v2meta = json.load(open(V2))
    v2 = {i: d["label"] for i, d in v2meta["labels"].items()}
    ids = list(dom)
    assert set(ids) <= set(v1) and set(ids) <= set(v2), "check items missing from benchmark"
    res = {
        "check_file": path, "check_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        "n_items": len(ids), "check_label_counts": dict(Counter(dom.values())),
        "v1": score(dom, v1, ids), "v2": score(dom, v2, ids),
        "v2_labeller": v2meta["meta"]["labeller"], "v2_sha256": hashlib.sha256(open(V2, "rb").read()).hexdigest(),
    }
    json.dump(res, open(out, "w"), indent=1)
    for k in ("v1", "v2"):
        r = res[k]
        print(f"{k}: n={r['n_scored']} agree={r['agreement']:.3f} kappa={r['kappa']:.3f} "
              f"CI95={r['kappa_ci95']} excl(check)={r['excluded_by_check']} excl(llm)={r['excluded_by_llm']}")
        for d in CLASSES:
            print(f"   check={d:8s} -> llm " + " ".join(f"{l[:3]}={r['confusion_rows_check_cols_llm'][d][l]:2d}" for l in CLASSES))
        if "--show-disagreements" in sys.argv:
            print("   " + Counter(f"{x['check'][:3]}->{x['llm'][:3]}" for x in r["disagreements"]).most_common().__repr__())


if __name__ == "__main__":
    main()
