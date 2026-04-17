"""
Compute Cohen's kappa between the author labels (labelled_dataset) and an
independent second annotator whose labels live in annotation_llm.json.

This script is explicit about its methodology: the second annotator is an
LLM that was given only the 200 raw texts (no labels, no gold dataset
visibility). Agreement is reported alongside per-class breakdown so the
reader can see where disagreement concentrates.

Output: backend/data/evaluation/kappa_results.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.evaluation.labeled_dataset import get_labeled_dataset  # noqa: E402


LABELS = ("positive", "negative", "neutral")


def cohen_kappa(y1: list[str], y2: list[str]) -> dict:
    n = len(y1)
    assert n == len(y2) and n > 0

    observed = sum(1 for a, b in zip(y1, y2) if a == b) / n

    c1 = Counter(y1)
    c2 = Counter(y2)
    expected = sum((c1[l] / n) * (c2[l] / n) for l in LABELS)

    if expected == 1.0:
        kappa = 1.0
    else:
        kappa = (observed - expected) / (1.0 - expected)

    # Standard error of kappa under the independence assumption.
    # se = sqrt( observed*(1-observed) / (n * (1-expected)^2) )
    import math
    if expected < 1.0:
        se = math.sqrt(observed * (1 - observed) / (n * (1 - expected) ** 2))
    else:
        se = 0.0

    z = kappa / se if se > 0 else 0.0
    # two-sided p-value against null kappa=0
    from statistics import NormalDist
    p = 2 * (1 - NormalDist().cdf(abs(z)))

    return {
        "n": n,
        "observed_agreement": round(observed, 4),
        "expected_agreement": round(expected, 4),
        "kappa": round(kappa, 4),
        "kappa_se": round(se, 4),
        "z": round(z, 4),
        "p_value": round(p, 6),
    }


def per_class_agreement(y1: list[str], y2: list[str]) -> dict:
    out = {}
    for l in LABELS:
        idxs = [i for i, y in enumerate(y1) if y == l]
        if not idxs:
            out[l] = {"n": 0, "agree": 0, "agreement": None}
            continue
        agree = sum(1 for i in idxs if y2[i] == l)
        out[l] = {
            "n": len(idxs),
            "agree": agree,
            "agreement": round(agree / len(idxs), 4),
        }
    return out


def confusion(y1: list[str], y2: list[str]) -> dict:
    """Confusion with y1 as rows (author) and y2 as cols (second annotator)."""
    table = {l: {m: 0 for m in LABELS} for l in LABELS}
    for a, b in zip(y1, y2):
        table[a][b] += 1
    return table


def main() -> None:
    ds = get_labeled_dataset()
    author = [row["label"] for row in ds]

    llm_path = REPO_ROOT / "backend" / "data" / "evaluation" / "annotation_llm.json"
    with open(llm_path, "r", encoding="utf-8") as fh:
        llm_raw = json.load(fh)
    llm_raw.sort(key=lambda r: r["idx"])
    llm = [r["label"] for r in llm_raw]

    assert len(author) == len(llm), f"author={len(author)} llm={len(llm)}"

    result = cohen_kappa(author, llm)
    result["per_class_agreement"] = per_class_agreement(author, llm)
    result["confusion_author_rows_vs_llm_cols"] = confusion(author, llm)
    result["methodology"] = (
        "Second annotator is an LLM (Claude) given only the 200 raw texts via "
        "annotation_task.json, with no access to the gold labels or the labelled "
        "dataset source file. Labels restricted to {positive,negative,neutral}. "
        "Reported as inter-annotator reliability between the author (primary "
        "annotator) and an independent automated second annotator."
    )

    out_path = REPO_ROOT / "backend" / "data" / "evaluation" / "kappa_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
