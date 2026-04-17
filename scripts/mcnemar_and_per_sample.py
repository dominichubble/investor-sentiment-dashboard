"""
Run keyword + FinBERT predictions on the 200-sample benchmark, save per-sample
predictions to JSON, and compute McNemar's paired significance test for the
accuracy difference between the two classifiers.

Usage (from repo root):
    python -m scripts.mcnemar_and_per_sample

Outputs:
    backend/data/evaluation/per_sample_predictions.json
    backend/data/evaluation/mcnemar_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the backend package importable without installing it.
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.evaluation.labeled_dataset import get_labeled_dataset  # noqa: E402
from app.models.keyword_sentiment import calculate_financial_sentiment  # noqa: E402
from app.models.finbert_model import get_model  # noqa: E402


def mcnemar_exact(b: int, c: int) -> dict:
    """
    Exact McNemar (binomial) test on the discordant pairs.

    b = samples where classifier A is correct and B is wrong
    c = samples where classifier B is correct and A is wrong

    Under H0 (equal error rates) the discordant counts follow Binomial(n=b+c, p=0.5).
    Returns the two-sided p-value and the chi-square continuity-corrected statistic
    as a sanity check.
    """
    import math
    from statistics import NormalDist

    n = b + c
    if n == 0:
        return {"n_discordant": 0, "p_value_exact": 1.0, "chi2_cc": 0.0, "p_value_chi2": 1.0}

    # Two-sided exact binomial p-value.
    k = min(b, c)
    tail = 0.0
    for i in range(0, k + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    p_exact = min(1.0, 2.0 * tail)

    # Continuity-corrected chi-square, for cross-check.
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    # Survival of chi2 with df=1 via the normal approximation (z = sqrt(chi2)).
    z = math.sqrt(chi2)
    p_chi2 = 2 * (1 - NormalDist().cdf(z))

    return {
        "n_discordant": n,
        "b_only_A_correct": b,
        "c_only_B_correct": c,
        "p_value_exact": p_exact,
        "chi2_cc": chi2,
        "p_value_chi2": p_chi2,
    }


def main() -> None:
    dataset = get_labeled_dataset()
    texts = [row["text"] for row in dataset]
    truth = [row["label"] for row in dataset]

    # Keyword predictions.
    print(f"Running keyword baseline on {len(texts)} samples...")
    kw_preds = [calculate_financial_sentiment(t)[0] for t in texts]

    # FinBERT predictions.
    print("Loading FinBERT and running batched inference...")
    model = get_model()
    fb_raw = model.predict_batch(texts, batch_size=32)
    fb_preds = [r["label"] for r in fb_raw]

    # Per-sample record.
    per_sample = []
    for i, (t, y, k, f) in enumerate(zip(texts, truth, kw_preds, fb_preds)):
        per_sample.append(
            {
                "idx": i,
                "text": t,
                "truth": y,
                "keyword_pred": k,
                "finbert_pred": f,
                "keyword_correct": k == y,
                "finbert_correct": f == y,
            }
        )

    # Pairwise contingency: b = keyword-only-correct, c = finbert-only-correct.
    b = sum(1 for r in per_sample if r["keyword_correct"] and not r["finbert_correct"])
    c = sum(1 for r in per_sample if r["finbert_correct"] and not r["keyword_correct"])
    both_correct = sum(1 for r in per_sample if r["keyword_correct"] and r["finbert_correct"])
    both_wrong = sum(1 for r in per_sample if not r["keyword_correct"] and not r["finbert_correct"])

    mc = mcnemar_exact(b, c)
    mc["both_correct"] = both_correct
    mc["both_wrong"] = both_wrong
    mc["keyword_accuracy"] = sum(r["keyword_correct"] for r in per_sample) / len(per_sample)
    mc["finbert_accuracy"] = sum(r["finbert_correct"] for r in per_sample) / len(per_sample)

    out_dir = BACKEND / "data" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_sample_predictions.json", "w", encoding="utf-8") as fh:
        json.dump(per_sample, fh, indent=2)
    with open(out_dir / "mcnemar_results.json", "w", encoding="utf-8") as fh:
        json.dump(mc, fh, indent=2)

    print("\nResults:")
    print(json.dumps(mc, indent=2))


if __name__ == "__main__":
    main()
