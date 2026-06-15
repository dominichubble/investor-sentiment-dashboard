"""
Paired significance testing and bootstrap confidence intervals.

These routines back the statistical claims made in the manuscript
(McNemar p-value, stratified-bootstrap CIs). The repository previously
contained only the *outputs* of these computations (mcnemar_results.json,
benchmark_results.json); this module makes them reproducible from raw
prediction lists.

Dependencies: scipy, numpy (already required transitively by statsmodels).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

LABELS = ["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# McNemar's test (paired classifier comparison)
# ---------------------------------------------------------------------------
def mcnemar_test(
    true_labels: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
) -> Dict[str, float]:
    """
    McNemar's test for the difference in error rate between two classifiers
    evaluated on the *same* samples.

    Args:
        true_labels: Ground-truth labels.
        pred_a: Predictions from classifier A (e.g. the keyword baseline).
        pred_b: Predictions from classifier B (e.g. FinBERT).

    Returns:
        Dict with the discordant-pair counts, the exact two-sided binomial
        p-value, and the continuity-corrected chi-square statistic with its
        approximate p-value. Naming mirrors mcnemar_results.json:
        b = "A correct, B wrong"; c = "B correct, A wrong".
    """
    if not (len(true_labels) == len(pred_a) == len(pred_b)):
        raise ValueError("true_labels, pred_a and pred_b must be the same length")

    both_correct = both_wrong = b = c = 0
    for t, a, pb in zip(true_labels, pred_a, pred_b):
        a_ok, b_ok = (a == t), (pb == t)
        if a_ok and b_ok:
            both_correct += 1
        elif not a_ok and not b_ok:
            both_wrong += 1
        elif a_ok and not b_ok:
            b += 1
        else:  # b_ok and not a_ok
            c += 1

    n_discordant = b + c
    # Exact two-sided binomial p-value on the discordant pairs (H0: p = 0.5).
    if n_discordant > 0:
        p_exact = float(sp_stats.binomtest(min(b, c), n_discordant, 0.5).pvalue)
        chi2_cc = (abs(b - c) - 1) ** 2 / n_discordant if n_discordant > 0 else 0.0
    else:
        p_exact = 1.0
        chi2_cc = 0.0
    p_chi2 = float(sp_stats.chi2.sf(chi2_cc, df=1)) if n_discordant > 0 else 1.0

    return {
        "n_discordant": n_discordant,
        "b_only_A_correct": b,
        "c_only_B_correct": c,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "p_value_exact": p_exact,
        "chi2_cc": chi2_cc,
        "p_value_chi2": p_chi2,
    }


# ---------------------------------------------------------------------------
# Metric helpers usable as bootstrap statistics
# ---------------------------------------------------------------------------
def accuracy(true_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
    """Plain accuracy."""
    if not true_labels:
        return 0.0
    return sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)


def macro_f1(true_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
    """Macro-averaged F1 over the three sentiment classes."""
    f1s = []
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(true_labels, pred_labels))
        fp = sum(t != label and p == label for t, p in zip(true_labels, pred_labels))
        fn = sum(t == label and p != label for t, p in zip(true_labels, pred_labels))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(LABELS)


def class_recall(label: str) -> Callable[[Sequence[str], Sequence[str]], float]:
    """Return a recall statistic for a single class (e.g. the neutral class)."""

    def _recall(true_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
        tp = sum(t == label and p == label for t, p in zip(true_labels, pred_labels))
        fn = sum(t == label and p != label for t, p in zip(true_labels, pred_labels))
        return tp / (tp + fn) if (tp + fn) else 0.0

    return _recall


# ---------------------------------------------------------------------------
# Stratified bootstrap confidence intervals
# ---------------------------------------------------------------------------
def stratified_bootstrap_ci(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    statistic: Callable[[Sequence[str], Sequence[str]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Non-parametric confidence interval for a classification statistic, using
    a stratified bootstrap that resamples *within* each true class so the
    positive/negative/neutral prevalence of the original sample is preserved.

    Args:
        true_labels: Ground-truth labels.
        pred_labels: Predicted labels (same order/length as true_labels).
        statistic:  Function (true, pred) -> float, e.g. ``accuracy`` or
                    ``macro_f1`` from this module.
        n_boot:     Number of bootstrap replicates.
        alpha:      Significance level; a 95% CI corresponds to alpha=0.05.
        seed:       RNG seed for reproducibility.

    Returns:
        Dict with the point estimate and the lower/upper percentile bounds.
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels and pred_labels must be the same length")

    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(pred_labels)
    rng = np.random.default_rng(seed)

    # Pre-compute the row indices belonging to each class.
    class_indices = {lab: np.where(true_arr == lab)[0] for lab in set(true_arr)}

    point = float(statistic(list(true_arr), list(pred_arr)))
    replicates = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        sampled = []
        for idx in class_indices.values():
            if len(idx) == 0:
                continue
            sampled.append(rng.choice(idx, size=len(idx), replace=True))
        boot_idx = np.concatenate(sampled)
        replicates[k] = statistic(
            list(true_arr[boot_idx]), list(pred_arr[boot_idx])
        )

    lower = float(np.percentile(replicates, 100 * (alpha / 2)))
    upper = float(np.percentile(replicates, 100 * (1 - alpha / 2)))
    return {
        "point_estimate": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_boot": n_boot,
        "alpha": alpha,
    }


def summarise_runs(values: List[float]) -> Dict[str, float]:
    """Mean / SD / min / max helper for multi-seed aggregation."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }
