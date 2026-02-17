"""
Evaluation metrics for sentiment analysis models.

Computes accuracy, precision, recall, F1 (macro and per-class),
and generates confusion matrices.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

LABELS = ["positive", "negative", "neutral"]


def _confusion_matrix(
    true_labels: List[str], pred_labels: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Build a confusion matrix as a nested dict.

    Returns:
        {actual_label: {predicted_label: count}}
    """
    matrix: Dict[str, Dict[str, int]] = {
        actual: {pred: 0 for pred in LABELS} for actual in LABELS
    }
    for true, pred in zip(true_labels, pred_labels):
        if true in LABELS and pred in LABELS:
            matrix[true][pred] += 1
    return matrix


def _per_class_metrics(
    true_labels: List[str], pred_labels: List[str], label: str
) -> Dict[str, float]:
    """Compute precision, recall, F1 for a single class."""
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": tp + fn,
    }


def evaluate_predictions(
    true_labels: List[str], pred_labels: List[str]
) -> Dict:
    """
    Full evaluation of predicted vs true labels.

    Args:
        true_labels: Ground-truth sentiment labels.
        pred_labels: Predicted sentiment labels.

    Returns:
        Dictionary with accuracy, per-class metrics, macro averages,
        and confusion matrix.
    """
    assert len(true_labels) == len(pred_labels), (
        f"Length mismatch: {len(true_labels)} true vs {len(pred_labels)} pred"
    )

    n = len(true_labels)
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / n if n > 0 else 0.0

    per_class = {}
    for label in LABELS:
        per_class[label] = _per_class_metrics(true_labels, pred_labels, label)

    macro_precision = sum(m["precision"] for m in per_class.values()) / len(LABELS)
    macro_recall = sum(m["recall"] for m in per_class.values()) / len(LABELS)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(LABELS)

    cm = _confusion_matrix(true_labels, pred_labels)

    return {
        "total_samples": n,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def format_report(results: Dict, model_name: str = "Model") -> str:
    """Format evaluation results as a human-readable report string."""
    lines = [
        f"\n{'=' * 60}",
        f"  Evaluation Report: {model_name}",
        f"{'=' * 60}",
        f"  Samples:   {results['total_samples']}",
        f"  Accuracy:  {results['accuracy']:.2%}",
        f"  Macro F1:  {results['macro_f1']:.2%}",
        f"  Macro P:   {results['macro_precision']:.2%}",
        f"  Macro R:   {results['macro_recall']:.2%}",
        f"\n  Per-Class Metrics:",
        f"  {'Label':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}",
        f"  {'-' * 56}",
    ]

    for label in LABELS:
        m = results["per_class"][label]
        lines.append(
            f"  {label:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} "
            f"{m['f1']:<12.4f} {m['support']:<10}"
        )

    lines.append(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    lines.append(f"  {'':>12} {'positive':>10} {'negative':>10} {'neutral':>10}")
    cm = results["confusion_matrix"]
    for actual in LABELS:
        row = cm[actual]
        lines.append(
            f"  {actual:>12} {row['positive']:>10} {row['negative']:>10} {row['neutral']:>10}"
        )

    lines.append(f"{'=' * 60}\n")
    return "\n".join(lines)
