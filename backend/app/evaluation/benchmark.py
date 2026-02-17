"""
Benchmark script comparing keyword-based vs FinBERT sentiment analysis.

Usage:
    cd backend
    python -m app.evaluation.benchmark              # keyword only (fast)
    python -m app.evaluation.benchmark --finbert     # include FinBERT comparison
    python -m app.evaluation.benchmark --all         # both approaches side-by-side
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

from .labeled_dataset import get_dataset_stats, get_labeled_dataset
from .metrics import evaluate_predictions, format_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _run_keyword_predictions(texts: List[str]) -> Tuple[List[str], float]:
    """Run keyword-based sentiment on all texts. Returns (labels, elapsed_seconds)."""
    from app.models.keyword_sentiment import calculate_financial_sentiment

    start = time.time()
    labels = []
    for text in texts:
        label, _score = calculate_financial_sentiment(text)
        labels.append(label)
    elapsed = time.time() - start
    return labels, elapsed


def _run_finbert_predictions(texts: List[str]) -> Tuple[List[str], float]:
    """Run FinBERT sentiment on all texts. Returns (labels, elapsed_seconds)."""
    from app.models.finbert_model import get_model

    logger.info("Loading FinBERT model...")
    model = get_model()
    logger.info(f"FinBERT running on {model.device}")

    start = time.time()
    results = model.predict_batch(texts, batch_size=32)
    labels = [r["label"] for r in results]
    elapsed = time.time() - start
    return labels, elapsed


def run_benchmark(
    include_keyword: bool = True,
    include_finbert: bool = False,
) -> Dict:
    """
    Run benchmark comparison.

    Args:
        include_keyword: Evaluate keyword-based approach.
        include_finbert: Evaluate FinBERT approach.

    Returns:
        Dictionary with evaluation results for each approach.
    """
    dataset = get_labeled_dataset()
    stats = get_dataset_stats()

    texts = [item["text"] for item in dataset]
    true_labels = [item["label"] for item in dataset]

    logger.info(
        f"Dataset: {stats['total']} samples "
        f"(pos={stats['positive']}, neg={stats['negative']}, neu={stats['neutral']})"
    )

    results = {}

    if include_keyword:
        logger.info("\nRunning keyword-based sentiment analysis...")
        keyword_preds, keyword_time = _run_keyword_predictions(texts)
        keyword_eval = evaluate_predictions(true_labels, keyword_preds)
        keyword_eval["inference_time_seconds"] = round(keyword_time, 3)
        results["keyword"] = keyword_eval

        report = format_report(keyword_eval, "Keyword Lexicon")
        logger.info(report)

    if include_finbert:
        logger.info("\nRunning FinBERT sentiment analysis...")
        finbert_preds, finbert_time = _run_finbert_predictions(texts)
        finbert_eval = evaluate_predictions(true_labels, finbert_preds)
        finbert_eval["inference_time_seconds"] = round(finbert_time, 3)
        results["finbert"] = finbert_eval

        report = format_report(finbert_eval, "FinBERT")
        logger.info(report)

    if include_keyword and include_finbert:
        logger.info("\n" + "=" * 60)
        logger.info("  COMPARISON SUMMARY")
        logger.info("=" * 60)
        kw = results["keyword"]
        fb = results["finbert"]
        logger.info(f"  {'Metric':<20} {'Keyword':<15} {'FinBERT':<15} {'Delta':<10}")
        logger.info(f"  {'-' * 55}")

        for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
            kw_val = kw[metric]
            fb_val = fb[metric]
            delta = fb_val - kw_val
            sign = "+" if delta > 0 else ""
            logger.info(
                f"  {metric:<20} {kw_val:<15.4f} {fb_val:<15.4f} {sign}{delta:.4f}"
            )

        logger.info(
            f"  {'speed (sec)':<20} {kw['inference_time_seconds']:<15.3f} "
            f"{fb['inference_time_seconds']:<15.3f}"
        )
        logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sentiment analysis approaches."
    )
    parser.add_argument(
        "--finbert",
        action="store_true",
        help="Include FinBERT in the benchmark.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both keyword and FinBERT benchmarks.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to a JSON file.",
    )
    args = parser.parse_args()

    include_keyword = True
    include_finbert = args.finbert or args.all

    results = run_benchmark(
        include_keyword=include_keyword,
        include_finbert=include_finbert,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
