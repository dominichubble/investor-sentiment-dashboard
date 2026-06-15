"""
External-comparability evaluation on the Financial PhraseBank benchmark.

Addresses the reviewer concern that the headline results are reported only on
a purpose-built 200-sample benchmark. This script runs the *same* two
classifiers (FinBERT and the keyword baseline) on the public Financial
PhraseBank dataset (Malo et al., 2014), so the manuscript can report a row that
is directly comparable to prior work, alongside a paired McNemar test and
stratified-bootstrap confidence intervals.

The script does NOT fabricate numbers: it computes them when run against the
real dataset and the real model wrappers.

Usage (from the backend directory):
    python -m app.evaluation.phrasebank_eval --finbert \
        --config sentences_allagree \
        --output data/evaluation/phrasebank_results.json

Config options (increasing size / decreasing annotator agreement):
    sentences_allagree  (~2264 sentences, 100% annotator agreement)
    sentences_75agree
    sentences_66agree
    sentences_50agree   (~4846 sentences)

Dependencies: datasets, scipy, numpy, transformers, torch.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

from . import stats as st
from .metrics import evaluate_predictions, format_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_phrasebank(config: str) -> Tuple[List[str], List[str]]:
    """
    Load the Financial PhraseBank split and map its integer labels to the
    string labels used throughout this project (positive/negative/neutral).

    Returns:
        (texts, true_labels)
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from exc

    logger.info("Loading Financial PhraseBank config '%s' ...", config)
    ds = load_dataset("financial_phrasebank", config, split="train")

    # The dataset exposes the label names via its ClassLabel feature; resolve
    # them dynamically rather than hard-coding the integer mapping.
    label_names = ds.features["label"].names  # e.g. ['negative','neutral','positive']
    texts = [row["sentence"] for row in ds]
    true_labels = [label_names[row["label"]] for row in ds]
    logger.info("Loaded %d sentences.", len(texts))
    return texts, true_labels


def _run_keyword(texts: List[str]) -> Tuple[List[str], float]:
    from app.models.keyword_sentiment import calculate_financial_sentiment

    start = time.time()
    labels = [calculate_financial_sentiment(t)[0] for t in texts]
    return labels, time.time() - start


def _run_finbert(texts: List[str]) -> Tuple[List[str], float]:
    from app.models.finbert_model import get_model

    logger.info("Loading FinBERT ...")
    model = get_model()
    logger.info("FinBERT running on %s", model.device)
    start = time.time()
    preds = model.predict_batch(texts, batch_size=32)
    labels = [p["label"] for p in preds]
    return labels, time.time() - start


def run(config: str, include_finbert: bool, n_boot: int) -> Dict:
    texts, true_labels = _load_phrasebank(config)

    results: Dict = {"dataset": "financial_phrasebank", "config": config,
                     "n_samples": len(texts)}

    logger.info("Scoring keyword baseline ...")
    kw_preds, kw_time = _run_keyword(texts)
    kw_eval = evaluate_predictions(true_labels, kw_preds)
    kw_eval["inference_time_seconds"] = round(kw_time, 3)
    kw_eval["bootstrap_accuracy"] = st.stratified_bootstrap_ci(
        true_labels, kw_preds, st.accuracy, n_boot=n_boot
    )
    kw_eval["bootstrap_macro_f1"] = st.stratified_bootstrap_ci(
        true_labels, kw_preds, st.macro_f1, n_boot=n_boot
    )
    results["keyword"] = kw_eval
    logger.info(format_report(kw_eval, "Keyword Lexicon (PhraseBank)"))

    if include_finbert:
        logger.info("Scoring FinBERT ...")
        fb_preds, fb_time = _run_finbert(texts)
        fb_eval = evaluate_predictions(true_labels, fb_preds)
        fb_eval["inference_time_seconds"] = round(fb_time, 3)
        fb_eval["bootstrap_accuracy"] = st.stratified_bootstrap_ci(
            true_labels, fb_preds, st.accuracy, n_boot=n_boot
        )
        fb_eval["bootstrap_macro_f1"] = st.stratified_bootstrap_ci(
            true_labels, fb_preds, st.macro_f1, n_boot=n_boot
        )
        results["finbert"] = fb_eval
        logger.info(format_report(fb_eval, "FinBERT (PhraseBank)"))

        # Paired McNemar: keyword (A) vs FinBERT (B) on identical samples.
        results["mcnemar_keyword_vs_finbert"] = st.mcnemar_test(
            true_labels, kw_preds, fb_preds
        )
        logger.info("McNemar (keyword vs FinBERT): %s",
                    json.dumps(results["mcnemar_keyword_vs_finbert"], indent=2))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finbert", action="store_true",
                        help="Include FinBERT (downloads weights on first run).")
    parser.add_argument("--config", default="sentences_allagree",
                        help="Financial PhraseBank agreement config.")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap replicates for confidence intervals.")
    parser.add_argument("--output", default=None, help="Path to write results JSON.")
    args = parser.parse_args()

    results = run(args.config, include_finbert=args.finbert, n_boot=args.n_boot)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()
