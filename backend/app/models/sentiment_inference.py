"""
FinBERT Sentiment Inference Module

This module provides a high-level interface for sentiment analysis using FinBERT.
It handles text preprocessing, batch processing, and returns structured sentiment results.

Usage:
    from app.pipelines.sentiment_inference import analyze_sentiment, analyze_batch

    # Single text analysis
    result = analyze_sentiment("Stock market surged today")
    # Returns: {'label': 'positive', 'score': 0.95}

    # Batch analysis
    texts = ["Market up", "Crisis looming"]
    results = analyze_batch(texts)
"""

import logging
from typing import Dict, List, Optional, Union

from ..logging_config import FailedItemsTracker, log_exception
from .finbert_model import get_model

logger = logging.getLogger(__name__)


class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis errors."""

    pass


class TokenizationError(SentimentAnalysisError):
    """Exception raised when text tokenization fails."""

    pass


class ModelInferenceError(SentimentAnalysisError):
    """Exception raised when model inference fails."""

    pass


def analyze_sentiment(
    text: str, return_all_scores: bool = False
) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """
    Analyze sentiment of a single text input.

    Args:
        text: Raw text string to analyze
        return_all_scores: If True, include confidence scores for all labels

    Returns:
        Dictionary containing:
            - label: Sentiment label ('positive', 'negative', or 'neutral')
            - score: Confidence score for the predicted label (0-1)
            - scores: (optional) All label scores if return_all_scores=True

    Raises:
        ValueError: If text is empty or None
        RuntimeError: If model fails to load

    Example:
        >>> analyze_sentiment("Stock prices surged 10% today")
        {'label': 'positive', 'score': 0.92}

        >>> analyze_sentiment("Market outlook uncertain", return_all_scores=True)
        {
            'label': 'neutral',
            'score': 0.78,
            'scores': {'positive': 0.11, 'negative': 0.11, 'neutral': 0.78}
        }
    """
    if not isinstance(text, str):
        raise ValueError("Text input must be a non-empty string")

    if not text or not text.strip():
        raise ValueError("Text input cannot be blank")

    try:
        # Check text length
        text_length = len(text)
        if text_length > 10000:
            logger.warning(
                f"Text is very long ({text_length} chars), may cause issues. "
                "Consider truncating before analysis."
            )

        model = get_model()
        result = model.predict(text, return_all_scores=return_all_scores)
        logger.debug(f"Analyzed text: '{text[:50]}...' -> {result['label']}")
        return result
    except ValueError as e:
        # Tokenization or input validation errors
        logger.error(f"Input validation failed for text: {text[:100]}... Error: {e}")
        raise TokenizationError(f"Text tokenization failed: {e}") from e
    except RuntimeError as e:
        # Model loading or inference errors
        logger.error(f"Model inference failed: {e}")
        raise ModelInferenceError(f"Model inference failed: {e}") from e
    except Exception as e:
        # Catch-all for unexpected errors
        log_exception(logger, e, f"Unexpected error analyzing text: {text[:100]}...")
        raise SentimentAnalysisError(f"Failed to analyze sentiment: {e}") from e


def analyze_batch(
    texts: List[str],
    batch_size: int = 32,
    return_all_scores: bool = False,
    skip_errors: bool = True,
    track_failures: bool = True,
) -> tuple[
    List[Optional[Dict[str, Union[str, float, Dict[str, float]]]]],
    Optional[FailedItemsTracker],
]:
    """
    Analyze sentiment for multiple text inputs efficiently.

    Args:
        texts: List of text strings to analyze
        batch_size: Number of texts to process per batch (default: 32)
        return_all_scores: If True, include confidence scores for all labels
        skip_errors: If True, return None for failed analyses instead of raising (default: True)
        track_failures: If True, track failed items and return tracker (default: True)

    Returns:
        Tuple containing:
            - List of dictionaries, one per input text. Each dict contains:
                - label: Sentiment label ('positive', 'negative', or 'neutral')
                - score: Confidence score for the predicted label (0-1)
                - scores: (optional) All label scores if return_all_scores=True
              If skip_errors=True, failed analyses will be None in the list.
            - FailedItemsTracker object if track_failures=True, else None

    Raises:
        ValueError: If texts is empty or not a list
        RuntimeError: If batch processing fails (when skip_errors=False)

    Example:
        >>> texts = [
        ...     "Stock market rally continues",
        ...     "Company reports losses",
        ...     "Quarterly earnings meet expectations"
        ... ]
        >>> results = analyze_batch(texts)
        >>> [r['label'] for r in results]
        ['positive', 'negative', 'neutral']
    """
    if not texts or not isinstance(texts, list):
        raise ValueError("Texts must be a non-empty list")

    if not all(isinstance(t, str) for t in texts):
        raise ValueError("All items in texts must be strings")

    # Initialize failure tracker
    failures = FailedItemsTracker() if track_failures else None

    # Filter out empty texts and track them
    valid_indices = []
    valid_texts = []

    for i, t in enumerate(texts):
        if not t or not t.strip():
            if failures:
                failures.add_failure(
                    item=f"Index {i}: {str(t)[:100]}",
                    error_type="EmptyText",
                    error_message="Text is empty or whitespace only",
                )
            logger.warning(f"Skipping empty text at index {i}")
        else:
            valid_indices.append(i)
            valid_texts.append(t)

    if not valid_texts:
        logger.error("No valid (non-empty) texts provided")
        if skip_errors:
            return [None] * len(texts), failures
        else:
            raise ValueError("No valid (non-empty) texts provided")

    logger.info(f"Processing {len(valid_texts)}/{len(texts)} valid texts")

    try:
        model = get_model()

        # Process in batches with individual error handling
        all_results = []

        for batch_start in range(0, len(valid_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]

            logger.debug(
                f"Processing batch {batch_start//batch_size + 1}: "
                f"items {batch_start} to {batch_end-1}"
            )

            try:
                batch_results = model.predict_batch(
                    batch_texts,
                    batch_size=len(batch_texts),
                    return_all_scores=return_all_scores,
                )
                all_results.extend(batch_results)

            except Exception as batch_error:
                # Batch failed, try individual items if skip_errors is True
                logger.error(f"Batch processing failed: {batch_error}")

                if skip_errors:
                    logger.info("Attempting individual analysis for failed batch")
                    for idx, text in enumerate(batch_texts):
                        try:
                            result = analyze_sentiment(
                                text, return_all_scores=return_all_scores
                            )
                            all_results.append(result)
                        except Exception as item_error:
                            all_results.append(None)
                            if failures:
                                failures.add_failure(
                                    item=f"Index {batch_indices[idx]}: {text[:200]}",
                                    error_type=type(item_error).__name__,
                                    error_message=str(item_error),
                                    additional_info={
                                        "text_length": len(text),
                                        "batch_index": batch_start + idx,
                                    },
                                )
                            logger.error(
                                f"Failed to analyze text at index {batch_indices[idx]}: "
                                f"{item_error}"
                            )
                else:
                    raise RuntimeError(
                        f"Batch analysis failed: {batch_error}"
                    ) from batch_error

        # Map results back to original indices
        full_results = [None] * len(texts)
        for i, result in zip(valid_indices, all_results):
            full_results[i] = result

        success_count = sum(1 for r in full_results if r is not None)
        logger.info(
            f"Batch analysis complete: {success_count}/{len(texts)} successful, "
            f"{failures.count() if failures else 0} failed"
        )

        return full_results, failures

    except Exception as e:
        log_exception(logger, e, "Critical error during batch analysis")
        if skip_errors:
            logger.warning("Returning None for all results due to critical error")
            if failures:
                for i, text in enumerate(texts):
                    if text and text.strip():
                        failures.add_failure(
                            item=f"Index {i}: {text[:200]}",
                            error_type=type(e).__name__,
                            error_message=f"Critical batch error: {str(e)}",
                            additional_info={"text_length": len(text)},
                        )
            return [None] * len(texts), failures
        else:
            raise RuntimeError(f"Failed to analyze batch: {e}") from e


def analyze_with_metadata(
    text: str, metadata: Dict = None
) -> Dict[str, Union[str, float, Dict]]:
    """
    Analyze sentiment and attach metadata to the result.

    Useful for tracking source information (e.g., post ID, timestamp, author).

    Args:
        text: Raw text string to analyze
        metadata: Optional dictionary of metadata to attach to result

    Returns:
        Dictionary containing:
            - label: Sentiment label
            - score: Confidence score
            - text: Original text (truncated to 100 chars)
            - metadata: Any provided metadata
            - text_length: Character count of original text

    Example:
        >>> analyze_with_metadata(
        ...     "Market volatility increases",
        ...     metadata={'post_id': '123', 'author': 'user1'}
        ... )
        {
            'label': 'negative',
            'score': 0.85,
            'text': 'Market volatility increases',
            'text_length': 27,
            'metadata': {'post_id': '123', 'author': 'user1'}
        }
    """
    result = analyze_sentiment(text, return_all_scores=False)

    return {
        **result,
        "text": text[:100],  # Truncate for storage
        "text_length": len(text),
        "metadata": metadata or {},
    }


def get_sentiment_summary(results: List[Dict]) -> Dict[str, Union[int, float, Dict]]:
    """
    Generate summary statistics from a list of sentiment results.

    Args:
        results: List of sentiment result dictionaries from analyze_batch

    Returns:
        Dictionary containing:
            - total: Total number of results
            - counts: Count per label
            - percentages: Percentage per label
            - average_confidence: Average confidence score

    Example:
        >>> results = analyze_batch(["Good news", "Bad news", "Neutral news"])
        >>> get_sentiment_summary(results)
        {
            'total': 3,
            'counts': {'positive': 1, 'negative': 1, 'neutral': 1},
            'percentages': {'positive': 33.3, 'negative': 33.3, 'neutral': 33.3},
            'average_confidence': 0.87
        }
    """
    if not results:
        return {
            "total": 0,
            "counts": {"positive": 0, "negative": 0, "neutral": 0},
            "percentages": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
            "average_confidence": 0.0,
        }

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    total_score = 0.0

    for result in valid_results:
        label = result["label"]
        counts[label] += 1
        total_score += result["score"]

    total = len(valid_results)
    percentages = {
        label: round((count / total * 100), 2) for label, count in counts.items()
    }
    average_confidence = round(total_score / total, 4) if total > 0 else 0.0

    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "average_confidence": average_confidence,
    }
