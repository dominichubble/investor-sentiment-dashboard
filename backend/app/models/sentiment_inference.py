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
from typing import Dict, List, Union

from .finbert_model import get_model

logger = logging.getLogger(__name__)


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
        model = get_model()
        result = model.predict(text, return_all_scores=return_all_scores)
        logger.debug(f"Analyzed text: '{text[:50]}...' -> {result['label']}")
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise RuntimeError(f"Failed to analyze sentiment: {e}")


def analyze_batch(
    texts: List[str],
    batch_size: int = 32,
    return_all_scores: bool = False,
    skip_errors: bool = False,
) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
    """
    Analyze sentiment for multiple text inputs efficiently.

    Args:
        texts: List of text strings to analyze
        batch_size: Number of texts to process per batch (default: 32)
        return_all_scores: If True, include confidence scores for all labels
        skip_errors: If True, return None for failed analyses instead of raising

    Returns:
        List of dictionaries, one per input text. Each dict contains:
            - label: Sentiment label ('positive', 'negative', or 'neutral')
            - score: Confidence score for the predicted label (0-1)
            - scores: (optional) All label scores if return_all_scores=True
        
        If skip_errors=True, failed analyses will be None in the list.

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

    # Filter out empty texts
    valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    valid_texts = [texts[i] for i in valid_indices]

    if not valid_texts:
        raise ValueError("No valid (non-empty) texts provided")

    try:
        model = get_model()
        results = model.predict_batch(
            valid_texts, batch_size=batch_size, return_all_scores=return_all_scores
        )

        # Map results back to original indices
        full_results = [None] * len(texts)
        for i, result in zip(valid_indices, results):
            full_results[i] = result

        logger.info(f"Batch analysis complete: {len(valid_texts)}/{len(texts)} texts")
        return full_results

    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {e}")
        if skip_errors:
            logger.warning("Returning None for all results due to error")
            return [None] * len(texts)
        else:
            raise RuntimeError(f"Failed to analyze batch: {e}")


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
