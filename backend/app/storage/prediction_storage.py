"""
Sentiment Prediction Storage Module

Provides functionality to save and load sentiment predictions in structured formats
(CSV and JSON). Ensures data integrity with validation for all required fields.

Usage:
    from app.storage import save_prediction, load_predictions

    # Save single prediction
    save_prediction(
        text="Stock prices surged today",
        source="reddit",
        label="positive",
        confidence=0.95,
        output_file="predictions.csv"
    )

    # Load predictions
    predictions = load_predictions("predictions.csv")
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Required fields for prediction records
REQUIRED_FIELDS = ["text", "source", "timestamp", "label", "confidence"]


class PredictionRecord:
    """
    Data class for sentiment prediction records.

    Attributes:
        text: The input text that was analyzed
        source: Data source (e.g., 'reddit', 'twitter', 'news')
        timestamp: ISO format timestamp of when prediction was made
        label: Sentiment label ('positive', 'negative', or 'neutral')
        confidence: Confidence score (0-1)
    """

    def __init__(
        self,
        text: str,
        source: str,
        label: str,
        confidence: float,
        timestamp: Optional[str] = None,
    ):
        """
        Initialize a prediction record.

        Args:
            text: Input text
            source: Data source identifier
            label: Sentiment label
            confidence: Confidence score (0-1)
            timestamp: ISO timestamp (auto-generated if None)

        Raises:
            ValueError: If any required field is missing or invalid
        """
        # Validate inputs
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        if not source or not isinstance(source, str):
            raise ValueError("source must be a non-empty string")

        if label not in ["positive", "negative", "neutral"]:
            raise ValueError(
                f"label must be 'positive', 'negative', or 'neutral', got: {label}"
            )

        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got: {confidence}")

        self.text = text
        self.source = source
        self.label = label
        self.confidence = float(confidence)
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """
        Convert record to dictionary.

        Returns:
            Dictionary with all fields
        """
        return {
            "text": self.text,
            "source": self.source,
            "timestamp": self.timestamp,
            "label": self.label,
            "confidence": self.confidence,
        }

    def __repr__(self) -> str:
        return (
            f"PredictionRecord(text={self.text[:30]}..., source={self.source}, "
            f"label={self.label}, confidence={self.confidence:.3f})"
        )


def validate_prediction(prediction: Dict) -> bool:
    """
    Validate that a prediction dictionary has all required fields.

    Args:
        prediction: Dictionary to validate

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails with details about missing/invalid fields

    Example:
        >>> pred = {"text": "...", "source": "reddit", ...}
        >>> validate_prediction(pred)
        True
    """
    missing_fields = [field for field in REQUIRED_FIELDS if field not in prediction]

    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    # Validate field types and values
    if not prediction["text"] or not isinstance(prediction["text"], str):
        raise ValueError("text must be a non-empty string")

    if not prediction["source"] or not isinstance(prediction["source"], str):
        raise ValueError("source must be a non-empty string")

    if prediction["label"] not in ["positive", "negative", "neutral"]:
        raise ValueError(
            f"label must be 'positive', 'negative', or 'neutral', "
            f"got: {prediction['label']}"
        )

    try:
        confidence = float(prediction["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError(
            f"confidence must be between 0 and 1, got: {prediction['confidence']}"
        )

    # Validate timestamp format (ISO 8601)
    try:
        datetime.fromisoformat(prediction["timestamp"])
    except (ValueError, TypeError):
        raise ValueError(
            f"timestamp must be valid ISO format, got: {prediction['timestamp']}"
        )

    return True


def save_prediction(
    text: str,
    source: str,
    label: str,
    confidence: float,
    output_file: Union[str, Path],
    timestamp: Optional[str] = None,
    format: str = "csv",
) -> PredictionRecord:
    """
    Save a single sentiment prediction to file.

    Args:
        text: The analyzed text
        source: Data source (e.g., 'reddit', 'twitter', 'news')
        label: Sentiment label ('positive', 'negative', 'neutral')
        confidence: Confidence score (0-1)
        output_file: Path to output file
        timestamp: Optional ISO timestamp (auto-generated if None)
        format: Output format ('csv' or 'json')

    Returns:
        PredictionRecord that was saved

    Raises:
        ValueError: If validation fails
        IOError: If file write fails

    Example:
        >>> record = save_prediction(
        ...     text="Stock prices surged",
        ...     source="reddit",
        ...     label="positive",
        ...     confidence=0.95,
        ...     output_file="predictions.csv"
        ... )
    """
    # Create record (validates inputs)
    record = PredictionRecord(text, source, label, confidence, timestamp)

    # Convert to dict for saving
    prediction = record.to_dict()

    # Save using batch function (handles both formats)
    save_predictions_batch([prediction], output_file, format=format)

    logger.info(
        f"Saved prediction to {output_file}: {record.label} ({record.confidence:.3f})"
    )

    return record


def save_predictions_batch(
    predictions: List[Dict],
    output_file: Union[str, Path],
    format: str = "csv",
    append: bool = True,
) -> int:
    """
    Save multiple sentiment predictions to file.

    Args:
        predictions: List of prediction dictionaries with required fields
        output_file: Path to output file
        format: Output format ('csv' or 'json')
        append: If True, append to existing file. If False, overwrite.

    Returns:
        Number of predictions saved

    Raises:
        ValueError: If any prediction fails validation
        IOError: If file write fails

    Example:
        >>> predictions = [
        ...     {"text": "...", "source": "reddit", "label": "positive", ...},
        ...     {"text": "...", "source": "twitter", "label": "negative", ...}
        ... ]
        >>> count = save_predictions_batch(predictions, "predictions.csv")
    """
    if not predictions:
        logger.warning("No predictions to save")
        return 0

    # Validate all predictions first
    for i, pred in enumerate(predictions):
        try:
            validate_prediction(pred)
        except ValueError as e:
            raise ValueError(f"Prediction {i} validation failed: {e}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    format = format.lower()

    if format == "csv":
        _save_csv(predictions, output_path, append)
    elif format == "json":
        _save_json(predictions, output_path, append)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")

    logger.info(
        f"Saved {len(predictions)} predictions to {output_file} "
        f"(format={format}, append={append})"
    )

    return len(predictions)


def _save_csv(predictions: List[Dict], output_path: Path, append: bool) -> None:
    """Save predictions to CSV format."""
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if append and file_exists else "w"

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)

        # Write header only if new file or overwrite mode
        if mode == "w":
            writer.writeheader()

        writer.writerows(predictions)


def _save_json(predictions: List[Dict], output_path: Path, append: bool) -> None:
    """Save predictions to JSON format."""
    if append and output_path.exists():
        # Load existing data
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = []

        # Append new predictions
        all_predictions = existing_data + predictions
    else:
        all_predictions = predictions

    # Save all data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)


def load_predictions(
    input_file: Union[str, Path],
    format: Optional[str] = None,
    validate: bool = True,
) -> List[Dict]:
    """
    Load sentiment predictions from file.

    Args:
        input_file: Path to input file
        format: File format ('csv' or 'json'). Auto-detected from extension if None.
        validate: If True, validate all loaded predictions

    Returns:
        List of prediction dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails
        IOError: If file read fails

    Example:
        >>> predictions = load_predictions("predictions.csv")
        >>> print(f"Loaded {len(predictions)} predictions")
        >>> print(predictions[0]['label'])  # 'positive'
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Auto-detect format from extension if not specified
    if format is None:
        format = input_path.suffix.lstrip(".").lower()
        if format not in ["csv", "json"]:
            raise ValueError(
                f"Cannot auto-detect format from extension: {input_path.suffix}. "
                "Specify format parameter."
            )

    format = format.lower()

    if format == "csv":
        predictions = _load_csv(input_path)
    elif format == "json":
        predictions = _load_json(input_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")

    # Validate all predictions if requested
    if validate:
        for i, pred in enumerate(predictions):
            try:
                validate_prediction(pred)
            except ValueError as e:
                logger.warning(f"Prediction {i} failed validation: {e}")
                raise

    logger.info(f"Loaded {len(predictions)} predictions from {input_file}")

    return predictions


def _load_csv(input_path: Path) -> List[Dict]:
    """Load predictions from CSV format."""
    predictions = []

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Convert confidence to float
            row["confidence"] = float(row["confidence"])
            predictions.append(row)

    return predictions


def _load_json(input_path: Path) -> List[Dict]:
    """Load predictions from JSON format."""
    with open(input_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if not isinstance(predictions, list):
        raise ValueError("JSON file must contain a list of predictions")

    return predictions


def get_storage_stats(input_file: Union[str, Path]) -> Dict:
    """
    Get statistics about stored predictions.

    Args:
        input_file: Path to predictions file

    Returns:
        Dictionary with statistics:
            - total: Total number of predictions
            - by_label: Count by sentiment label
            - by_source: Count by data source
            - avg_confidence: Average confidence score
            - date_range: First and last timestamp

    Example:
        >>> stats = get_storage_stats("predictions.csv")
        >>> print(f"Total: {stats['total']}")
        >>> print(f"Positive: {stats['by_label']['positive']}")
    """
    predictions = load_predictions(input_file, validate=False)

    if not predictions:
        return {
            "total": 0,
            "by_label": {},
            "by_source": {},
            "avg_confidence": 0.0,
            "date_range": None,
        }

    # Count by label
    by_label = {}
    for pred in predictions:
        label = pred["label"]
        by_label[label] = by_label.get(label, 0) + 1

    # Count by source
    by_source = {}
    for pred in predictions:
        source = pred["source"]
        by_source[source] = by_source.get(source, 0) + 1

    # Calculate average confidence
    total_confidence = sum(float(pred["confidence"]) for pred in predictions)
    avg_confidence = total_confidence / len(predictions)

    # Get date range
    timestamps = [pred["timestamp"] for pred in predictions]
    timestamps.sort()
    date_range = {"first": timestamps[0], "last": timestamps[-1]}

    return {
        "total": len(predictions),
        "by_label": by_label,
        "by_source": by_source,
        "avg_confidence": avg_confidence,
        "date_range": date_range,
    }
