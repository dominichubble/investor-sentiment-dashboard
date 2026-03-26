"""Storage module for sentiment predictions."""

from .prediction_storage import (
    PredictionRecord,
    get_storage_stats,
    load_predictions,
    save_prediction,
    save_predictions_batch,
    validate_prediction,
)
from .sqlite_storage import SentimentStorage as StockSentimentStorage

__all__ = [
    "PredictionRecord",
    "get_storage_stats",
    "load_predictions",
    "save_prediction",
    "save_predictions_batch",
    "validate_prediction",
    "StockSentimentStorage",
]
