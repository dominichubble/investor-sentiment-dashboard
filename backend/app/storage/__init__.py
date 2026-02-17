"""Storage module for sentiment predictions."""

import os

from .prediction_storage import (
    PredictionRecord,
    get_storage_stats,
    load_predictions,
    save_prediction,
    save_predictions_batch,
    validate_prediction,
)
from .stock_sentiment_storage import StockSentimentStorage as _JSONStorage

# Use SQLite storage by default; set STORAGE_BACKEND=json to use JSON
_USE_SQLITE = os.environ.get("STORAGE_BACKEND", "sqlite").lower() != "json"

if _USE_SQLITE:
    try:
        from .sqlite_storage import SQLiteStockSentimentStorage as StockSentimentStorage
    except ImportError:
        # Fallback to JSON if SQLAlchemy is not installed
        StockSentimentStorage = _JSONStorage
else:
    StockSentimentStorage = _JSONStorage

__all__ = [
    "PredictionRecord",
    "get_storage_stats",
    "load_predictions",
    "save_prediction",
    "save_predictions_batch",
    "validate_prediction",
    "StockSentimentStorage",
]
