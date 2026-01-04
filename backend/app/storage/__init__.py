"""Storage module for sentiment predictions."""

from .prediction_storage import (
    PredictionRecord,
    get_storage_stats,
    load_predictions,
    save_prediction,
    save_predictions_batch,
    validate_prediction,
)

__all__ = [
    "PredictionRecord",
    "save_prediction",
    "save_predictions_batch",
    "load_predictions",
    "validate_prediction",
    "get_storage_stats",
]
