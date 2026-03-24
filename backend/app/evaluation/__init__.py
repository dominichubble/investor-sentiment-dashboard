"""
Evaluation module for sentiment analysis model benchmarking.

Provides tools to evaluate keyword-based and FinBERT sentiment analysis
against a labeled dataset with accuracy, precision, recall, and F1 metrics.
"""

from .benchmark import run_benchmark
from .metrics import evaluate_predictions

__all__ = ["run_benchmark", "evaluate_predictions"]
