"""
Models package for sentiment analysis.

This package contains model wrappers and utilities for financial sentiment analysis.
"""

from app.models.finbert_model import FinBERTModel, get_model

__all__ = ["FinBERTModel", "get_model"]
