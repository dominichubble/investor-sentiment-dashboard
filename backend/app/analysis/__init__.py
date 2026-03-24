"""
Analysis module for sentiment-price correlation.

Provides tools to fetch stock prices, calculate correlations between
sentiment scores and price movements, and support lag analysis.
"""

from .correlation import CorrelationAnalyzer
from .price_service import PriceService

__all__ = ["PriceService", "CorrelationAnalyzer"]
