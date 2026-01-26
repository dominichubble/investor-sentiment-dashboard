"""
Stock sentiment analysis module.

Pairs stock entities with sentiment analysis.
"""

from .stock_sentiment import StockSentimentAnalyzer, analyze_stock_sentiment

__all__ = [
    "StockSentimentAnalyzer",
    "analyze_stock_sentiment",
]
