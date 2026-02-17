"""
Pydantic validation schemas for all data records.

Provides strict validation at ingestion and storage boundaries to prevent
malformed records from propagating through the pipeline.
"""

from .news import NewsArticleRecord, NewsRawRecord
from .reddit import RedditPostRecord, RedditRawRecord
from .sentiment import StockSentimentRecord, SentimentMetadata

__all__ = [
    "RedditRawRecord",
    "RedditPostRecord",
    "NewsRawRecord",
    "NewsArticleRecord",
    "StockSentimentRecord",
    "SentimentMetadata",
]
