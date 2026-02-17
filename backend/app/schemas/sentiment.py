"""Pydantic schemas for unified sentiment records."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SentimentRecord(BaseModel):
    """Schema for a unified sentiment record (document or stock)."""

    id: str = Field(..., min_length=1, description="Unique record identifier")
    record_type: Literal["document", "stock"] = Field(
        ..., description="Record type (document or stock mention)"
    )
    document_id: Optional[str] = Field(
        default=None, description="Document group ID for stock mentions"
    )
    text: str = Field(default="", description="Full original text")
    ticker: Optional[str] = Field(
        default=None, min_length=1, max_length=10, description="Stock ticker symbol"
    )
    mentioned_as: str = Field(
        default="", description="How the stock was mentioned (e.g. $AAPL, Apple)"
    )
    sentiment_label: Literal["positive", "negative", "neutral"] = Field(
        ..., description="Sentiment classification"
    )
    sentiment_score: float = Field(
        ..., ge=0.0, le=1.0, description="Sentiment confidence score"
    )
    context: str = Field(
        default="", max_length=500, description="Text snippet for context"
    )
    source: str = Field(
        default="", description="Data source (reddit, news, prediction)"
    )
    source_id: str = Field(default="", description="Original record ID from source")
    position_start: Optional[int] = Field(
        default=None, description="Start position of mention in text"
    )
    position_end: Optional[int] = Field(
        default=None, description="End position of mention in text"
    )
    timestamp: str = Field(..., description="ISO format timestamp")
    sentiment_mode: str = Field(
        default="keyword", description="Sentiment method used (keyword or finbert)"
    )

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return v.upper().strip()

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid ISO timestamp: {v}")
        return v


# Backwards compatibility alias
class StockSentimentRecord(SentimentRecord):
    """Legacy alias for stock sentiment records."""

    pass


class SentimentMetadata(BaseModel):
    """Schema for sentiment processing metadata (legacy JSON export support)."""

    total_sentiments: int = Field(..., ge=0)
    unique_tickers: int = Field(..., ge=0)
    last_updated: str = Field(..., description="ISO timestamp of last processing run")
    sentiment_mode: str = Field(default="keyword", description="keyword or finbert")
    processing_stats: Optional[dict] = Field(
        default=None, description="Processing statistics"
    )
