"""Pydantic schemas for stock sentiment records."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class StockSentimentRecord(BaseModel):
    """Schema for a single stock-sentiment record in the output dataset."""

    id: str = Field(..., min_length=1, description="Unique record identifier")
    ticker: str = Field(
        ..., min_length=1, max_length=10, description="Stock ticker symbol"
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
    full_text: Optional[str] = Field(default=None, description="Full original text")
    position: Optional[int] = Field(
        default=None, description="Position of mention in text"
    )
    timestamp: str = Field(..., description="ISO format timestamp")
    sentiment_mode: str = Field(
        default="keyword", description="Sentiment method used (keyword or finbert)"
    )

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid ISO timestamp: {v}")
        return v


class SentimentMetadata(BaseModel):
    """Schema for the metadata section of the stock_sentiments.json output."""

    total_sentiments: int = Field(..., ge=0)
    unique_tickers: int = Field(..., ge=0)
    last_updated: str = Field(..., description="ISO timestamp of last processing run")
    sentiment_mode: str = Field(
        default="keyword", description="keyword or finbert"
    )
    processing_stats: Optional[dict] = Field(
        default=None, description="Processing statistics"
    )
