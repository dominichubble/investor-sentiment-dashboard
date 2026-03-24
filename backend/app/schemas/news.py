"""Pydantic schemas for news article data records."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class NewsRawRecord(BaseModel):
    """Schema for raw news article data as received from the news API."""

    title: str = Field(default="", description="Article title")
    description: Optional[str] = Field(default=None, description="Article description")
    content: Optional[str] = Field(default=None, description="Article content")
    clean_title: Optional[str] = Field(default=None, description="Cleaned title")
    clean_description: Optional[str] = Field(
        default=None, description="Cleaned description"
    )
    clean_content: Optional[str] = Field(default=None, description="Cleaned content")
    published_at: Optional[str] = Field(
        default=None, description="Publication timestamp"
    )
    source_id: Optional[str] = Field(default=None, description="Source identifier")
    source_name: Optional[str] = Field(default=None, description="Source name")
    url: Optional[str] = Field(default=None, description="Article URL")
    author: Optional[str] = Field(default=None, description="Article author")


class NewsArticleRecord(BaseModel):
    """Schema for a processed news record ready for sentiment analysis."""

    text: str = Field(
        ..., min_length=15, max_length=2000, description="Cleaned text for analysis"
    )
    source: str = Field(default="news", description="Data source identifier")
    timestamp: str = Field(..., description="ISO format timestamp")
    source_id: str = Field(default="", description="Original source ID")
    source_name: str = Field(default="", description="News source name")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid ISO timestamp: {v}")
        return v
