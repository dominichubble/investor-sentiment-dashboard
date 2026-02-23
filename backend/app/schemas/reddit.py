"""Pydantic schemas for Reddit data records."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RedditRawRecord(BaseModel):
    """Schema for raw Reddit post data as received from the Reddit API."""

    id: str = Field(..., min_length=1, description="Reddit post ID")
    title: str = Field(default="", description="Post title")
    selftext: Optional[str] = Field(default=None, description="Post body text")
    subreddit: str = Field(default="", description="Subreddit name")
    created_utc: Optional[float] = Field(default=None, description="Unix timestamp")
    score: Optional[int] = Field(default=None, description="Post score/upvotes")
    num_comments: Optional[int] = Field(default=None, description="Comment count")
    author: Optional[str] = Field(default=None, description="Author username")
    url: Optional[str] = Field(default=None, description="Post URL")

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Title must not be empty")
        return v.strip()

    @field_validator("subreddit")
    @classmethod
    def clean_subreddit(cls, v: str) -> str:
        return v.strip().lower() if v else ""


class RedditPostRecord(BaseModel):
    """Schema for a processed Reddit record ready for sentiment analysis."""

    text: str = Field(
        ..., min_length=15, max_length=2000, description="Cleaned text for analysis"
    )
    source: str = Field(default="reddit", description="Data source identifier")
    timestamp: str = Field(..., description="ISO format timestamp")
    source_id: str = Field(default="", description="Original Reddit post ID")
    subreddit: str = Field(default="", description="Subreddit name")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid ISO timestamp: {v}")
        return v
