"""
PostgreSQL (Neon) database module using SQLAlchemy.

Requires DATABASE_URL (connection string). Tables are created on first engine use.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class SentimentRecordRow(Base):
    """SQLAlchemy model for unified sentiment_records table."""

    __tablename__ = "sentiment_records"

    id = Column(String, primary_key=True)
    text = Column(Text, default="")
    ticker = Column(String(10), nullable=True, index=True)
    mentioned_as = Column(String(100), default="")
    sentiment_label = Column(String(10), nullable=False, index=True)
    sentiment_score = Column(Float, nullable=False)
    score_positive = Column(Float, nullable=True)
    score_negative = Column(Float, nullable=True)
    score_neutral = Column(Float, nullable=True)
    sentiment_uncertainty = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)
    aspects_json = Column(Text, nullable=True)
    source = Column(String(20), default="")
    data_source = Column(String(20), nullable=True, index=True)
    source_id = Column(String(100), default="")
    source_meta_json = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=False, index=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_sentiment_records_ticker_published", "ticker", "published_at"),
        Index("ix_sentiment_records_ticker_label", "ticker", "sentiment_label"),
        Index("ix_sentiment_records_data_source_published", "data_source", "published_at"),
    )

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "record_type": "stock" if self.ticker else "document",
            "text": self.text,
            "ticker": self.ticker,
            "mentioned_as": self.mentioned_as,
            "sentiment_label": self.sentiment_label,
            "sentiment_score": self.sentiment_score,
            "score_positive": self.score_positive,
            "score_negative": self.score_negative,
            "score_neutral": self.score_neutral,
            "sentiment_uncertainty": self.sentiment_uncertainty,
            "rationale": self.rationale,
            "aspects_json": self.aspects_json,
            "source": self.source,
            "data_source": self.data_source,
            "source_id": self.source_id,
            "source_meta_json": self.source_meta_json,
            "published_at": self.published_at.isoformat() + "Z" if self.published_at else None,
        }


class SentimentNarrativeCacheRow(Base):
    """Cached LLM narrative summaries for ticker sentiment (Groq / OpenAI-compatible)."""

    __tablename__ = "sentiment_narrative_cache"

    id = Column(String(36), primary_key=True)
    ticker = Column(String(12), nullable=False, index=True)
    period_key = Column(String(80), nullable=False)
    data_signature = Column(String(64), nullable=False)
    narrative_text = Column(Text, nullable=False)
    model_id = Column(String(80), nullable=False)
    record_count = Column(Integer, nullable=False)
    window_start = Column(DateTime, nullable=True)
    window_end = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "period_key",
            "data_signature",
            name="uq_sentiment_narrative_cache_key",
        ),
    )


_engine = None
_SessionLocal = None
_dotenv_attempted = False


def _load_dotenv_if_needed() -> None:
    """Load repo-root .env when DATABASE_URL is missing (uvicorn workers, IDE runs)."""
    global _dotenv_attempted
    if os.environ.get("DATABASE_URL"):
        return
    if _dotenv_attempted:
        return
    _dotenv_attempted = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = here / ".env"
        if candidate.is_file():
            load_dotenv(candidate)
            logger.debug("Loaded environment from %s", candidate)
            return
        if here.parent == here:
            break
        here = here.parent


def get_engine():
    """Get or create the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        _load_dotenv_if_needed()
        url = os.environ.get("DATABASE_URL", "").strip()
        if not url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Set it to your Neon PostgreSQL connection string."
            )
        _engine = create_engine(url, pool_pre_ping=True, echo=False)
        Base.metadata.create_all(_engine)
        logger.info("PostgreSQL database initialized via %s", url.split("@")[-1])
    return _engine


def get_session() -> Session:
    """Get a new database session."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()
