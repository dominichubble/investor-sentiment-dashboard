"""
SQLite database module using SQLAlchemy.

Provides the database engine, session factory, and ORM models
for stock sentiment data storage.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class StockSentimentRow(Base):
    """SQLAlchemy model for stock_sentiments table."""

    __tablename__ = "stock_sentiments"

    id = Column(String, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    mentioned_as = Column(String(100), default="")
    sentiment_label = Column(String(10), nullable=False, index=True)
    sentiment_score = Column(Float, nullable=False)
    context = Column(Text, default="")
    source = Column(String(20), default="")
    source_id = Column(String(100), default="")
    full_text = Column(Text, nullable=True)
    position = Column(Integer, nullable=True)
    timestamp = Column(String(50), nullable=False, index=True)
    sentiment_mode = Column(String(20), default="keyword")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ticker_timestamp", "ticker", "timestamp"),
        Index("ix_ticker_label", "ticker", "sentiment_label"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "ticker": self.ticker,
            "mentioned_as": self.mentioned_as,
            "sentiment_label": self.sentiment_label,
            "sentiment_score": self.sentiment_score,
            "context": self.context,
            "source": self.source,
            "source_id": self.source_id,
            "full_text": self.full_text,
            "position": self.position,
            "timestamp": self.timestamp,
            "sentiment_mode": self.sentiment_mode,
        }


_engine = None
_SessionLocal = None


def get_db_path() -> Path:
    """Get the default database path."""
    backend_dir = Path(__file__).parent.parent.parent
    db_dir = backend_dir.parent / "data" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sentiments.db"


def get_engine(db_path: Optional[Path] = None):
    """Get or create the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        if db_path is None:
            db_path = get_db_path()
        db_url = f"sqlite:///{db_path}"
        _engine = create_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(_engine)
        logger.info(f"SQLite database initialized at {db_path}")
    return _engine


def get_session() -> Session:
    """Get a new database session."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()


def migrate_from_json(json_path: Path, db_path: Optional[Path] = None) -> int:
    """
    Migrate stock_sentiments.json data into the SQLite database.

    Args:
        json_path: Path to the stock_sentiments.json file.
        db_path: Optional database path.

    Returns:
        Number of records migrated.
    """
    import json

    if not json_path.exists():
        logger.warning(f"JSON file not found: {json_path}")
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentiments = data.get("sentiments", [])
    if not sentiments:
        logger.info("No sentiments to migrate")
        return 0

    engine = get_engine(db_path)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        existing_count = session.query(StockSentimentRow).count()
        if existing_count > 0:
            logger.info(f"Database already has {existing_count} records. Skipping migration.")
            return existing_count

        batch_size = 1000
        count = 0
        skipped = 0
        seen_ids = set()

        for i in range(0, len(sentiments), batch_size):
            batch = sentiments[i: i + batch_size]
            rows = []
            for record in batch:
                record_id = record.get("id", "")
                if record_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(record_id)

                rows.append(StockSentimentRow(
                    id=record_id,
                    ticker=record.get("ticker", ""),
                    mentioned_as=record.get("mentioned_as", ""),
                    sentiment_label=record.get("sentiment_label", "neutral"),
                    sentiment_score=record.get("sentiment_score", 0.5),
                    context=record.get("context", ""),
                    source=record.get("source", ""),
                    source_id=record.get("source_id", ""),
                    full_text=record.get("full_text"),
                    position=record.get("position"),
                    timestamp=record.get("timestamp", ""),
                    sentiment_mode=record.get("sentiment_mode", "keyword"),
                ))
            if rows:
                session.bulk_save_objects(rows)
                session.commit()
            count += len(rows)
            logger.info(f"  Migrated {count}/{len(sentiments)} records...")

        if skipped:
            logger.info(f"  Skipped {skipped} duplicate records during migration.")
        logger.info(f"Migration complete: {count} records inserted.")
        return count

    except Exception as e:
        session.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        session.close()
