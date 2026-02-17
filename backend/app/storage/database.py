"""
SQLite database module using SQLAlchemy.

Provides the database engine, session factory, ORM model, and migration helpers
for unified sentiment record storage.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    func,
    inspect,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .record_ids import make_record_id

logger = logging.getLogger(__name__)

Base = declarative_base()


class SentimentRecordRow(Base):
    """SQLAlchemy model for unified sentiment_records table."""

    __tablename__ = "sentiment_records"

    id = Column(String, primary_key=True)
    record_type = Column(String(20), nullable=False, index=True)  # document | stock
    document_id = Column(String, nullable=True, index=True)
    text = Column(Text, default="")
    ticker = Column(String(10), nullable=True, index=True)
    mentioned_as = Column(String(100), default="")
    sentiment_label = Column(String(10), nullable=False, index=True)
    sentiment_score = Column(Float, nullable=False)
    context = Column(Text, default="")
    source = Column(String(20), default="")
    source_id = Column(String(100), default="")
    position_start = Column(Integer, nullable=True)
    position_end = Column(Integer, nullable=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    sentiment_mode = Column(String(20), default="keyword")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_sentiment_records_record_type_timestamp", "record_type", "timestamp"),
        Index("ix_sentiment_records_ticker_timestamp", "ticker", "timestamp"),
        Index("ix_sentiment_records_ticker_label", "ticker", "sentiment_label"),
    )

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "record_type": self.record_type,
            "document_id": self.document_id,
            "text": self.text,
            "ticker": self.ticker,
            "mentioned_as": self.mentioned_as,
            "sentiment_label": self.sentiment_label,
            "sentiment_score": self.sentiment_score,
            "context": self.context,
            "source": self.source,
            "source_id": self.source_id,
            "position_start": self.position_start,
            "position_end": self.position_end,
            "timestamp": self.timestamp.isoformat() + "Z" if self.timestamp else None,
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


def _normalize_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _coerce_position(position: Optional[object]) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(position, dict):
        start = position.get("start")
        end = position.get("end")
        return (
            int(start) if isinstance(start, int) else None,
            int(end) if isinstance(end, int) else None,
        )
    if isinstance(position, int):
        return (position, None)
    return (None, None)


def _bulk_insert_records(session: Session, records: List[Dict]) -> int:
    if not records:
        return 0

    # SQLite default variable limit is 999. Keep batches under that.
    columns = len(records[0])
    max_vars = 900
    batch_size = max(1, max_vars // max(columns, 1))

    total_inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        stmt = sqlite_insert(SentimentRecordRow).values(batch)
        stmt = stmt.prefix_with("OR IGNORE")
        result = session.execute(stmt)
        total_inserted += int(result.rowcount or 0)

    session.commit()
    return total_inserted


def _migrate_from_legacy_table(session: Session) -> int:
    engine = session.get_bind()
    inspector = inspect(engine)
    if "stock_sentiments" not in inspector.get_table_names():
        return 0

    logger.info("Migrating legacy table stock_sentiments into sentiment_records...")
    rows = (
        session.execute(
            sql_text(
                "SELECT id, ticker, mentioned_as, sentiment_label, sentiment_score, "
                "context, source, source_id, full_text, position, timestamp, sentiment_mode "
                "FROM stock_sentiments"
            )
        )
        .mappings()
        .all()
    )

    records: List[Dict] = []
    for row in rows:
        ts = _normalize_timestamp(row.get("timestamp")) or datetime.utcnow()
        pos_start, pos_end = _coerce_position(row.get("position"))
        text_val = row.get("full_text") or row.get("context") or ""
        records.append(
            {
                "id": row.get("id")
                or make_record_id(
                    "stock",
                    row.get("ticker", ""),
                    row.get("timestamp", ""),
                    row.get("context", ""),
                ),
                "record_type": "stock",
                "document_id": None,
                "text": text_val,
                "ticker": row.get("ticker"),
                "mentioned_as": row.get("mentioned_as") or "",
                "sentiment_label": row.get("sentiment_label") or "neutral",
                "sentiment_score": float(row.get("sentiment_score") or 0.5),
                "context": row.get("context") or "",
                "source": row.get("source") or "",
                "source_id": row.get("source_id") or "",
                "position_start": pos_start,
                "position_end": pos_end,
                "timestamp": ts,
                "sentiment_mode": row.get("sentiment_mode") or "keyword",
            }
        )

    inserted = _bulk_insert_records(session, records)
    if inserted:
        logger.info(f"  Migrated {inserted} records from legacy table")
    return inserted


def _migrate_from_stock_json(session: Session, json_path: Path) -> int:
    if not json_path.exists():
        return 0

    logger.info(f"Migrating stock_sentiments.json from {json_path}...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read {json_path}: {e}")
        return 0

    sentiments = data.get("sentiments", []) if isinstance(data, dict) else []
    if not sentiments:
        return 0

    records: List[Dict] = []
    for record in sentiments:
        ts = _normalize_timestamp(record.get("timestamp")) or datetime.utcnow()
        pos_start, pos_end = _coerce_position(record.get("position"))
        text_val = record.get("full_text") or record.get("context") or ""
        records.append(
            {
                "id": record.get("id")
                or make_record_id(
                    "stock",
                    record.get("ticker", ""),
                    record.get("timestamp", ""),
                    record.get("context", ""),
                ),
                "record_type": "stock",
                "document_id": None,
                "text": text_val,
                "ticker": record.get("ticker"),
                "mentioned_as": record.get("mentioned_as", ""),
                "sentiment_label": record.get("sentiment_label", "neutral"),
                "sentiment_score": float(record.get("sentiment_score", 0.5)),
                "context": record.get("context", ""),
                "source": record.get("source", "") or "",
                "source_id": record.get("source_id", "") or "",
                "position_start": pos_start,
                "position_end": pos_end,
                "timestamp": ts,
                "sentiment_mode": record.get("sentiment_mode", "keyword"),
            }
        )

    inserted = _bulk_insert_records(session, records)
    if inserted:
        logger.info(f"  Migrated {inserted} records from JSON")
    return inserted


def _migrate_predictions_files(session: Session, predictions_dir: Path) -> int:
    if not predictions_dir.exists():
        return 0

    inserted_total = 0

    # CSV predictions
    for csv_path in predictions_dir.glob("*.csv"):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = []
                for row in reader:
                    ts = _normalize_timestamp(row.get("timestamp")) or datetime.utcnow()
                    label = row.get("label") or "neutral"
                    confidence = row.get("confidence")
                    try:
                        confidence_val = (
                            float(confidence) if confidence is not None else 0.5
                        )
                    except (ValueError, TypeError):
                        confidence_val = 0.5
                    text_val = row.get("text") or ""
                    source_val = row.get("source") or ""
                    record_id = make_record_id(
                        "doc", source_val, row.get("timestamp", ""), text_val[:120]
                    )
                    records.append(
                        {
                            "id": record_id,
                            "record_type": "document",
                            "document_id": record_id,
                            "text": text_val,
                            "ticker": None,
                            "mentioned_as": "",
                            "sentiment_label": label,
                            "sentiment_score": confidence_val,
                            "context": "",
                            "source": source_val,
                            "source_id": "",
                            "position_start": None,
                            "position_end": None,
                            "timestamp": ts,
                            "sentiment_mode": "finbert",
                        }
                    )
                inserted_total += _bulk_insert_records(session, records)
        except OSError as e:
            logger.warning(f"Failed to read {csv_path}: {e}")

    # JSON predictions
    for json_path in predictions_dir.glob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read {json_path}: {e}")
            continue

        if not isinstance(data, list):
            continue

        records = []
        for item in data:
            ts = _normalize_timestamp(item.get("timestamp")) or datetime.utcnow()
            label = item.get("label") or "neutral"
            confidence = item.get("confidence")
            try:
                confidence_val = float(confidence) if confidence is not None else 0.5
            except (ValueError, TypeError):
                confidence_val = 0.5
            text_val = item.get("text") or ""
            source_val = item.get("source") or ""
            record_id = make_record_id(
                "doc", source_val, item.get("timestamp", ""), text_val[:120]
            )
            records.append(
                {
                    "id": record_id,
                    "record_type": "document",
                    "document_id": record_id,
                    "text": text_val,
                    "ticker": None,
                    "mentioned_as": "",
                    "sentiment_label": label,
                    "sentiment_score": confidence_val,
                    "context": "",
                    "source": source_val,
                    "source_id": "",
                    "position_start": None,
                    "position_end": None,
                    "timestamp": ts,
                    "sentiment_mode": "finbert",
                }
            )
        inserted_total += _bulk_insert_records(session, records)

    if inserted_total:
        logger.info(
            f"  Migrated {inserted_total} document records from predictions files"
        )
    return inserted_total


def migrate_legacy_data(db_path: Optional[Path] = None) -> int:
    """
    Migrate legacy JSON/CSV data into the unified sentiment_records table.

    Sources:
    - Legacy SQLite table: stock_sentiments
    - data/stock_sentiments/stock_sentiments.json
    - data/predictions/*.csv and *.json

    Returns:
        Number of records inserted.
    """
    engine = get_engine(db_path)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        existing_count = session.query(func.count(SentimentRecordRow.id)).scalar() or 0
        if existing_count > 0:
            logger.info(
                f"sentiment_records already has {existing_count} records. Skipping migration."
            )
            return int(existing_count)

        total_inserted = 0
        total_inserted += _migrate_from_legacy_table(session)

        backend_dir = Path(__file__).parent.parent.parent
        data_dir = backend_dir.parent / "data"
        stock_json = data_dir / "stock_sentiments" / "stock_sentiments.json"
        total_inserted += _migrate_from_stock_json(session, stock_json)

        predictions_dir = data_dir / "predictions"
        total_inserted += _migrate_predictions_files(session, predictions_dir)

        logger.info(f"Migration complete: {total_inserted} records inserted.")
        return total_inserted

    except Exception as e:
        session.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        session.close()
