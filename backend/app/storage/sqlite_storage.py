"""
SQLite-backed unified sentiment storage.

Stores both document-level predictions and per-stock mentions in a single table.
Stock rows are identified by having a non-null ticker column.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .database import SentimentRecordRow, get_engine, get_session
from .record_ids import make_record_id

logger = logging.getLogger(__name__)


class SQLiteSentimentStorage:
    """SQLite-backed storage for unified sentiment records."""

    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            backend_dir = Path(__file__).parent.parent.parent
            storage_dir = backend_dir.parent / "data"

        self.storage_dir = Path(storage_dir)
        get_engine()

    def load(self) -> None:
        """No-op kept for backward compatibility."""

    # ---- Save helpers ----

    @staticmethod
    def _normalize_timestamp(ts: Optional[str]) -> datetime:
        if ts:
            try:
                parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                return parsed
            except (ValueError, TypeError):
                pass
        return datetime.utcnow()

    def _insert_records(self, records: List[Dict]) -> int:
        if not records:
            return 0
        session = get_session()
        try:
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
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save records: {e}")
            raise
        finally:
            session.close()

    def save_records_batch(self, records: List[Dict]) -> int:
        """Save multiple sentiment records."""
        normalized: List[Dict] = []
        for record in records:
            ts_val = record.get("timestamp")
            if isinstance(ts_val, datetime):
                ts = ts_val
            else:
                ts = self._normalize_timestamp(ts_val)

            normalized.append(
                {
                    "id": record["id"],
                    "text": record.get("text", "") or "",
                    "ticker": record.get("ticker"),
                    "mentioned_as": record.get("mentioned_as", "") or "",
                    "sentiment_label": record.get("sentiment_label", "neutral"),
                    "sentiment_score": float(record.get("sentiment_score", 0.5)),
                    "score_positive": record.get("score_positive"),
                    "score_negative": record.get("score_negative"),
                    "score_neutral": record.get("score_neutral"),
                    "sentiment_uncertainty": record.get("sentiment_uncertainty"),
                    "rationale": record.get("rationale"),
                    "aspects_json": record.get("aspects_json"),
                    "source": record.get("source", "") or "",
                    "data_source": record.get("data_source"),
                    "source_id": record.get("source_id", "") or "",
                    "timestamp": ts,
                }
            )

        return self._insert_records(normalized)

    def save_document_sentiment(self, record: Dict) -> str:
        """Save a single document-level sentiment record."""
        self.save_records_batch([record])
        return record["id"]

    def save_stock_sentiment(self, record: Dict) -> str:
        """Save a single stock mention sentiment record."""
        self.save_records_batch([record])
        return record["id"]

    def save_analysis_result(
        self, result: Dict, source: Optional[str] = None
    ) -> List[str]:
        """Save a StockSentimentAnalyzer result (one document + stock mentions)."""
        text = result.get("text", "")
        overall = result.get("overall_sentiment", {}) or {}
        timestamp = datetime.utcnow().isoformat() + "Z"

        source_id = result.get("source_id", "")
        doc_id = result.get("document_id") or make_record_id(
            "doc", source or "", source_id, timestamp, text[:200]
        )

        document_record = {
            "id": doc_id,
            "text": text,
            "ticker": None,
            "sentiment_label": overall.get("label", "neutral"),
            "sentiment_score": overall.get("score", 0.5),
            "source": source or "",
            "data_source": result.get("data_source"),
            "source_id": source_id,
            "timestamp": timestamp,
        }

        stock_records = []
        for stock in result.get("stocks", []):
            stock_records.append(
                {
                    "id": make_record_id(
                        "stock",
                        source or "",
                        source_id,
                        stock.get("ticker", ""),
                        stock.get("mentioned_as", ""),
                    ),
                    "text": text,
                    "ticker": stock.get("ticker"),
                    "mentioned_as": stock.get("mentioned_as", ""),
                    "sentiment_label": stock.get("sentiment", {}).get(
                        "label", "neutral"
                    ),
                    "sentiment_score": stock.get("sentiment", {}).get("score", 0.5),
                    "source": source or "",
                    "data_source": result.get("data_source"),
                    "source_id": source_id,
                    "timestamp": timestamp,
                }
            )

        self.save_records_batch([document_record] + stock_records)
        return [doc_id] + [r["id"] for r in stock_records]

    # ---- Query helpers ----

    def query_records(
        self,
        source: Optional[str] = None,
        sentiment: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ticker: Optional[str] = None,
        data_source: Optional[str] = None,
        stocks_only: bool = False,
        documents_only: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_desc: bool = True,
    ) -> Tuple[List[Dict], int]:
        """Query sentiment records with filters and pagination."""
        session = get_session()
        try:
            query = session.query(SentimentRecordRow)

            if stocks_only:
                query = query.filter(SentimentRecordRow.ticker.isnot(None))
            elif documents_only:
                query = query.filter(SentimentRecordRow.ticker.is_(None))
            if source:
                query = query.filter(SentimentRecordRow.source == source)
            if data_source:
                query = query.filter(SentimentRecordRow.data_source == data_source)
            if sentiment:
                query = query.filter(SentimentRecordRow.sentiment_label == sentiment)
            if start_date:
                query = query.filter(SentimentRecordRow.timestamp >= start_date)
            if end_date:
                query = query.filter(SentimentRecordRow.timestamp <= end_date)
            if ticker:
                query = query.filter(SentimentRecordRow.ticker == ticker)

            total = query.order_by(None).count()

            if order_desc:
                query = query.order_by(SentimentRecordRow.timestamp.desc())
            else:
                query = query.order_by(SentimentRecordRow.timestamp.asc())

            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            rows = query.all()
            return [row.to_dict() for row in rows], total
        finally:
            session.close()

    def get_all_sentiments(self) -> List[Dict]:
        """Get all sentiment records."""
        records, _ = self.query_records()
        return records

    def get_record_by_id(self, record_id: str) -> Optional[Dict]:
        """Fetch a single record by ID."""
        session = get_session()
        try:
            row = (
                session.query(SentimentRecordRow)
                .filter(SentimentRecordRow.id == record_id)
                .first()
            )
            return row.to_dict() if row else None
        finally:
            session.close()

    def get_stock_sentiment(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: Optional[str] = None,
    ) -> List[Dict]:
        """Get sentiment records for a specific stock."""
        ticker = ticker.upper()
        records, _ = self.query_records(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            source=source,
            stocks_only=True,
        )
        return records

    def get_stock_sentiments(self, ticker: str) -> List[Dict]:
        """Alias for get_stock_sentiment (backwards compatibility)."""
        return self.get_stock_sentiment(ticker)

    def get_trending_stocks(self, min_mentions: int = 5, hours: int = 24) -> List[Dict]:
        """Get stocks with most mentions in recent period."""
        session = get_session()
        try:
            latest = (
                session.query(func.max(SentimentRecordRow.timestamp))
                .filter(SentimentRecordRow.ticker.isnot(None))
                .scalar()
            )
            anchor = latest if latest else datetime.utcnow()
            cutoff_dt = anchor - timedelta(hours=hours)

            results = (
                session.query(
                    SentimentRecordRow.ticker,
                    func.count(SentimentRecordRow.id).label("mentions"),
                )
                .filter(SentimentRecordRow.ticker.isnot(None))
                .filter(SentimentRecordRow.timestamp >= cutoff_dt)
                .group_by(SentimentRecordRow.ticker)
                .having(func.count(SentimentRecordRow.id) >= min_mentions)
                .order_by(func.count(SentimentRecordRow.id).desc())
                .all()
            )
            return [{"ticker": r.ticker, "mentions": r.mentions} for r in results]
        finally:
            session.close()

    def aggregate_sentiment(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """Aggregate sentiment for a ticker."""
        session = get_session()
        try:
            query = session.query(SentimentRecordRow).filter(
                SentimentRecordRow.ticker == ticker,
            )
            if start_date:
                query = query.filter(SentimentRecordRow.timestamp >= start_date)
            if end_date:
                query = query.filter(SentimentRecordRow.timestamp <= end_date)

            total = query.count()
            if total == 0:
                return {
                    "ticker": ticker,
                    "total_mentions": 0,
                    "average_score": 0.0,
                    "sentiment_distribution": {
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0,
                    },
                }

            avg_query = session.query(
                func.avg(SentimentRecordRow.sentiment_score)
            ).filter(SentimentRecordRow.ticker == ticker)
            if start_date:
                avg_query = avg_query.filter(SentimentRecordRow.timestamp >= start_date)
            if end_date:
                avg_query = avg_query.filter(SentimentRecordRow.timestamp <= end_date)
            avg_score = avg_query.scalar()

            dist_query = session.query(
                SentimentRecordRow.sentiment_label,
                func.count(SentimentRecordRow.id),
            ).filter(SentimentRecordRow.ticker == ticker)
            if start_date:
                dist_query = dist_query.filter(
                    SentimentRecordRow.timestamp >= start_date
                )
            if end_date:
                dist_query = dist_query.filter(SentimentRecordRow.timestamp <= end_date)
            dist_rows = dist_query.group_by(SentimentRecordRow.sentiment_label).all()

            distribution = {"positive": 0, "negative": 0, "neutral": 0}
            for label, count in dist_rows:
                distribution[label] = int(count)

            return {
                "ticker": ticker,
                "total_mentions": total,
                "average_score": round(float(avg_score or 0.0), 4),
                "sentiment_distribution": distribution,
            }
        finally:
            session.close()

    def get_statistics(self) -> Dict:
        """Get overall stock sentiment statistics (stock records only)."""
        session = get_session()
        try:
            total = (
                session.query(func.count(SentimentRecordRow.id))
                .filter(SentimentRecordRow.ticker.isnot(None))
                .scalar()
                or 0
            )
            unique_tickers = (
                session.query(func.count(func.distinct(SentimentRecordRow.ticker)))
                .filter(SentimentRecordRow.ticker.isnot(None))
                .scalar()
                or 0
            )
            last_updated = (
                session.query(func.max(SentimentRecordRow.timestamp))
                .filter(SentimentRecordRow.ticker.isnot(None))
                .scalar()
            )
            last_updated_str = last_updated.isoformat() + "Z" if last_updated else None
            return {
                "total_records": int(total),
                "unique_tickers": int(unique_tickers),
                "last_updated": last_updated_str,
            }
        finally:
            session.close()
