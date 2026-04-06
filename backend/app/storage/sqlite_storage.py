"""
PostgreSQL-backed unified sentiment storage (Neon-compatible).

Stores both document-level predictions and per-stock mentions in a single table.
Stock rows are identified by having a non-null ticker column.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError

from .database import SentimentRecordRow, get_session
from .record_ids import make_record_id

logger = logging.getLogger(__name__)

BATCH_SIZE = 500
_MAX_AUTO_PRUNE_ROUNDS = 20


def _storage_pressure_error(exc: BaseException) -> bool:
    """True when the driver/DB reports disk or storage quota exhaustion."""
    e: BaseException | None = exc
    while e is not None:
        if type(e).__name__ == "DiskFull":
            return True
        msg = str(e).lower()
        if "diskfull" in msg.replace(" ", "") or "no space left" in msg:
            return True
        if "53100" in msg:
            return True
        if "storage limit" in msg or "disk quota" in msg:
            return True
        e = getattr(e, "__cause__", None) or getattr(e, "orig", None)  # type: ignore[assignment]
    return False


class SentimentStorage:
    """PostgreSQL-backed storage for unified sentiment records."""

    # CorrelationAnalyzer and legacy code expect this flag; DB is always ready.
    _loaded = True

    def __init__(self) -> None:
        # Defer get_engine() to first get_session() — avoids DB + create_all during
        # app import (critical for PaaS port checks and Docker without DATABASE_URL).
        pass

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

    def prune_oldest_published(
        self,
        *,
        limit: int,
        source: Optional[str] = None,
        data_source: Optional[str] = None,
    ) -> int:
        """
        Delete up to ``limit`` rows with the oldest ``published_at`` (ties broken by ``id``).

        Optional ``source`` / ``data_source`` narrow the candidate set; omit both to prune globally.
        """
        if limit <= 0:
            return 0
        parts = ["1=1"]
        params: Dict[str, object] = {"lim": limit}
        if source is not None:
            parts.append("source = :source")
            params["source"] = source
        if data_source is not None:
            parts.append("data_source = :data_source")
            params["data_source"] = data_source
        where_sql = " AND ".join(parts)
        delete_sql = text(
            f"""
            DELETE FROM sentiment_records
            WHERE id IN (
                SELECT id FROM sentiment_records
                WHERE {where_sql}
                ORDER BY published_at ASC, id ASC
                LIMIT :lim
            )
            """
        )
        session = get_session()
        try:
            result = session.execute(delete_sql, params)
            session.commit()
            deleted = int(result.rowcount or 0)
            if deleted:
                logger.info(
                    "Pruned %s sentiment row(s) (oldest published_at; filters source=%r data_source=%r)",
                    deleted,
                    source,
                    data_source,
                )
            return deleted
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _insert_records(self, records: List[Dict]) -> int:
        if not records:
            return 0
        auto_prune = os.environ.get("SENTIMENT_AUTO_PRUNE_ON_STORAGE_ERROR") == "1"
        prune_batch = int(os.environ.get("SENTIMENT_AUTO_PRUNE_BATCH", "5000"))
        prune_rounds = 0

        while True:
            session = get_session()
            try:
                total_inserted = 0
                for i in range(0, len(records), BATCH_SIZE):
                    batch = records[i : i + BATCH_SIZE]
                    stmt = pg_insert(SentimentRecordRow).values(batch)
                    stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
                    result = session.execute(stmt)
                    total_inserted += int(result.rowcount or 0)

                session.commit()
                return total_inserted
            except OperationalError as e:
                session.rollback()
                if (
                    auto_prune
                    and _storage_pressure_error(e)
                    and prune_rounds < _MAX_AUTO_PRUNE_ROUNDS
                ):
                    prune_rounds += 1
                    deleted = self.prune_oldest_published(limit=prune_batch)
                    logger.warning(
                        "Insert hit storage limit; pruned %s oldest row(s) (round %s/%s), retrying",
                        deleted,
                        prune_rounds,
                        _MAX_AUTO_PRUNE_ROUNDS,
                    )
                    if deleted == 0:
                        logger.error("Failed to save records: %s", e)
                        raise
                    continue
                logger.error("Failed to save records: %s", e)
                raise
            except Exception as e:
                session.rollback()
                logger.error("Failed to save records: %s", e)
                raise
            finally:
                session.close()

    def save_records_batch(self, records: List[Dict]) -> int:
        """Save multiple sentiment records."""
        normalized: List[Dict] = []
        for record in records:
            ts_val = record.get("published_at")
            if isinstance(ts_val, datetime):
                ts = ts_val
            else:
                ts = self._normalize_timestamp(ts_val)

            src = (record.get("source", "") or "")[:64]
            dsrc = record.get("data_source")
            if isinstance(dsrc, str):
                dsrc = dsrc[:64]
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
                    "source": src,
                    "data_source": dsrc,
                    "source_id": record.get("source_id", "") or "",
                    "source_meta_json": record.get("source_meta_json"),
                    "published_at": ts,
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
        now_iso = datetime.utcnow().isoformat() + "Z"

        source_id = result.get("source_id", "")
        doc_id = result.get("document_id") or make_record_id(
            "doc", source or "", source_id, now_iso, text[:200]
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
            "source_meta_json": result.get("source_meta_json"),
            "published_at": now_iso,
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
                    "source_meta_json": result.get("source_meta_json"),
                    "published_at": now_iso,
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
                query = query.filter(SentimentRecordRow.published_at >= start_date)
            if end_date:
                query = query.filter(SentimentRecordRow.published_at <= end_date)
            if ticker:
                query = query.filter(SentimentRecordRow.ticker == ticker)

            total = query.order_by(None).count()

            if order_desc:
                query = query.order_by(SentimentRecordRow.published_at.desc())
            else:
                query = query.order_by(SentimentRecordRow.published_at.asc())

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
        data_source: Optional[str] = None,
    ) -> List[Dict]:
        """Get sentiment records for a specific stock."""
        ticker = ticker.upper()
        records, _ = self.query_records(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            source=source,
            data_source=data_source,
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
                session.query(func.max(SentimentRecordRow.published_at))
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
                .filter(SentimentRecordRow.published_at >= cutoff_dt)
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
                query = query.filter(SentimentRecordRow.published_at >= start_date)
            if end_date:
                query = query.filter(SentimentRecordRow.published_at <= end_date)

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
                avg_query = avg_query.filter(SentimentRecordRow.published_at >= start_date)
            if end_date:
                avg_query = avg_query.filter(SentimentRecordRow.published_at <= end_date)
            avg_score = avg_query.scalar()

            dist_query = session.query(
                SentimentRecordRow.sentiment_label,
                func.count(SentimentRecordRow.id),
            ).filter(SentimentRecordRow.ticker == ticker)
            if start_date:
                dist_query = dist_query.filter(
                    SentimentRecordRow.published_at >= start_date
                )
            if end_date:
                dist_query = dist_query.filter(SentimentRecordRow.published_at <= end_date)
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
                session.query(func.max(SentimentRecordRow.published_at))
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


# Backward-compatible alias (legacy name; implementation is PostgreSQL-only).
SQLiteSentimentStorage = SentimentStorage
