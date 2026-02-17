"""
SQLite-backed stock sentiment storage.

Drop-in replacement for the JSON-based StockSentimentStorage,
using SQLAlchemy for efficient querying with database indexes.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import func

from .database import StockSentimentRow, get_session, get_engine, migrate_from_json

logger = logging.getLogger(__name__)


class SQLiteStockSentimentStorage:
    """SQLite-backed storage for stock sentiment records."""

    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            backend_dir = Path(__file__).parent.parent.parent
            storage_dir = backend_dir.parent / "data" / "stock_sentiments"

        self.storage_dir = Path(storage_dir)
        self.json_file = self.storage_dir / "stock_sentiments.json"
        self._loaded = False

        # Initialize engine (creates tables if needed)
        get_engine()

    def load(self) -> None:
        """Load data - migrates from JSON if database is empty."""
        if self._loaded:
            return

        session = get_session()
        try:
            count = session.query(StockSentimentRow).count()
            if count == 0 and self.json_file.exists():
                logger.info("SQLite database is empty. Migrating from JSON...")
                migrate_from_json(self.json_file)
                count = session.query(StockSentimentRow).count()

            logger.info(f"SQLite storage loaded: {count} records")
            self._loaded = True
        finally:
            session.close()

    def save(self) -> None:
        """No-op for SQLite (auto-committed)."""
        pass

    def save_stock_sentiment(
        self,
        ticker: str,
        sentiment: Dict,
        context: str,
        mentioned_as: str,
        source: Optional[str] = None,
        source_id: Optional[str] = None,
        full_text: Optional[str] = None,
        position: Optional[Dict] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Save a single stock sentiment record."""
        if not self._loaded:
            self.load()

        record_id = f"{ticker}_{int(datetime.now().timestamp() * 1000)}"

        session = get_session()
        try:
            row = StockSentimentRow(
                id=record_id,
                ticker=ticker,
                mentioned_as=mentioned_as,
                sentiment_label=sentiment["label"],
                sentiment_score=sentiment["score"],
                context=context,
                source=source or "",
                source_id=source_id or "",
                full_text=full_text,
                position=None,
                timestamp=timestamp.isoformat() if timestamp else datetime.now().isoformat(),
                sentiment_mode="keyword",
            )
            session.add(row)
            session.commit()
            return record_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save sentiment: {e}")
            raise
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
        if not self._loaded:
            self.load()

        session = get_session()
        try:
            query = session.query(StockSentimentRow).filter(
                StockSentimentRow.ticker == ticker
            )

            if start_date:
                query = query.filter(StockSentimentRow.timestamp >= start_date.isoformat())
            if end_date:
                query = query.filter(StockSentimentRow.timestamp <= end_date.isoformat())
            if source:
                query = query.filter(StockSentimentRow.source == source)

            return [row.to_dict() for row in query.all()]
        finally:
            session.close()

    def get_trending_stocks(self, min_mentions: int = 5, hours: int = 24) -> List[Dict]:
        """Get stocks with most mentions in recent period."""
        if not self._loaded:
            self.load()

        cutoff_dt = datetime.fromtimestamp(
            datetime.now().timestamp() - (hours * 3600)
        )
        cutoff_str = cutoff_dt.isoformat()

        session = get_session()
        try:
            results = (
                session.query(
                    StockSentimentRow.ticker,
                    func.count(StockSentimentRow.id).label("mentions"),
                )
                .filter(StockSentimentRow.timestamp >= cutoff_str)
                .group_by(StockSentimentRow.ticker)
                .having(func.count(StockSentimentRow.id) >= min_mentions)
                .order_by(func.count(StockSentimentRow.id).desc())
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
        records = self.get_stock_sentiment(ticker, start_date, end_date)

        if not records:
            return {
                "ticker": ticker,
                "total_mentions": 0,
                "average_score": 0.0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            }

        total = len(records)
        scores = [r["sentiment_score"] for r in records]
        avg_score = sum(scores) / total

        distribution = {"positive": 0, "negative": 0, "neutral": 0}
        for record in records:
            label = record["sentiment_label"]
            distribution[label] = distribution.get(label, 0) + 1

        return {
            "ticker": ticker,
            "total_mentions": total,
            "average_score": round(avg_score, 4),
            "sentiment_distribution": distribution,
        }

    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        if not self._loaded:
            self.load()

        session = get_session()
        try:
            total = session.query(StockSentimentRow).count()
            unique_tickers = session.query(
                func.count(func.distinct(StockSentimentRow.ticker))
            ).scalar()

            return {
                "total_records": total,
                "unique_tickers": unique_tickers or 0,
                "last_updated": datetime.now().isoformat(),
            }
        finally:
            session.close()

    def get_all_sentiments(self) -> List[Dict]:
        """Get all sentiment records."""
        if not self._loaded:
            self.load()

        session = get_session()
        try:
            rows = session.query(StockSentimentRow).all()
            return [row.to_dict() for row in rows]
        finally:
            session.close()

    def get_stock_sentiments(self, ticker: str) -> List[Dict]:
        """Get all sentiment records for a specific stock ticker."""
        return self.get_stock_sentiment(ticker)
