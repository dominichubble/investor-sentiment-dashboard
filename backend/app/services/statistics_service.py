"""Statistics aggregation service backed by sentiment record storage."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, case, cast, Date

from app.entities.stock_database import StockDatabase
from app.storage.database import SentimentRecordRow, get_session


def _to_utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() + "Z"


class StatisticsService:
    """Build dashboard statistics from persisted sentiment records."""

    def __init__(self) -> None:
        self.stock_db = StockDatabase()
        self.stock_db.load()

    def get_statistics(self, days: int | None = None) -> dict[str, Any]:
        """Return statistics payload aligned with frontend expectations.

        Args:
            days: If provided, only include records from the last N days
                  relative to the latest record timestamp. None means all data.
        """
        session = get_session()
        try:
            # Determine the anchor (latest record) for relative calculations.
            latest_ts = (
                session.query(func.max(SentimentRecordRow.published_at)).scalar()
            )
            anchor = latest_ts if latest_ts else datetime.utcnow()

            # Base query filter: optionally restrict to the last N days.
            def _apply_window(query):
                if days is not None:
                    cutoff = anchor - timedelta(days=days)
                    return query.filter(SentimentRecordRow.published_at >= cutoff)
                return query

            total_predictions = (
                _apply_window(
                    session.query(func.count(SentimentRecordRow.id))
                ).scalar()
                or 0
            )

            sentiment_rows = (
                _apply_window(
                    session.query(
                        SentimentRecordRow.sentiment_label,
                        func.count(SentimentRecordRow.id).label("count"),
                    )
                )
                .group_by(SentimentRecordRow.sentiment_label)
                .all()
            )
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for label, count in sentiment_rows:
                if label in sentiment_counts:
                    sentiment_counts[label] = int(count)

            total_with_sentiment = sum(sentiment_counts.values())
            sentiment_distribution = {
                "positive": sentiment_counts["positive"],
                "negative": sentiment_counts["negative"],
                "neutral": sentiment_counts["neutral"],
                "positive_percentage": (
                    round(
                        (sentiment_counts["positive"] / total_with_sentiment) * 100, 2
                    )
                    if total_with_sentiment
                    else 0.0
                ),
                "negative_percentage": (
                    round(
                        (sentiment_counts["negative"] / total_with_sentiment) * 100, 2
                    )
                    if total_with_sentiment
                    else 0.0
                ),
                "neutral_percentage": (
                    round((sentiment_counts["neutral"] / total_with_sentiment) * 100, 2)
                    if total_with_sentiment
                    else 0.0
                ),
            }

            stock_rows = (
                _apply_window(
                    session.query(
                        SentimentRecordRow.ticker,
                        SentimentRecordRow.sentiment_label,
                        func.count(SentimentRecordRow.id).label("count"),
                    )
                    .filter(SentimentRecordRow.ticker.isnot(None))
                )
                .group_by(SentimentRecordRow.ticker, SentimentRecordRow.sentiment_label)
                .all()
            )
            stock_map: dict[str, dict[str, Any]] = {}
            for ticker, label, count in stock_rows:
                if not ticker:
                    continue
                if ticker not in stock_map:
                    stock_info = self.stock_db.get_by_ticker(ticker) or {}
                    stock_map[ticker] = {
                        "ticker": ticker,
                        "company_name": stock_info.get("company_name", ticker),
                        "count": 0,
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0,
                    }
                stock_map[ticker]["count"] += int(count)
                if label in ("positive", "negative", "neutral"):
                    stock_map[ticker][label] += int(count)

            top_stocks = sorted(
                stock_map.values(),
                key=lambda item: item["count"],
                reverse=True,
            )[:10]

            total_stocks_analyzed = (
                _apply_window(
                    session.query(func.count(func.distinct(SentimentRecordRow.ticker)))
                    .filter(SentimentRecordRow.ticker.isnot(None))
                ).scalar()
                or 0
            )

            earliest_q = _apply_window(
                session.query(
                    func.min(SentimentRecordRow.published_at),
                    func.max(SentimentRecordRow.published_at),
                )
            )
            earliest, latest = earliest_q.one()

            # Use the latest record timestamp as the reference point so that
            # historical datasets (e.g. 2021-2022) show meaningful activity.
            recent_anchor = latest if latest else anchor
            recent_activity = {
                "last_24h": _apply_window(
                    session.query(func.count(SentimentRecordRow.id))
                    .filter(SentimentRecordRow.published_at >= recent_anchor - timedelta(days=1))
                ).scalar()
                or 0,
                "last_7d": _apply_window(
                    session.query(func.count(SentimentRecordRow.id))
                    .filter(SentimentRecordRow.published_at >= recent_anchor - timedelta(days=7))
                ).scalar()
                or 0,
                "last_30d": _apply_window(
                    session.query(func.count(SentimentRecordRow.id))
                    .filter(SentimentRecordRow.published_at >= recent_anchor - timedelta(days=30))
                ).scalar()
                or 0,
            }

            # ---- Daily trend time-series for mini-charts ----
            day_col = func.date(SentimentRecordRow.published_at).label("day")
            daily_rows = (
                _apply_window(
                    session.query(
                        day_col,
                        func.count(SentimentRecordRow.id).label("count"),
                        func.sum(
                            case(
                                (SentimentRecordRow.sentiment_label == "positive", 1),
                                else_=0,
                            )
                        ).label("pos"),
                        func.sum(
                            case(
                                (SentimentRecordRow.sentiment_label == "negative", 1),
                                else_=0,
                            )
                        ).label("neg"),
                    )
                )
                .group_by(day_col)
                .order_by(day_col)
                .all()
            )
            daily_trend = []
            for day, count, pos, neg in daily_rows:
                net = round(((pos or 0) - (neg or 0)) / max(count, 1), 4)
                daily_trend.append(
                    {"date": str(day), "count": int(count), "net_sentiment": net}
                )

            return {
                "total_predictions": int(total_predictions),
                "total_stocks_analyzed": int(total_stocks_analyzed),
                "sentiment_distribution": sentiment_distribution,
                "top_stocks": top_stocks,
                "recent_activity": {
                    "last_24h": int(recent_activity["last_24h"]),
                    "last_7d": int(recent_activity["last_7d"]),
                    "last_30d": int(recent_activity["last_30d"]),
                },
                "date_range": {
                    "earliest": _to_utc_iso(earliest),
                    "latest": _to_utc_iso(latest),
                },
                "daily_trend": daily_trend,
            }
        finally:
            session.close()
