"""Statistics aggregation service backed by sentiment record storage."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable

from sqlalchemy import func, case
from sqlalchemy.orm import Query

from app.analysis.source_disagreement import (
    MIN_ROWS_PER_CHANNEL_PER_DAY,
    STANDARD_CHANNELS,
    disagreement_metrics,
)
from app.entities.stock_database import StockDatabase
from app.storage.database import SentimentRecordRow, get_session


def _to_utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() + "Z"


def _normalize_data_source_param(raw: str | None) -> str | None:
    """Map query param to DB ``data_source`` value (reddit, news, twitter)."""
    if not raw:
        return None
    s = raw.strip().lower()
    if s in ("all", "", "any"):
        return None
    if s in ("x", "twitter", "tweet", "tweets"):
        return "twitter"
    if s in ("reddit",):
        return "reddit"
    if s in ("news", "article", "articles"):
        return "news"
    return None


def _build_source_disagreement_trend(
    session,
    apply_window: Callable[[Query], Query],
) -> list[dict[str, Any]]:
    """
    Per calendar day: net sentiment by channel (reddit / news / twitter), then
    cross-source disagreement (range and std of nets) when ≥2 channels meet
    the minimum row threshold.
    """
    day_col = func.date(SentimentRecordRow.published_at).label("day")
    rows = (
        apply_window(
            session.query(
                day_col,
                SentimentRecordRow.data_source,
                func.count(SentimentRecordRow.id).label("cnt"),
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
        .filter(SentimentRecordRow.data_source.in_(list(STANDARD_CHANNELS)))
        .group_by(day_col, SentimentRecordRow.data_source)
        .order_by(day_col)
        .all()
    )

    by_day: dict[str, dict[str, tuple[int, int, int]]] = defaultdict(dict)
    for day, ds, cnt, pos, neg in rows:
        key = (ds or "").strip().lower()
        if key not in STANDARD_CHANNELS:
            continue
        c = int(cnt or 0)
        p = int(pos or 0)
        n = int(neg or 0)
        by_day[str(day)][key] = (c, p, n)

    out: list[dict[str, Any]] = []
    for day in sorted(by_day.keys()):
        chans = by_day[day]
        nets: dict[str, float] = {}
        counts: dict[str, int] = {}
        total_mentions = 0
        for ch in STANDARD_CHANNELS:
            if ch not in chans:
                continue
            c, p, n = chans[ch]
            total_mentions += c
            if c < MIN_ROWS_PER_CHANNEL_PER_DAY:
                continue
            nets[ch] = round((p - n) / max(c, 1), 4)
            counts[ch] = c

        d_range, d_std = disagreement_metrics(nets)
        out.append(
            {
                "date": day,
                "total_mentions": total_mentions,
                "n_sources_active": len(nets),
                "disagreement_range": d_range,
                "disagreement_std": d_std,
                "net_by_source": nets,
                "counts_by_source": counts,
            }
        )
    return out


def _sentiment_breakdown_from_counts(
    positive: int, negative: int, neutral: int
) -> dict[str, Any]:
    total = positive + negative + neutral
    if total <= 0:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "positive_percentage": 0.0,
            "negative_percentage": 0.0,
            "neutral_percentage": 0.0,
        }
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_percentage": round((positive / total) * 100, 2),
        "negative_percentage": round((negative / total) * 100, 2),
        "neutral_percentage": round((neutral / total) * 100, 2),
    }


class StatisticsService:
    """Build dashboard statistics from persisted sentiment records."""

    def __init__(self) -> None:
        self.stock_db = StockDatabase()

    def get_statistics(
        self,
        days: int | None = None,
        data_source: str | None = None,
        include_source_comparison: bool = True,
    ) -> dict[str, Any]:
        """Return statistics payload aligned with frontend expectations.

        Args:
            days: If provided, only include records from the last N days
                  relative to the latest record timestamp. None means all data.
            data_source: If set (reddit, news, twitter), restrict to that channel.
            include_source_comparison: When True and ``data_source`` is unset,
                include per-channel sentiment breakdown (``source_comparison``).
        """
        # Load after import so startup never blocks on SEC download / disk I/O.
        self.stock_db.load()
        session = get_session()
        try:
            ds_filter = _normalize_data_source_param(data_source)

            def _apply_scope(query: Query) -> Query:
                query = query.filter(SentimentRecordRow.ticker.isnot(None))
                if ds_filter:
                    query = query.filter(SentimentRecordRow.data_source == ds_filter)
                return query

            # Determine the anchor (latest stock-level record) for relative calculations.
            # When a source filter is active, anchor relative windows to that source only.
            latest_ts = _apply_scope(
                session.query(func.max(SentimentRecordRow.published_at))
            ).scalar()
            anchor = latest_ts if latest_ts else datetime.utcnow()

            # Base query filter: optionally restrict to the last N days.
            def _apply_window(query: Query) -> Query:
                if days is not None:
                    cutoff = anchor - timedelta(days=days)
                    return query.filter(SentimentRecordRow.published_at >= cutoff)
                return query

            def _apply_channel(query: Query) -> Query:
                return _apply_window(_apply_scope(query))

            total_predictions = (
                _apply_channel(session.query(func.count(SentimentRecordRow.id))).scalar()
                or 0
            )

            sentiment_rows = (
                _apply_channel(
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
                _apply_channel(
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
                _apply_channel(
                    session.query(func.count(func.distinct(SentimentRecordRow.ticker)))
                    .filter(SentimentRecordRow.ticker.isnot(None))
                ).scalar()
                or 0
            )

            earliest_q = _apply_channel(
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
                "last_24h": _apply_channel(
                    session.query(func.count(SentimentRecordRow.id))
                    .filter(SentimentRecordRow.published_at >= recent_anchor - timedelta(days=1))
                ).scalar()
                or 0,
                "last_7d": _apply_channel(
                    session.query(func.count(SentimentRecordRow.id))
                    .filter(SentimentRecordRow.published_at >= recent_anchor - timedelta(days=7))
                ).scalar()
                or 0,
                "last_30d": _apply_channel(
                    session.query(func.count(SentimentRecordRow.id))
                    .filter(SentimentRecordRow.published_at >= recent_anchor - timedelta(days=30))
                ).scalar()
                or 0,
            }

            # ---- Daily trend time-series for mini-charts ----
            day_col = func.date(SentimentRecordRow.published_at).label("day")
            daily_rows = (
                _apply_channel(
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

            source_comparison: dict[str, Any] | None = None
            if include_source_comparison and ds_filter is None:
                src_rows = (
                    _apply_window(
                        session.query(
                            SentimentRecordRow.data_source,
                            SentimentRecordRow.sentiment_label,
                            func.count(SentimentRecordRow.id),
                        )
                        .filter(SentimentRecordRow.ticker.isnot(None))
                    )
                    .group_by(
                        SentimentRecordRow.data_source,
                        SentimentRecordRow.sentiment_label,
                    )
                    .all()
                )
                acc: dict[str, dict[str, int]] = {
                    "reddit": {"positive": 0, "negative": 0, "neutral": 0},
                    "news": {"positive": 0, "negative": 0, "neutral": 0},
                    "twitter": {"positive": 0, "negative": 0, "neutral": 0},
                }
                for src, label, cnt in src_rows:
                    key = (src or "").strip().lower()
                    if key not in acc:
                        continue
                    if label in acc[key]:
                        acc[key][label] += int(cnt)
                source_comparison = {
                    ch: _sentiment_breakdown_from_counts(
                        v["positive"], v["negative"], v["neutral"]
                    )
                    for ch, v in acc.items()
                }

            out: dict[str, Any] = {
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
            if source_comparison is not None:
                out["source_comparison"] = source_comparison
                out["source_disagreement_trend"] = _build_source_disagreement_trend(
                    session, _apply_window
                )
            else:
                out["source_disagreement_trend"] = []
            return out
        finally:
            session.close()
