"""Tests for stock-scoped and source-scoped statistics aggregation."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.statistics_service import StatisticsService
from app.storage.database import Base, SentimentRecordRow


def _make_row(
    *,
    row_id: str,
    published_at: str,
    ticker: str | None,
    data_source: str | None,
    sentiment_label: str,
    source: str = "",
) -> SentimentRecordRow:
    return SentimentRecordRow(
        id=row_id,
        text=f"text-{row_id}",
        ticker=ticker,
        mentioned_as=f"${ticker}" if ticker else "",
        sentiment_label=sentiment_label,
        sentiment_score=0.75,
        source=source,
        data_source=data_source,
        source_id=f"src-{row_id}",
        published_at=datetime.fromisoformat(published_at),
    )


def _make_service(monkeypatch, rows: list[SentimentRecordRow]) -> StatisticsService:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    session = SessionLocal()
    try:
        session.add_all(rows)
        session.commit()
    finally:
        session.close()

    monkeypatch.setattr(
        "app.services.statistics_service.get_session",
        SessionLocal,
    )

    class DummyStockDB:
        def load(self) -> None:
            pass

        def get_by_ticker(self, ticker: str):
            return {"company_name": f"{ticker} Corp"}

    service = StatisticsService()
    service.stock_db = DummyStockDB()
    return service


def test_statistics_only_count_stock_rows(monkeypatch):
    service = _make_service(
        monkeypatch,
        [
            _make_row(
                row_id="doc-1",
                published_at="2026-03-01T12:00:00",
                ticker=None,
                data_source="news",
                sentiment_label="positive",
                source="reuters",
            ),
            _make_row(
                row_id="stock-1",
                published_at="2026-03-01T12:00:00",
                ticker="AAPL",
                data_source="news",
                sentiment_label="negative",
                source="reuters",
            ),
            _make_row(
                row_id="stock-2",
                published_at="2026-03-02T12:00:00",
                ticker="MSFT",
                data_source="reddit",
                sentiment_label="positive",
                source="stocks",
            ),
        ],
    )

    stats = service.get_statistics()

    assert stats["total_predictions"] == 2
    assert stats["total_stocks_analyzed"] == 2
    assert stats["sentiment_distribution"] == {
        "positive": 1,
        "negative": 1,
        "neutral": 0,
        "positive_percentage": 50.0,
        "negative_percentage": 50.0,
        "neutral_percentage": 0.0,
    }
    assert [stock["ticker"] for stock in stats["top_stocks"]] == ["AAPL", "MSFT"]
    assert stats["source_comparison"]["news"]["total"] == 1
    assert stats["source_comparison"]["reddit"]["total"] == 1


def test_source_filtered_days_anchor_uses_latest_row_in_that_source(monkeypatch):
    service = _make_service(
        monkeypatch,
        [
            _make_row(
                row_id="reddit-old",
                published_at="2026-01-01T12:00:00",
                ticker="TSLA",
                data_source="reddit",
                sentiment_label="positive",
                source="stocks",
            ),
            _make_row(
                row_id="reddit-latest",
                published_at="2026-01-10T12:00:00",
                ticker="TSLA",
                data_source="reddit",
                sentiment_label="negative",
                source="stocks",
            ),
            _make_row(
                row_id="news-newer",
                published_at="2026-04-01T12:00:00",
                ticker="TSLA",
                data_source="news",
                sentiment_label="positive",
                source="reuters",
            ),
        ],
    )

    stats = service.get_statistics(days=7, data_source="reddit")

    assert stats["total_predictions"] == 1
    assert stats["date_range"] == {
        "earliest": "2026-01-10T12:00:00Z",
        "latest": "2026-01-10T12:00:00Z",
    }
    assert stats["recent_activity"] == {
        "last_24h": 1,
        "last_7d": 1,
        "last_30d": 1,
    }
    assert stats["daily_trend"] == [
        {"date": "2026-01-10", "count": 1, "net_sentiment": -1.0}
    ]
