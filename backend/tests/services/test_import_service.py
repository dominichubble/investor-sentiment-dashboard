"""Tests for import service record normalization and persistence wiring."""

from unittest.mock import patch

from app.services.import_service import ImportService


class DummyStorage:
    """Captures rows written by import service."""

    def __init__(self):
        self.saved_rows = []

    def save_records_batch(self, rows):
        self.saved_rows.extend(rows)
        return len(rows)


def _make_service(storage=None):
    return ImportService(
        storage=storage or DummyStorage(),
        analyzer=lambda texts, **kwargs: [
            {"label": "neutral", "score": 0.5} for _ in texts
        ],
    )


def test_import_creates_document_and_stock_rows():
    """Text mentioning a ticker should produce both a document row and a stock row."""
    storage = DummyStorage()
    service = _make_service(storage)

    result = service.import_from_records(
        [
            {
                "id": "r1",
                "title": "AAPL earnings beat estimates",
                "selftext": "Investors reacted positively to $TSLA as well.",
                "source": "reddit",
                "created_utc": 1764064812,
            }
        ]
    )

    assert result["records_loaded"] == 1
    doc_rows = [r for r in storage.saved_rows if r["ticker"] is None]
    stock_rows = [r for r in storage.saved_rows if r["ticker"] is not None]

    assert len(doc_rows) == 1
    assert doc_rows[0]["sentiment_label"] == "neutral"
    assert doc_rows[0]["ticker"] is None

    tickers_found = {r["ticker"] for r in stock_rows}
    # Both AAPL (bare) and TSLA ($TSLA cashtag) should be detected.
    assert "TSLA" in tickers_found


def test_import_no_ticker_still_creates_document():
    """Generic text with no recognisable ticker creates a document row only."""
    storage = DummyStorage()
    service = _make_service(storage)

    result = service.import_from_records(
        [
            {
                "text": "Markets look uncertain today.",
                "source": "news",
                "timestamp": "2025-01-01T00:00:00Z",
            }
        ]
    )

    assert result["records_loaded"] == 1
    assert all(r["ticker"] is None for r in storage.saved_rows)


def test_import_multiple_tickers_produce_separate_rows():
    """Each detected ticker gets its own stock row."""
    storage = DummyStorage()
    service = _make_service(storage)

    service.import_from_records(
        [
            {
                "text": "$AAPL up 3%, $MSFT down 1%, $GOOGL flat.",
                "source": "twitter",
                "timestamp": "2025-06-01T12:00:00Z",
            }
        ]
    )

    stock_rows = [r for r in storage.saved_rows if r["ticker"] is not None]
    tickers = {r["ticker"] for r in stock_rows}
    assert {"AAPL", "MSFT", "GOOGL"}.issubset(tickers)
    for sr in stock_rows:
        assert sr["sentiment_label"] == "neutral"
