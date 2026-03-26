"""Tests for import service record normalization and persistence wiring."""

from app.services.import_service import ImportService


class DummyStorage:
    """Captures rows written by import service."""

    def __init__(self):
        self.saved_rows = []

    def save_records_batch(self, rows):
        self.saved_rows.extend(rows)
        return len(rows)


def _stub_analyzer(texts, **kwargs):
    return [
        {
            "label": "neutral",
            "score": 0.5,
            "scores": {"positive": 0.2, "negative": 0.2, "neutral": 0.6},
        }
        for _ in texts
    ]


def _make_service(storage=None):
    return ImportService(
        storage=storage or DummyStorage(),
        analyzer=_stub_analyzer,
    )


def test_import_creates_stock_rows_per_ticker():
    """Text mentioning tickers should produce one stock row per ticker."""
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
    assert all(r["ticker"] is not None for r in storage.saved_rows)

    tickers_found = {r["ticker"] for r in storage.saved_rows}
    assert "TSLA" in tickers_found
    for r in storage.saved_rows:
        assert r["sentiment_label"] == "neutral"
        assert r["score_positive"] == 0.2
        assert r["rationale"]
        assert r["sentiment_uncertainty"] is not None


def test_import_no_ticker_skips_row():
    """Text with no recognisable ticker is not stored."""
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
    assert result["records_inserted"] == 0
    assert len(storage.saved_rows) == 0


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

    tickers = {r["ticker"] for r in storage.saved_rows}
    assert {"AAPL", "MSFT", "GOOGL"}.issubset(tickers)
    assert all(r["ticker"] is not None for r in storage.saved_rows)
