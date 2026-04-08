"""Tests for import service record normalization and persistence wiring."""

import json

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
                "subreddit": "stocks",
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
        assert r["data_source"] == "reddit"
        assert r["emotion_label"]
        assert json.loads(r["emotion_scores_json"])
        assert r["emotion_rationale"]


def test_import_merges_detected_and_hint_tickers():
    """hint_tickers add symbols not spelled in text; detection still runs."""
    storage = DummyStorage()
    service = _make_service(storage)

    service.import_from_records(
        [
            {
                "text": "$AAPL outlook strong versus peers",
                "source": "twitter",
                "author_id": "1",
                "timestamp": "2025-06-01T12:00:00Z",
                "hint_tickers": ["MSFT"],
            }
        ]
    )

    tickers = {r["ticker"] for r in storage.saved_rows}
    assert tickers == {"AAPL", "MSFT"}


def test_import_hint_tickers_when_text_has_no_symbol():
    """Reddit-style search hits can carry hint_tickers so rows still persist."""
    storage = DummyStorage()
    service = _make_service(storage)

    result = service.import_from_records(
        [
            {
                "id": "abc123",
                "title": "Thoughts on this quarter?",
                "selftext": "Seems overvalued but holding.",
                "source": "wallstreetbets",
                "subreddit": "wallstreetbets",
                "data_source": "reddit",
                "created_utc": 1764064812,
                "hint_tickers": ["NVDA"],
            }
        ]
    )

    assert result["records_inserted"] >= 1
    assert any(r["ticker"] == "NVDA" for r in storage.saved_rows)


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


def test_import_document_fallback_without_ticker():
    """Optional fallback stores document-level row when no ticker is found."""
    storage = DummyStorage()
    service = _make_service(storage)

    result = service.import_from_records(
        [
            {
                "text": "Markets look uncertain today.",
                "source": "news",
                "timestamp": "2025-01-01T00:00:00Z",
            }
        ],
        document_fallback_without_ticker=True,
    )

    assert result["records_loaded"] == 1
    assert result["records_inserted"] == 1
    assert len(storage.saved_rows) == 1
    assert storage.saved_rows[0]["ticker"] is None
    assert storage.saved_rows[0]["mentioned_as"] == ""
    assert storage.saved_rows[0]["id"].startswith("doc_")


def test_import_multiple_tickers_produce_separate_rows():
    """Each detected ticker gets its own stock row."""
    storage = DummyStorage()
    service = _make_service(storage)

    service.import_from_records(
        [
            {
                "text": "$AAPL up 3%, $MSFT down 1%, $GOOGL flat.",
                "source": "twitter",
                "author_id": "999",
                "timestamp": "2025-06-01T12:00:00Z",
            }
        ]
    )

    tickers = {r["ticker"] for r in storage.saved_rows}
    assert {"AAPL", "MSFT", "GOOGL"}.issubset(tickers)
    assert all(r["ticker"] is not None for r in storage.saved_rows)
    assert all(r["data_source"] == "twitter" for r in storage.saved_rows)


def test_import_news_infers_data_source():
    storage = DummyStorage()
    service = _make_service(storage)
    service.import_from_records(
        [
            {
                "clean_title": "Nvidia outlook",
                "source_name": "Reuters",
                "timestamp": "2025-06-01T12:00:00Z",
            }
        ]
    )
    assert storage.saved_rows
    assert all(r["data_source"] == "news" for r in storage.saved_rows)


def test_import_news_source_name_beats_generic_source_key():
    """Generic ``source: news`` must not hide the publisher in ``source_name``."""
    storage = DummyStorage()
    service = _make_service(storage)
    service.import_from_records(
        [
            {
                "clean_title": "Fed holds rates; $AAPL steady",
                "source": "news",
                "source_name": "Bloomberg",
                "timestamp": "2025-06-01T12:00:00Z",
            }
        ]
    )
    assert storage.saved_rows
    assert storage.saved_rows[0]["source"] == "bloomberg"
