"""Tests for import service record normalization and persistence wiring."""

from app.services.import_service import ImportService


class DummyStorage:
    """Captures rows written by import service."""

    def __init__(self):
        self.saved_rows = []

    def save_records_batch(self, rows):
        self.saved_rows.extend(rows)
        return len(rows)


def test_import_from_records_runs_sentiment_and_saves():
    """Import service should classify input text and save document records."""
    storage = DummyStorage()
    service = ImportService(
        storage=storage,
        analyzer=lambda texts, **kwargs: [
            {"label": "neutral", "score": 0.5} for _ in texts
        ],
    )

    result = service.import_from_records(
        [
            {
                "id": "r1",
                "title": "AAPL earnings beat estimates",
                "selftext": "Investors reacted positively.",
                "source": "reddit",
                "created_utc": 1764064812,
                "ticker": "aapl",
            }
        ]
    )

    assert result["records_loaded"] == 1
    assert result["records_inserted"] == 1
    assert len(storage.saved_rows) == 1
    assert storage.saved_rows[0]["ticker"] == "AAPL"
    assert storage.saved_rows[0]["sentiment_label"] == "neutral"
