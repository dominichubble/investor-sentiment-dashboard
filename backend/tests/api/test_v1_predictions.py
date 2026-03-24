"""Tests for canonical v1 predictions endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class DummyStorage:
    """Simple fake storage for endpoint testing."""

    def query_records(self, **kwargs):
        rows = [
            {
                "id": "doc_1",
                "record_type": "document",
                "text": "Market sentiment is improving.",
                "sentiment_label": "positive",
                "sentiment_score": 0.91,
                "source": "reddit",
                "source_id": "abc",
                "timestamp": "2026-02-23T10:00:00Z",
            },
            {
                "id": "stock_1",
                "record_type": "stock",
                "text": "AAPL has momentum.",
                "ticker": "AAPL",
                "mentioned_as": "AAPL",
                "sentiment_label": "positive",
                "sentiment_score": 0.88,
                "source": "news",
                "timestamp": "2026-02-23T11:00:00Z",
            },
        ]
        return rows, 2


def test_v1_predictions_shape(monkeypatch):
    """Predictions endpoint returns frontend-compatible shape."""
    from app.api.v1 import data

    monkeypatch.setattr(data, "storage", DummyStorage())
    response = client.get("/api/v1/data/predictions?page=1&page_size=20")

    assert response.status_code == 200
    payload = response.json()

    assert payload["total"] == 2
    assert payload["page"] == 1
    assert payload["page_size"] == 20
    assert payload["has_more"] is False
    assert len(payload["predictions"]) == 2

    first = payload["predictions"][0]
    assert "id" in first
    assert "record_type" in first
    assert "text" in first
    assert "sentiment" in first
    assert "timestamp" in first


def test_v1_predictions_invalid_page():
    """Validation should reject invalid page values."""
    response = client.get("/api/v1/data/predictions?page=0")
    assert response.status_code == 422
