"""Tests for canonical v1 statistics endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class DummyStatisticsService:
    """Stub service returning deterministic statistics payload."""

    def get_statistics(self):
        return {
            "total_predictions": 12,
            "total_stocks_analyzed": 3,
            "sentiment_distribution": {
                "positive": 6,
                "negative": 3,
                "neutral": 3,
                "positive_percentage": 50.0,
                "negative_percentage": 25.0,
                "neutral_percentage": 25.0,
            },
            "top_stocks": [
                {
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "count": 4,
                    "positive": 2,
                    "negative": 1,
                    "neutral": 1,
                }
            ],
            "recent_activity": {"last_24h": 2, "last_7d": 7, "last_30d": 12},
            "date_range": {
                "earliest": "2026-01-01T00:00:00Z",
                "latest": "2026-02-23T00:00:00Z",
            },
        }


def test_v1_statistics_shape(monkeypatch):
    """Statistics endpoint returns frontend-compatible shape."""
    from app.api.v1 import data

    monkeypatch.setattr(data, "statistics_service", DummyStatisticsService())
    response = client.get("/api/v1/data/statistics")

    assert response.status_code == 200
    payload = response.json()

    assert payload["total_predictions"] == 12
    assert payload["total_stocks_analyzed"] == 3
    assert "sentiment_distribution" in payload
    assert "top_stocks" in payload
    assert "recent_activity" in payload
    assert "date_range" in payload

