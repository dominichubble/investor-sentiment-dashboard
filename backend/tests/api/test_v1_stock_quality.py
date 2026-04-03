"""Tests for per-ticker data quality endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@patch("app.api.v1.data.get_stock_data_quality", autospec=True)
def test_stock_quality_returns_payload(mock_q):
    mock_q.return_value = {
        "ticker": "AAPL",
        "window_start": "2024-01-01T00:00:00Z",
        "window_end": "2024-03-01T23:59:59Z",
        "calendar_days": 61,
        "days_with_mentions": 20,
        "calendar_coverage": 0.328,
        "longest_gap_days": 5,
        "total_mentions": 120,
        "by_label": {"positive": 40, "neutral": 50, "negative": 30},
        "label_shares": {"positive": 0.33, "neutral": 0.42, "negative": 0.25},
        "by_channel": {"reddit": 80, "news": 40},
        "confidence_score": 0.72,
        "confidence_label": "high",
        "flags": [
            {
                "id": "moderate_sample",
                "severity": "info",
                "title": "Moderate sample size",
                "detail": "120 mentions — interpret edge cases cautiously.",
            }
        ],
    }
    r = client.get("/api/v1/data/stock-quality/AAPL?period=90d")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["total_mentions"] == 120
    assert body["confidence_label"] == "high"
    assert len(body["flags"]) == 1
    assert body["flags"][0]["id"] == "moderate_sample"
    mock_q.assert_called_once()


def test_stock_quality_custom_range_requires_both_dates():
    r = client.get(
        "/api/v1/data/stock-quality/MSFT",
        params={"start_date": "2024-01-01"},
    )
    assert r.status_code == 400
