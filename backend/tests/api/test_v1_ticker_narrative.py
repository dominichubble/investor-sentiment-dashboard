"""Tests for grounded ticker sentiment narrative endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@patch("app.api.v1.sentiment.generate_ticker_narrative", autospec=True)
def test_ticker_narrative_returns_payload(mock_gen):
    mock_gen.return_value = {
        "narrative": "## Summary\nTest narrative.",
        "cached": True,
        "model": "llama-3.1-8b-instant",
        "record_count": 42,
        "window_start": "2024-01-01T00:00:00Z",
        "window_end": "2024-03-01T23:59:59Z",
        "period_key": "p:90d",
        "data_signature": "abc123",
        "error": None,
    }
    r = client.get("/api/v1/sentiment/ticker-narrative/AAPL?period=90d")
    assert r.status_code == 200
    body = r.json()
    assert body["narrative"].startswith("## Summary")
    assert body["cached"] is True
    assert body["record_count"] == 42
    assert body["period_key"] == "p:90d"
    mock_gen.assert_called_once()


def test_ticker_narrative_custom_range_requires_both_dates():
    r = client.get(
        "/api/v1/sentiment/ticker-narrative/MSFT",
        params={"start_date": "2024-01-01"},
    )
    assert r.status_code == 400
