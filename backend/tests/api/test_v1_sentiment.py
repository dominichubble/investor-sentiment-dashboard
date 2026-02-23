"""Tests for canonical v1 sentiment endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_v1_analyze_happy_path(monkeypatch):
    """Analyze endpoint returns label/score payload."""
    from app.api.v1 import sentiment

    monkeypatch.setattr(
        sentiment,
        "run_sentiment",
        lambda text, return_all_scores=False: {"label": "positive", "score": 0.9},
    )

    response = client.post(
        "/api/v1/sentiment/analyze", json={"text": "Stocks rallied."}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "Stocks rallied."
    assert payload["sentiment"]["label"] == "positive"
    assert payload["sentiment"]["score"] == 0.9
    assert "metadata" in payload


def test_v1_analyze_empty_text_returns_400():
    """Analyze endpoint should reject blank text with a clear 400."""
    response = client.post("/api/v1/sentiment/analyze", json={"text": ""})
    assert response.status_code == 400
    assert "non-empty" in response.json()["detail"]


def test_v1_batch_happy_path(monkeypatch):
    """Batch endpoint should preserve input order."""
    from app.api.v1 import sentiment

    monkeypatch.setattr(
        sentiment,
        "run_batch_sentiment",
        lambda texts, **kwargs: [
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.7},
        ],
    )

    response = client.post(
        "/api/v1/sentiment/batch",
        json={"texts": ["good", "bad"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 2
    assert payload["results"][0]["text"] == "good"
    assert payload["results"][1]["text"] == "bad"


def test_v1_batch_empty_list_returns_400():
    """Batch endpoint should reject empty inputs."""
    response = client.post("/api/v1/sentiment/batch", json={"texts": []})
    assert response.status_code == 400
