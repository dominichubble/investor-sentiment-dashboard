"""Tests for canonical v1 explainability endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_v1_explain_happy_path(monkeypatch):
    """Explain endpoint should return token weights and prediction."""
    from app.api.v1 import sentiment

    monkeypatch.setattr(
        sentiment,
        "run_lime_explain",
        lambda text, num_features=12, num_samples=1000: {
            "prediction": {
                "label": "positive",
                "score": 0.92,
                "scores": {"positive": 0.92, "negative": 0.03, "neutral": 0.05},
            },
            "top_features": [("surged", 0.41), ("earnings", 0.27)],
        },
    )

    response = client.post(
        "/api/v1/sentiment/explain", json={"text": "Stocks surged on earnings."}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"]["label"] == "positive"
    assert len(payload["tokens"]) == 2
    assert payload["tokens"][0]["token"] == "surged"


def test_v1_explain_empty_text_returns_400():
    """Explain endpoint should reject blank text."""
    response = client.post("/api/v1/sentiment/explain", json={"text": ""})
    assert response.status_code == 400
    assert "non-empty" in response.json()["detail"]
