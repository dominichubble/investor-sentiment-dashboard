"""Tests for stock sentiment endpoint emotion payload."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class DummyStorage:
    def aggregate_sentiment(self, **kwargs):
        return {
            "ticker": "AAPL",
            "total_mentions": 8,
            "average_score": 0.41,
            "sentiment_distribution": {
                "positive": 3,
                "negative": 2,
                "neutral": 3,
            },
        }

    def aggregate_emotions(self, **kwargs):
        return {
            "dominant_distribution": {
                "fear": 1,
                "optimism": 3,
                "uncertainty": 1,
                "confidence": 2,
                "skepticism": 0,
                "mixed": 1,
            },
            "dominant_percentages": {
                "fear": 12.5,
                "optimism": 37.5,
                "uncertainty": 12.5,
                "confidence": 25.0,
                "skepticism": 0.0,
                "mixed": 12.5,
            },
            "top_emotion": "optimism",
            "top_emotion_count": 3,
            "timeline": [
                {
                    "date": "2026-03-01",
                    "total_mentions": 4,
                    "dominant_emotion": "optimism",
                    "counts": {
                        "fear": 0,
                        "optimism": 2,
                        "uncertainty": 1,
                        "confidence": 1,
                        "skepticism": 0,
                        "mixed": 0,
                    },
                }
            ],
        }

    def get_stock_sentiment(self, **kwargs):
        return []


def test_stock_sentiment_includes_emotion_analysis(monkeypatch):
    from api.routers import stocks

    monkeypatch.setattr(stocks, "storage", DummyStorage())
    response = client.get("/api/v1/stocks/AAPL/sentiment")

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["emotion_analysis"]["top_emotion"] == "optimism"
    assert body["emotion_analysis"]["dominant_distribution"]["confidence"] == 2
