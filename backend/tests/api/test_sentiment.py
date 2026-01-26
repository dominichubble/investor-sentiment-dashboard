"""
Tests for sentiment API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestSentimentEndpoints:
    """Test sentiment analysis endpoints."""

    def test_analyze_sentiment_positive(self):
        """Test sentiment analysis with positive text."""
        response = client.post(
            "/api/v1/sentiment/analyze",
            json={
                "text": "Stock prices surged to record highs on strong earnings",
                "options": {"include_scores": True},
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "text" in data
        assert "sentiment" in data
        assert "metadata" in data

        # Check sentiment
        sentiment = data["sentiment"]
        assert "label" in sentiment
        assert "score" in sentiment
        assert "scores" in sentiment

        # Should be positive
        assert sentiment["label"] == "positive"
        assert sentiment["score"] > 0.5

        # Check scores
        scores = sentiment["scores"]
        assert "positive" in scores
        assert "negative" in scores
        assert "neutral" in scores
        assert abs(scores["positive"] + scores["negative"] + scores["neutral"] - 1.0) < 0.01

        # Check metadata
        metadata = data["metadata"]
        assert metadata["model"] == "finbert"
        assert "processing_time_ms" in metadata
        assert "timestamp" in metadata

    def test_analyze_sentiment_negative(self):
        """Test sentiment analysis with negative text."""
        response = client.post(
            "/api/v1/sentiment/analyze",
            json={
                "text": "Markets crashed following disappointing employment figures",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should be negative
        sentiment = data["sentiment"]
        assert sentiment["label"] == "negative"
        assert sentiment["score"] > 0.5

    def test_analyze_sentiment_neutral(self):
        """Test sentiment analysis with neutral text."""
        response = client.post(
            "/api/v1/sentiment/analyze",
            json={
                "text": "The company held its annual meeting today",
            },
        )

        assert response.status_code == 200
        data = response.json()

        sentiment = data["sentiment"]
        assert sentiment["label"] in ["positive", "negative", "neutral"]
        assert 0 <= sentiment["score"] <= 1

    def test_analyze_sentiment_empty_text(self):
        """Test sentiment analysis with empty text."""
        response = client.post(
            "/api/v1/sentiment/analyze",
            json={"text": ""},
        )

        # Should return validation error
        assert response.status_code == 422

    def test_analyze_sentiment_without_scores(self):
        """Test sentiment analysis without detailed scores."""
        response = client.post(
            "/api/v1/sentiment/analyze",
            json={
                "text": "Stock prices increased",
                "options": {"include_scores": False},
            },
        )

        assert response.status_code == 200
        data = response.json()

        sentiment = data["sentiment"]
        # Scores should still be included by default in current implementation
        assert "label" in sentiment
        assert "score" in sentiment

    def test_batch_sentiment_analysis(self):
        """Test batch sentiment analysis."""
        response = client.post(
            "/api/v1/sentiment/batch",
            json={
                "texts": [
                    "Markets hit record highs",
                    "Stock prices tumbled",
                    "Trading remained steady",
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "results" in data
        assert "metadata" in data

        # Check results
        results = data["results"]
        assert len(results) == 3

        for result in results:
            assert "text" in result
            assert "sentiment" in result
            sentiment = result["sentiment"]
            assert "label" in sentiment
            assert "score" in sentiment

        # Check metadata
        metadata = data["metadata"]
        assert metadata["model"] == "finbert"
        assert "processing_time_ms" in metadata

    def test_batch_sentiment_single_text(self):
        """Test batch analysis with single text."""
        response = client.post(
            "/api/v1/sentiment/batch",
            json={"texts": ["Single text to analyze"]},
        )

        assert response.status_code == 200
        data = response.json()

        results = data["results"]
        assert len(results) == 1
        assert results[0]["text"] == "Single text to analyze"

    def test_batch_sentiment_empty_list(self):
        """Test batch analysis with empty list."""
        response = client.post(
            "/api/v1/sentiment/batch",
            json={"texts": []},
        )

        # Should return validation error (min_items=1)
        assert response.status_code == 422

    def test_batch_sentiment_too_many_texts(self):
        """Test batch analysis with too many texts."""
        # Create 101 texts (exceeds limit of 100)
        texts = [f"Text number {i}" for i in range(101)]

        response = client.post(
            "/api/v1/sentiment/batch",
            json={"texts": texts},
        )

        # Should return 422 error for validation (Pydantic validation error)
        assert response.status_code == 422

    def test_batch_sentiment_with_options(self):
        """Test batch analysis with options."""
        response = client.post(
            "/api/v1/sentiment/batch",
            json={
                "texts": ["Positive news", "Negative news"],
                "options": {"include_scores": True},
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check that scores are included
        for result in data["results"]:
            sentiment = result["sentiment"]
            assert "scores" in sentiment


class TestHealthAndInfo:
    """Test health and info endpoints."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "timestamp" in data
        assert "services" in data

        services = data["services"]
        assert "finbert_model" in services
        assert "storage" in services
        assert "api" in services

    def test_api_info(self):
        """Test API info endpoint."""
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()

        assert data["version"] == "1.0.0"
        assert "name" in data
        assert "capabilities" in data
        assert "models" in data
        assert "rate_limits" in data
        assert "documentation" in data

        # Check capabilities
        capabilities = data["capabilities"]
        assert "sentiment_analysis" in capabilities
        assert "stock_entity_extraction" in capabilities

        # Check models info
        models = data["models"]
        assert "sentiment" in models
        assert "ner" in models
        assert models["sentiment"]["name"] == "FinBERT"

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "Investor Sentiment Dashboard API"
        assert data["version"] == "1.0.0"
        assert "documentation" in data
        assert "endpoints" in data

        endpoints = data["endpoints"]
        assert "/api/v1/sentiment" in endpoints.values()
        assert "/api/v1/stocks" in endpoints.values()
