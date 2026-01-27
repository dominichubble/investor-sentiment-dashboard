"""
Tests for data retrieval API endpoints.
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestPredictionsEndpoint:
    """Test predictions retrieval endpoint."""

    def test_get_predictions_default(self):
        """Test getting predictions with default parameters."""
        response = client.get("/api/v1/data/predictions")

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "predictions" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "has_more" in data

        # Check defaults
        assert data["page"] == 1
        assert data["page_size"] == 20

        # Check predictions structure
        if data["total"] > 0:
            prediction = data["predictions"][0]
            assert "id" in prediction
            assert "text" in prediction
            assert "sentiment" in prediction
            assert "timestamp" in prediction

            # Check sentiment structure
            sentiment = prediction["sentiment"]
            assert "label" in sentiment
            assert "score" in sentiment

    def test_get_predictions_with_pagination(self):
        """Test pagination parameters."""
        response = client.get("/api/v1/data/predictions?page=1&page_size=5")

        assert response.status_code == 200
        data = response.json()

        assert data["page"] == 1
        assert data["page_size"] == 5
        assert len(data["predictions"]) <= 5

    def test_get_predictions_invalid_page(self):
        """Test with invalid page number."""
        response = client.get("/api/v1/data/predictions?page=0")

        # Should return validation error
        assert response.status_code == 422

    def test_get_predictions_invalid_page_size(self):
        """Test with invalid page size."""
        response = client.get("/api/v1/data/predictions?page_size=1000")

        # Should return validation error (max 100)
        assert response.status_code == 422

    def test_get_predictions_filter_by_sentiment(self):
        """Test filtering by sentiment."""
        response = client.get("/api/v1/data/predictions?sentiment=positive")

        assert response.status_code == 200
        data = response.json()

        # All predictions should be positive if any returned
        for prediction in data["predictions"]:
            assert prediction["sentiment"]["label"] == "positive"

    def test_get_predictions_filter_by_source(self):
        """Test filtering by source."""
        response = client.get("/api/v1/data/predictions?source=reddit")

        assert response.status_code == 200
        data = response.json()

        # All predictions should be from reddit if any returned
        for prediction in data["predictions"]:
            if prediction.get("source"):
                assert prediction["source"] == "reddit"

    def test_get_predictions_filter_by_date_range(self):
        """Test filtering by date range."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        response = client.get(
            f"/api/v1/data/predictions?start_date={yesterday}&end_date={today}"
        )

        assert response.status_code == 200
        data = response.json()

        # Response should be valid
        assert "predictions" in data
        assert "total" in data

    def test_get_predictions_multiple_filters(self):
        """Test combining multiple filters."""
        response = client.get(
            "/api/v1/data/predictions?sentiment=positive&page_size=10"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["page_size"] == 10
        assert len(data["predictions"]) <= 10


class TestSinglePredictionEndpoint:
    """Test single prediction retrieval endpoint."""

    def test_get_prediction_not_found(self):
        """Test getting non-existent prediction."""
        response = client.get("/api/v1/data/predictions/nonexistent_id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_prediction_invalid_id(self):
        """Test with invalid prediction ID format."""
        # Note: trailing slash matches the list endpoint, so returns 200
        # Test with an actual invalid ID instead
        response = client.get("/api/v1/data/predictions/invalid-id-format")

        # Should return 404 (prediction not found)
        assert response.status_code == 404


class TestStockSentimentEndpoint:
    """Test stock-specific sentiment endpoint."""

    def test_get_stock_sentiment_not_found(self):
        """Test getting sentiment for non-existent stock."""
        response = client.get("/api/v1/data/stocks/INVALID/sentiment")

        assert response.status_code == 404
        assert "No sentiment data found" in response.json()["detail"]

    def test_get_stock_sentiment_with_limit(self):
        """Test limit parameter."""
        # Use a common stock that might have data
        response = client.get("/api/v1/data/stocks/AAPL/sentiment?limit=10")

        # Could be 200 with data or 404 if no data yet
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()

            assert "ticker" in data
            assert "sentiments" in data
            assert "total" in data
            assert "summary" in data

            assert data["ticker"] == "AAPL"
            assert len(data["sentiments"]) <= 10

            # Check summary structure
            summary = data["summary"]
            assert "total_mentions" in summary
            assert "positive" in summary
            assert "negative" in summary
            assert "neutral" in summary
            assert "positive_percentage" in summary
            assert "negative_percentage" in summary
            assert "neutral_percentage" in summary

    def test_get_stock_sentiment_with_date_filter(self):
        """Test date filtering for stock sentiment."""
        today = date.today()
        week_ago = today - timedelta(days=7)

        response = client.get(
            f"/api/v1/data/stocks/TSLA/sentiment?start_date={week_ago}"
        )

        # Valid request format
        assert response.status_code in [200, 404]

    def test_get_stock_sentiment_invalid_limit(self):
        """Test with invalid limit parameter."""
        response = client.get("/api/v1/data/stocks/AAPL/sentiment?limit=10000")

        # Should return validation error (max 500)
        assert response.status_code == 422

    def test_get_stock_sentiment_case_insensitive(self):
        """Test that ticker is case-insensitive."""
        # Both should work the same way
        response1 = client.get("/api/v1/data/stocks/aapl/sentiment")
        response2 = client.get("/api/v1/data/stocks/AAPL/sentiment")

        # Both should return same status code
        assert response1.status_code == response2.status_code

        if response1.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # Ticker should be uppercase in response
            assert data1["ticker"] == "AAPL"
            assert data2["ticker"] == "AAPL"


class TestStatisticsEndpoint:
    """Test statistics aggregation endpoint."""

    def test_get_statistics(self):
        """Test getting aggregate statistics."""
        response = client.get("/api/v1/data/statistics")

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "total_predictions" in data
        assert "total_stocks_analyzed" in data
        assert "sentiment_distribution" in data
        assert "top_stocks" in data
        assert "recent_activity" in data
        assert "date_range" in data

        # Check types
        assert isinstance(data["total_predictions"], int)
        assert isinstance(data["total_stocks_analyzed"], int)

        # Check sentiment distribution structure
        sentiment_dist = data["sentiment_distribution"]
        assert "positive" in sentiment_dist
        assert "negative" in sentiment_dist
        assert "neutral" in sentiment_dist
        assert "positive_percentage" in sentiment_dist
        assert "negative_percentage" in sentiment_dist
        assert "neutral_percentage" in sentiment_dist

        # Percentages should sum to ~100 if there's data
        if data["total_predictions"] > 0:
            total_pct = (
                sentiment_dist["positive_percentage"]
                + sentiment_dist["negative_percentage"]
                + sentiment_dist["neutral_percentage"]
            )
            assert 99.5 <= total_pct <= 100.5  # Allow for rounding

        # Check top stocks structure
        assert isinstance(data["top_stocks"], list)
        if len(data["top_stocks"]) > 0:
            top_stock = data["top_stocks"][0]
            assert "ticker" in top_stock
            assert "company_name" in top_stock
            assert "count" in top_stock
            assert "positive" in top_stock
            assert "negative" in top_stock
            assert "neutral" in top_stock

        # Check recent activity structure
        recent = data["recent_activity"]
        assert "last_24h" in recent
        assert "last_7d" in recent
        assert "last_30d" in recent
        assert isinstance(recent["last_24h"], int)
        assert isinstance(recent["last_7d"], int)
        assert isinstance(recent["last_30d"], int)

        # Check date range structure
        date_range = data["date_range"]
        assert "earliest" in date_range
        assert "latest" in date_range

    def test_statistics_consistency(self):
        """Test that statistics are internally consistent."""
        response = client.get("/api/v1/data/statistics")

        assert response.status_code == 200
        data = response.json()

        # Recent activity counts should be cumulative
        recent = data["recent_activity"]
        assert recent["last_24h"] <= recent["last_7d"]
        assert recent["last_7d"] <= recent["last_30d"]

        # Top stocks should be sorted by count
        top_stocks = data["top_stocks"]
        if len(top_stocks) > 1:
            for i in range(len(top_stocks) - 1):
                assert top_stocks[i]["count"] >= top_stocks[i + 1]["count"]

    def test_statistics_top_stocks_limit(self):
        """Test that top stocks are limited to 10."""
        response = client.get("/api/v1/data/statistics")

        assert response.status_code == 200
        data = response.json()

        # Should return max 10 top stocks
        assert len(data["top_stocks"]) <= 10


class TestDataEndpointsIntegration:
    """Integration tests for data endpoints."""

    def test_predictions_to_statistics_consistency(self):
        """Test that predictions count matches statistics."""
        # Get all predictions
        predictions_response = client.get("/api/v1/data/predictions?page_size=100")
        stats_response = client.get("/api/v1/data/statistics")

        assert predictions_response.status_code == 200
        assert stats_response.status_code == 200

        # If both succeed, counts should match
        predictions_data = predictions_response.json()
        stats_data = stats_response.json()

        if predictions_data["total"] > 0:
            # Total predictions in stats should match total in predictions endpoint
            assert stats_data["total_predictions"] == predictions_data["total"]

    def test_filter_combinations(self):
        """Test various filter combinations work together."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        # Test multiple valid filter combinations
        filters = [
            "?sentiment=positive&page_size=5",
            f"?start_date={yesterday}",
            "?source=reddit&sentiment=negative",
            "?page=1&page_size=10&sentiment=neutral",
        ]

        for filter_str in filters:
            response = client.get(f"/api/v1/data/predictions{filter_str}")
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "total" in data
