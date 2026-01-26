"""
Tests for explainability API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestLIMEExplainability:
    """Test LIME explanation endpoints."""

    def test_explain_positive_text(self):
        """Test LIME explanation for positive sentiment."""
        response = client.post(
            "/api/v1/explainability/explain",
            json={
                "text": "Stock prices surged to record highs on strong earnings",
                "num_features": 10,
                "num_samples": 1000,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "text" in data
        assert "prediction" in data
        assert "features" in data
        assert "metadata" in data

        # Check prediction
        prediction = data["prediction"]
        assert "label" in prediction
        assert "score" in prediction
        assert "all_scores" in prediction
        assert prediction["label"] in ["positive", "negative", "neutral"]
        assert 0 <= prediction["score"] <= 1

        # Check features
        features = data["features"]
        assert len(features) > 0
        assert len(features) <= 10

        for feature in features:
            assert "feature" in feature
            assert "weight" in feature
            assert isinstance(feature["feature"], str)
            assert isinstance(feature["weight"], (int, float))

        # Check metadata
        metadata = data["metadata"]
        assert metadata["method"] == "LIME"
        assert metadata["num_features"] == len(features)
        assert metadata["num_samples"] == 1000
        assert "processing_time_ms" in metadata
        assert "timestamp" in metadata

    def test_explain_negative_text(self):
        """Test LIME explanation for negative sentiment."""
        response = client.post(
            "/api/v1/explainability/explain",
            json={
                "text": "Markets crashed following disappointing employment figures",
                "num_features": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()

        features = data["features"]
        assert len(features) <= 5

        # Negative words should have positive weights for negative prediction
        prediction = data["prediction"]
        if prediction["label"] == "negative":
            # At least some features should contribute to negative sentiment
            assert any(f["weight"] != 0 for f in features)

    def test_explain_custom_parameters(self):
        """Test explanation with custom parameters."""
        response = client.post(
            "/api/v1/explainability/explain",
            json={
                "text": "The Federal Reserve raised interest rates",
                "num_features": 15,
                "num_samples": 500,
            },
        )

        assert response.status_code == 200
        data = response.json()

        metadata = data["metadata"]
        assert metadata["num_samples"] == 500
        assert len(data["features"]) <= 15

    def test_explain_empty_text(self):
        """Test explanation with empty text."""
        response = client.post(
            "/api/v1/explainability/explain",
            json={"text": ""},
        )

        # Should return validation error
        assert response.status_code == 422

    def test_explain_very_long_text(self):
        """Test explanation with text exceeding max length."""
        long_text = "word " * 2000  # Exceeds 5000 char limit

        response = client.post(
            "/api/v1/explainability/explain",
            json={"text": long_text},
        )

        # Should return validation error
        assert response.status_code == 422

    def test_explain_invalid_num_features(self):
        """Test explanation with invalid num_features."""
        response = client.post(
            "/api/v1/explainability/explain",
            json={
                "text": "Test text",
                "num_features": 100,  # Exceeds max of 50
            },
        )

        # Should return validation error
        assert response.status_code == 422

    def test_explain_invalid_num_samples(self):
        """Test explanation with invalid num_samples."""
        response = client.post(
            "/api/v1/explainability/explain",
            json={
                "text": "Test text",
                "num_samples": 10000,  # Exceeds max of 5000
            },
        )

        # Should return validation error
        assert response.status_code == 422


class TestBatchExplainability:
    """Test batch explanation endpoints."""

    def test_batch_explain_multiple_texts(self):
        """Test batch LIME explanations."""
        response = client.post(
            "/api/v1/explainability/batch",
            json={
                "texts": [
                    "Markets hit record highs",
                    "Stock prices tumbled on bad news",
                    "Trading remained steady",
                ],
                "num_features": 5,
                "num_samples": 500,
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
            assert "prediction" in result
            assert "features" in result

            # Check each result has valid structure
            prediction = result["prediction"]
            assert "label" in prediction
            assert "score" in prediction

            features = result["features"]
            assert len(features) <= 5
            for feature in features:
                assert "feature" in feature
                assert "weight" in feature

        # Check metadata
        metadata = data["metadata"]
        assert metadata["method"] == "LIME"
        assert metadata["num_features"] == 5
        assert metadata["num_samples"] == 500

    def test_batch_explain_single_text(self):
        """Test batch explanation with single text."""
        response = client.post(
            "/api/v1/explainability/batch",
            json={
                "texts": ["Single text to explain"],
                "num_features": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()

        results = data["results"]
        assert len(results) == 1
        assert results[0]["text"] == "Single text to explain"

    def test_batch_explain_empty_list(self):
        """Test batch explanation with empty list."""
        response = client.post(
            "/api/v1/explainability/batch",
            json={"texts": []},
        )

        # Should return validation error (min_items=1)
        assert response.status_code == 422

    def test_batch_explain_too_many_texts(self):
        """Test batch explanation with too many texts."""
        # Create 21 texts (exceeds limit of 20)
        texts = [f"Text number {i}" for i in range(21)]

        response = client.post(
            "/api/v1/explainability/batch",
            json={"texts": texts},
        )

        # Should return 400 error for exceeding limit
        assert response.status_code == 400
        assert "Maximum 20 texts" in response.json()["detail"]

    def test_batch_explain_performance(self):
        """Test batch explanation processing time."""
        response = client.post(
            "/api/v1/explainability/batch",
            json={
                "texts": ["Text one", "Text two"],
                "num_features": 5,
                "num_samples": 100,  # Lower samples for faster test
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should complete in reasonable time
        metadata = data["metadata"]
        assert "processing_time_ms" in metadata
        assert metadata["processing_time_ms"] > 0


class TestSHAPExplainability:
    """Test SHAP explanation endpoints."""

    def test_shap_explain_positive_text(self):
        """Test SHAP explanation for positive sentiment."""
        response = client.post(
            "/api/v1/explainability/shap",
            json={
                "text": "Stock prices surged to record highs",
                "num_features": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "text" in data
        assert "prediction" in data
        assert "features" in data
        assert "metadata" in data

        # Check metadata is SHAP-specific
        metadata = data["metadata"]
        assert metadata["method"] == "SHAP"
        assert metadata["num_samples"] == 0  # SHAP doesn't use sampling

        # Check features
        features = data["features"]
        assert len(features) > 0
        assert len(features) <= 10

        for feature in features:
            assert "feature" in feature
            assert "weight" in feature

    def test_shap_explain_negative_text(self):
        """Test SHAP explanation for negative sentiment."""
        response = client.post(
            "/api/v1/explainability/shap",
            json={
                "text": "Markets crashed on disappointing news",
                "num_features": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()

        prediction = data["prediction"]
        assert prediction["label"] in ["positive", "negative", "neutral"]
        assert 0 <= prediction["score"] <= 1

        features = data["features"]
        assert len(features) <= 5

    def test_shap_explain_empty_text(self):
        """Test SHAP explanation with empty text."""
        response = client.post(
            "/api/v1/explainability/shap",
            json={"text": ""},
        )

        # Should return validation error
        assert response.status_code == 422

    def test_shap_explain_custom_features(self):
        """Test SHAP explanation with custom feature count."""
        response = client.post(
            "/api/v1/explainability/shap",
            json={
                "text": "Federal Reserve announces policy change",
                "num_features": 20,
            },
        )

        assert response.status_code == 200
        data = response.json()

        features = data["features"]
        assert len(features) <= 20


class TestExampleExplanations:
    """Test pre-computed example explanations."""

    def test_get_all_examples(self):
        """Test getting all example explanations."""
        response = client.get("/api/v1/explainability/examples")

        assert response.status_code == 200
        data = response.json()

        assert "examples" in data
        assert "total" in data

        examples = data["examples"]
        assert len(examples) > 0
        assert data["total"] == len(examples)

        # Check example structure
        for example in examples:
            assert "id" in example
            assert "text" in example
            assert "category" in example
            assert "prediction" in example
            assert "top_features" in example

            # Check prediction
            prediction = example["prediction"]
            assert "label" in prediction
            assert "score" in prediction
            assert "all_scores" in prediction

            # Check features
            features = example["top_features"]
            assert len(features) > 0
            for feature in features:
                assert "feature" in feature
                assert "weight" in feature

    def test_get_examples_by_category(self):
        """Test filtering examples by category."""
        response = client.get("/api/v1/explainability/examples?category=positive")

        assert response.status_code == 200
        data = response.json()

        examples = data["examples"]
        assert len(examples) > 0

        # All examples should be positive
        for example in examples:
            assert example["category"] == "positive"

    def test_get_examples_negative_category(self):
        """Test filtering for negative examples."""
        response = client.get("/api/v1/explainability/examples?category=negative")

        assert response.status_code == 200
        data = response.json()

        examples = data["examples"]
        assert len(examples) > 0

        for example in examples:
            assert example["category"] == "negative"

    def test_get_examples_neutral_category(self):
        """Test filtering for neutral examples."""
        response = client.get("/api/v1/explainability/examples?category=neutral")

        assert response.status_code == 200
        data = response.json()

        examples = data["examples"]
        assert len(examples) > 0

        for example in examples:
            assert example["category"] == "neutral"

    def test_get_examples_with_limit(self):
        """Test limiting number of examples."""
        response = client.get("/api/v1/explainability/examples?limit=2")

        assert response.status_code == 200
        data = response.json()

        examples = data["examples"]
        assert len(examples) <= 2
        assert data["total"] == len(examples)

    def test_get_examples_invalid_limit(self):
        """Test with invalid limit parameter."""
        response = client.get("/api/v1/explainability/examples?limit=100")

        # Should return validation error (max 50)
        assert response.status_code == 422

    def test_get_examples_category_and_limit(self):
        """Test combining category filter and limit."""
        response = client.get(
            "/api/v1/explainability/examples?category=positive&limit=1"
        )

        assert response.status_code == 200
        data = response.json()

        examples = data["examples"]
        assert len(examples) <= 1
        if len(examples) > 0:
            assert examples[0]["category"] == "positive"
