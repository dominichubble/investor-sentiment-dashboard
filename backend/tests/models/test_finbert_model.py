"""
Test suite for FinBERT model wrapper.
"""

import pytest
import torch

from app.models.finbert_model import FinBERTModel, get_model


class TestFinBERTModel:
    """Test cases for FinBERTModel class."""

    @pytest.fixture
    def model(self):
        """Fixture to provide a FinBERT model instance."""
        return FinBERTModel()

    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.device in ["cuda", "cpu"]

    def test_device_detection(self, model):
        """Test GPU/CPU device detection."""
        if torch.cuda.is_available():
            assert model.device == "cuda"
        else:
            assert model.device == "cpu"

    def test_single_prediction(self, model):
        """Test prediction on a single text."""
        text = "The stock market is performing well today"
        result = model.predict(text)

        assert "label" in result
        assert "score" in result
        assert result["label"] in ["positive", "negative", "neutral"]
        assert 0 <= result["score"] <= 1

    def test_positive_sentiment(self, model):
        """Test that clearly positive text is classified as positive."""
        text = "The company exceeded earnings expectations and stock soared"
        result = model.predict(text)

        assert result["label"] == "positive"
        assert result["score"] > 0.5

    def test_negative_sentiment(self, model):
        """Test that clearly negative text is classified as negative."""
        text = "The company filed for bankruptcy and lost all value"
        result = model.predict(text)

        assert result["label"] == "negative"
        assert result["score"] > 0.5

    def test_batch_prediction(self, model):
        """Test prediction on multiple texts."""
        texts = [
            "Stock prices are rising",
            "The market crashed today",
            "Trading volume was normal",
        ]
        results = model.predict(texts)

        assert len(results) == 3
        assert all("label" in r and "score" in r for r in results)

    def test_get_device_info(self, model):
        """Test device information retrieval."""
        info = model.get_device_info()

        assert "device" in info
        assert "device_name" in info
        assert "cuda_available" in info
        assert "model_loaded" in info
        assert info["model_loaded"] is True

    def test_singleton_pattern(self):
        """Test that get_model() returns the same instance."""
        model1 = get_model()
        model2 = get_model()

        assert model1 is model2
