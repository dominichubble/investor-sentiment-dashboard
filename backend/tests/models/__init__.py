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

    def test_single_prediction_with_all_scores(self, model):
        """Test prediction with all scores returned."""
        text = "The company reported strong earnings"
        result = model.predict(text, return_all_scores=True)

        assert "label" in result
        assert "score" in result
        assert "scores" in result
        assert len(result["scores"]) == 3
        assert all(
            label in result["scores"] for label in ["positive", "negative", "neutral"]
        )

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

    def test_neutral_sentiment(self, model):
        """Test neutral text classification."""
        text = "The company announced its quarterly report"
        result = model.predict(text)

        # Neutral predictions might vary, so just check it's valid
        assert result["label"] in ["positive", "negative", "neutral"]

    def test_batch_processing(self, model):
        """Test efficient batch processing."""
        texts = [f"This is test sentence number {i}" for i in range(10)]
        results = model.predict_batch(texts, batch_size=5)

        assert len(results) == 10
        assert all("label" in r for r in results)

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

    def test_empty_text_handling(self, model):
        """Test handling of empty text."""
        result = model.predict("")

        # Should still return a valid result
        assert "label" in result
        assert "score" in result

    def test_long_text_truncation(self, model):
        """Test that long texts are truncated properly."""
        # Create a very long text (over 512 tokens)
        long_text = " ".join(["stock market trading"] * 200)
        result = model.predict(long_text)

        # Should handle without errors
        assert "label" in result
        assert "score" in result


class TestModelIntegration:
    """Integration tests for the model."""

    def test_financial_domain_texts(self):
        """Test on various financial domain texts."""
        model = get_model()

        test_cases = [
            ("Q3 earnings beat expectations", "positive"),
            ("Company faces regulatory scrutiny", "negative"),
            ("Stock split announced", None),  # Could be any
            ("Dividend increased by 10%", "positive"),
            ("CEO resigned amid scandal", "negative"),
        ]

        for text, expected_label in test_cases:
            result = model.predict(text)
            if expected_label:
                assert result["label"] == expected_label, (
                    f"Expected label '{expected_label}' for: {text}, got '{result['label']}'"
                )
            else:
                assert result["label"] in ["positive", "negative", "neutral"]
