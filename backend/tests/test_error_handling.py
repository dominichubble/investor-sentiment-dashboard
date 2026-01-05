"""
Tests for Error Handling and Logging

Tests the comprehensive error handling, logging, and failed items tracking
functionality across the sentiment analysis pipeline.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.logging_config import (
    FailedItemsTracker,
    get_logger,
    log_exception,
    setup_logging,
)
from app.models.sentiment_inference import (
    ModelInferenceError,
    SentimentAnalysisError,
    TokenizationError,
    analyze_batch,
    analyze_sentiment,
)
from app.preprocessing.finbert_tokenizer import (
    MAX_REASONABLE_LENGTH,
)
from app.preprocessing.finbert_tokenizer import TokenizationError as TokenizerError
from app.preprocessing.finbert_tokenizer import (
    tokenize_for_inference,
)
from app.storage.prediction_storage import (
    save_predictions_batch,
    validate_prediction,
)


class TestLoggingConfiguration:
    """Tests for logging configuration module."""

    def test_setup_logging_creates_log_directory(self, tmp_path):
        """Test that setup_logging creates the log directory."""
        log_dir = tmp_path / "test_logs"
        setup_logging(log_dir=log_dir)

        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a valid logger."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_log_exception_logs_with_traceback(self, caplog):
        """Test that log_exception logs exceptions with traceback."""
        logger = get_logger("test_exception")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            with caplog.at_level(logging.ERROR):
                log_exception(logger, e, "Test context")

        assert "Test context" in caplog.text
        assert "Test error" in caplog.text


class TestFailedItemsTracker:
    """Tests for FailedItemsTracker class."""

    def test_add_failure_tracks_item(self):
        """Test that add_failure correctly tracks a failed item."""
        tracker = FailedItemsTracker()

        tracker.add_failure(
            item="test text",
            error_type="ValueError",
            error_message="Test error message",
        )

        assert tracker.count() == 1
        assert tracker.failed_items[0]["item"] == "test text"
        assert tracker.failed_items[0]["error_type"] == "ValueError"
        assert tracker.failed_items[0]["error_message"] == "Test error message"

    def test_add_failure_with_additional_info(self):
        """Test that add_failure includes additional_info."""
        tracker = FailedItemsTracker()

        tracker.add_failure(
            item="test text",
            error_type="TokenizationError",
            error_message="Text too long",
            additional_info={"length": 5000},
        )

        assert tracker.failed_items[0]["additional_info"]["length"] == 5000

    def test_add_failure_truncates_long_text(self):
        """Test that add_failure truncates very long texts."""
        tracker = FailedItemsTracker()
        long_text = "a" * 1000

        tracker.add_failure(item=long_text, error_type="Error", error_message="Test")

        assert len(tracker.failed_items[0]["item"]) <= 500

    def test_save_creates_json_file(self, tmp_path):
        """Test that save creates a properly formatted JSON file."""
        tracker = FailedItemsTracker()
        tracker.add_failure("text1", "Error1", "msg1")
        tracker.add_failure("text2", "Error2", "msg2")

        output_file = tmp_path / "failed_items.json"
        count = tracker.save(output_file)

        assert count == 2
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["item"] == "text1"
        assert data[1]["item"] == "text2"

    def test_save_empty_tracker_no_file(self, tmp_path):
        """Test that save doesn't create file for empty tracker."""
        tracker = FailedItemsTracker()

        output_file = tmp_path / "failed_items.json"
        count = tracker.save(output_file)

        assert count == 0
        assert not output_file.exists()

    def test_clear_removes_all_items(self):
        """Test that clear removes all failed items."""
        tracker = FailedItemsTracker()
        tracker.add_failure("text1", "Error1", "msg1")
        tracker.add_failure("text2", "Error2", "msg2")

        assert tracker.count() == 2

        tracker.clear()

        assert tracker.count() == 0


class TestSentimentInferenceErrorHandling:
    """Tests for sentiment inference error handling."""

    def test_analyze_sentiment_with_empty_text(self):
        """Test that analyze_sentiment raises ValueError for empty text."""
        with pytest.raises(ValueError, match="cannot be blank"):
            analyze_sentiment("")

    def test_analyze_sentiment_with_whitespace_text(self):
        """Test that analyze_sentiment raises ValueError for whitespace."""
        with pytest.raises(ValueError, match="cannot be blank"):
            analyze_sentiment("   \n\t  ")

    def test_analyze_sentiment_with_non_string(self):
        """Test that analyze_sentiment raises ValueError for non-string."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            analyze_sentiment(123)

    def test_analyze_sentiment_with_very_long_text(self, caplog):
        """Test that analyze_sentiment warns about very long texts."""
        long_text = "a" * 15000

        with patch("app.models.sentiment_inference.get_model") as mock_model:
            mock_model.return_value.predict.return_value = {
                "label": "neutral",
                "score": 0.5,
            }

            with caplog.at_level(logging.WARNING):
                analyze_sentiment(long_text)

            assert "very long" in caplog.text.lower()

    def test_analyze_batch_with_empty_list(self):
        """Test that analyze_batch raises ValueError for empty list."""
        with pytest.raises(ValueError, match="non-empty list"):
            analyze_batch([])

    def test_analyze_batch_with_non_list(self):
        """Test that analyze_batch raises ValueError for non-list."""
        with pytest.raises(ValueError, match="must be a non-empty list"):
            analyze_batch("not a list")

    def test_analyze_batch_tracks_empty_texts(self):
        """Test that analyze_batch tracks empty texts in failures."""
        texts = ["good text", "", "another text", "   ", "final text"]

        with patch("app.models.sentiment_inference.get_model") as mock_model:
            mock_model.return_value.predict_batch.return_value = [
                {"label": "positive", "score": 0.9},
                {"label": "neutral", "score": 0.6},
                {"label": "negative", "score": 0.8},
            ]

            results, failures = analyze_batch(texts, skip_errors=True)

        assert failures is not None
        assert failures.count() == 2  # Two empty texts
        assert results[1] is None  # Empty text at index 1
        assert results[3] is None  # Whitespace text at index 3

    def test_analyze_batch_skip_errors_returns_none_for_failures(self):
        """Test that analyze_batch returns None for failed items when skip_errors=True."""
        texts = ["text1", "text2", "text3"]

        with patch("app.models.sentiment_inference.get_model") as mock_model:
            # Simulate batch failure, then individual success/failure
            mock_model.return_value.predict_batch.side_effect = RuntimeError(
                "Batch failed"
            )
            mock_model.return_value.predict.side_effect = [
                {"label": "positive", "score": 0.9},
                RuntimeError("Individual failure"),
                {"label": "negative", "score": 0.8},
            ]

            results, failures = analyze_batch(texts, skip_errors=True)

        assert results[0] is not None
        assert results[1] is None  # Failed
        assert results[2] is not None
        assert failures.count() >= 1

    def test_analyze_batch_raises_without_skip_errors(self):
        """Test that analyze_batch raises exception when skip_errors=False."""
        texts = ["text1", "text2"]

        with patch("app.models.sentiment_inference.get_model") as mock_model:
            mock_model.return_value.predict_batch.side_effect = RuntimeError(
                "Batch failed"
            )

            with pytest.raises(RuntimeError, match="Batch analysis failed"):
                analyze_batch(texts, skip_errors=False)


class TestTokenizationErrorHandling:
    """Tests for tokenization error handling."""

    def test_tokenize_empty_text_raises_error(self):
        """Test that tokenizing empty text raises ValueError."""
        with pytest.raises(ValueError, match="empty or whitespace"):
            tokenize_for_inference("")

    def test_tokenize_whitespace_text_raises_error(self):
        """Test that tokenizing whitespace text raises ValueError."""
        with pytest.raises(ValueError, match="empty or whitespace"):
            tokenize_for_inference("   \n\t  ")

    def test_tokenize_non_string_raises_error(self):
        """Test that tokenizing non-string raises ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            tokenize_for_inference(123)

    def test_tokenize_extremely_long_text_raises_error(self):
        """Test that extremely long text raises ValueError."""
        extremely_long_text = "a" * (MAX_REASONABLE_LENGTH + 1000)

        with pytest.raises(ValueError, match="too long"):
            tokenize_for_inference(extremely_long_text)

    def test_tokenize_long_text_warns(self, caplog):
        """Test that long texts trigger a warning."""
        long_text = "a" * 3000

        with patch(
            "app.preprocessing.finbert_tokenizer.get_tokenizer"
        ) as mock_tokenizer:
            mock_tok = MagicMock()
            mock_tok.return_value = {
                "input_ids": MagicMock(shape=(1, 512)),
                "attention_mask": MagicMock(shape=(1, 512)),
            }
            mock_tokenizer.return_value = mock_tok

            with caplog.at_level(logging.WARNING):
                tokenize_for_inference(long_text)

            assert "long" in caplog.text.lower()


class TestPredictionStorageErrorHandling:
    """Tests for prediction storage error handling."""

    def test_validate_prediction_with_missing_fields(self):
        """Test that validate_prediction raises error for missing fields."""
        incomplete_pred = {
            "text": "test",
            "source": "test",
            # Missing label, confidence, timestamp
        }

        with pytest.raises(ValueError, match="Missing required fields"):
            validate_prediction(incomplete_pred)

    def test_validate_prediction_with_invalid_label(self):
        """Test that validate_prediction raises error for invalid label."""
        pred = {
            "text": "test",
            "source": "test",
            "timestamp": "2025-01-01T00:00:00",
            "label": "invalid_label",
            "confidence": 0.9,
        }

        with pytest.raises(ValueError, match="label must be"):
            validate_prediction(pred)

    def test_validate_prediction_with_invalid_confidence(self):
        """Test that validate_prediction raises error for out-of-range confidence."""
        pred = {
            "text": "test",
            "source": "test",
            "timestamp": "2025-01-01T00:00:00",
            "label": "positive",
            "confidence": 1.5,  # Out of range
        }

        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            validate_prediction(pred)

    def test_save_predictions_batch_invalid_format(self, tmp_path):
        """Test that save_predictions_batch raises error for unsupported format."""
        predictions = [
            {
                "text": "test",
                "source": "test",
                "timestamp": "2025-01-01T00:00:00",
                "label": "positive",
                "confidence": 0.9,
            }
        ]

        output_file = tmp_path / "test.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            save_predictions_batch(predictions, output_file, format="txt")

    def test_save_predictions_batch_empty_list(self, tmp_path, caplog):
        """Test that save_predictions_batch handles empty list."""
        output_file = tmp_path / "test.csv"

        with caplog.at_level(logging.WARNING):
            count = save_predictions_batch([], output_file)

        assert count == 0
        assert "No predictions to save" in caplog.text

    def test_save_predictions_batch_creates_directory(self, tmp_path):
        """Test that save_predictions_batch creates missing directories."""
        predictions = [
            {
                "text": "test",
                "source": "test",
                "timestamp": "2025-01-01T00:00:00",
                "label": "positive",
                "confidence": 0.9,
            }
        ]

        output_file = tmp_path / "subdir" / "nested" / "test.csv"

        save_predictions_batch(predictions, output_file)

        assert output_file.exists()
        assert output_file.parent.exists()


class TestIntegrationErrorHandling:
    """Integration tests for end-to-end error handling."""

    def test_full_pipeline_with_mixed_valid_invalid_texts(self, tmp_path):
        """Test full pipeline with mix of valid and invalid texts."""
        texts = [
            "Good market performance today",
            "",  # Empty
            "Stock prices dropping",
            "   ",  # Whitespace
            "Neutral outlook for tomorrow",
        ]

        with patch("app.models.sentiment_inference.get_model") as mock_model:
            mock_model.return_value.predict_batch.return_value = [
                {"label": "positive", "score": 0.9},
                {"label": "negative", "score": 0.8},
                {"label": "neutral", "score": 0.6},
            ]

            results, failures = analyze_batch(texts, skip_errors=True)

        # Check results
        assert results[0] is not None  # Valid
        assert results[1] is None  # Empty
        assert results[2] is not None  # Valid
        assert results[3] is None  # Whitespace
        assert results[4] is not None  # Valid

        # Check failures tracked
        assert failures.count() == 2

        # Save failures
        failed_file = tmp_path / "failed_items.json"
        count = failures.save(failed_file)

        assert count == 2
        assert failed_file.exists()

    def test_full_pipeline_saves_only_successful_predictions(self, tmp_path):
        """Test that only successful predictions are saved."""
        texts = ["text1", "text2", "text3"]

        with patch("app.models.sentiment_inference.get_model") as mock_model:
            # Simulate one failure
            mock_model.return_value.predict_batch.side_effect = RuntimeError()
            mock_model.return_value.predict.side_effect = [
                {"label": "positive", "score": 0.9},
                RuntimeError("Failed"),
                {"label": "negative", "score": 0.8},
            ]

            results, failures = analyze_batch(texts, skip_errors=True)

        # Prepare predictions
        predictions = []
        for text, result in zip(texts, results):
            if result is not None:
                predictions.append(
                    {
                        "text": text,
                        "source": "test",
                        "timestamp": "2025-01-01T00:00:00",
                        "label": result["label"],
                        "confidence": result["score"],
                    }
                )

        # Save predictions
        output_file = tmp_path / "predictions.csv"
        count = save_predictions_batch(predictions, output_file)

        assert count == 2  # Only 2 successful
