"""
Tests for Sentiment Prediction Storage Module

Tests saving, loading, and validation of sentiment predictions
in both CSV and JSON formats.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from app.storage import (
    PredictionRecord,
    get_storage_stats,
    load_predictions,
    save_prediction,
    save_predictions_batch,
    validate_prediction,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_prediction():
    """Sample valid prediction dictionary."""
    return {
        "text": "Stock prices surged today",
        "source": "reddit",
        "timestamp": "2026-01-04T10:30:00",
        "label": "positive",
        "confidence": 0.95,
    }


@pytest.fixture
def sample_predictions():
    """Sample list of valid predictions."""
    return [
        {
            "text": "Market rally continues",
            "source": "twitter",
            "timestamp": "2026-01-04T09:00:00",
            "label": "positive",
            "confidence": 0.92,
        },
        {
            "text": "Economic downturn expected",
            "source": "news",
            "timestamp": "2026-01-04T09:15:00",
            "label": "negative",
            "confidence": 0.88,
        },
        {
            "text": "Federal Reserve maintains rates",
            "source": "news",
            "timestamp": "2026-01-04T09:30:00",
            "label": "neutral",
            "confidence": 0.85,
        },
    ]


class TestPredictionRecord:
    """Test PredictionRecord class."""

    def test_create_record_with_all_fields(self):
        """Test creating a record with all fields."""
        record = PredictionRecord(
            text="Market up",
            source="reddit",
            label="positive",
            confidence=0.9,
            timestamp="2026-01-04T10:00:00",
        )

        assert record.text == "Market up"
        assert record.source == "reddit"
        assert record.label == "positive"
        assert record.confidence == 0.9
        assert record.timestamp == "2026-01-04T10:00:00"

    def test_create_record_auto_timestamp(self):
        """Test that timestamp is auto-generated if not provided."""
        record = PredictionRecord(
            text="Market up", source="reddit", label="positive", confidence=0.9
        )

        # Should have a timestamp
        assert record.timestamp is not None
        # Should be valid ISO format
        datetime.fromisoformat(record.timestamp)

    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = PredictionRecord(
            text="Market up",
            source="reddit",
            label="positive",
            confidence=0.9,
            timestamp="2026-01-04T10:00:00",
        )

        result = record.to_dict()

        assert result["text"] == "Market up"
        assert result["source"] == "reddit"
        assert result["label"] == "positive"
        assert result["confidence"] == 0.9
        assert result["timestamp"] == "2026-01-04T10:00:00"

    def test_record_validates_empty_text(self):
        """Test that empty text raises error."""
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            PredictionRecord(text="", source="reddit", label="positive", confidence=0.9)

    def test_record_validates_empty_source(self):
        """Test that empty source raises error."""
        with pytest.raises(ValueError, match="source must be a non-empty string"):
            PredictionRecord(
                text="Market up", source="", label="positive", confidence=0.9
            )

    def test_record_validates_invalid_label(self):
        """Test that invalid label raises error."""
        with pytest.raises(ValueError, match="label must be"):
            PredictionRecord(
                text="Market up", source="reddit", label="bullish", confidence=0.9
            )

    def test_record_validates_confidence_range(self):
        """Test that confidence must be 0-1."""
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            PredictionRecord(
                text="Market up", source="reddit", label="positive", confidence=1.5
            )

    def test_record_validates_negative_confidence(self):
        """Test that negative confidence raises error."""
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            PredictionRecord(
                text="Market up", source="reddit", label="positive", confidence=-0.1
            )


class TestValidatePrediction:
    """Test prediction validation function."""

    def test_validate_complete_prediction(self, sample_prediction):
        """Test validation of complete valid prediction."""
        assert validate_prediction(sample_prediction) is True

    def test_validate_missing_text(self, sample_prediction):
        """Test that missing text field raises error."""
        del sample_prediction["text"]
        with pytest.raises(ValueError, match="Missing required fields: text"):
            validate_prediction(sample_prediction)

    def test_validate_missing_multiple_fields(self, sample_prediction):
        """Test that multiple missing fields are reported."""
        del sample_prediction["text"]
        del sample_prediction["label"]
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_prediction(sample_prediction)

    def test_validate_empty_text(self, sample_prediction):
        """Test that empty text raises error."""
        sample_prediction["text"] = ""
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            validate_prediction(sample_prediction)

    def test_validate_invalid_label(self, sample_prediction):
        """Test that invalid label raises error."""
        sample_prediction["label"] = "bullish"
        with pytest.raises(ValueError, match="label must be"):
            validate_prediction(sample_prediction)

    def test_validate_confidence_out_of_range(self, sample_prediction):
        """Test that confidence > 1 raises error."""
        sample_prediction["confidence"] = 1.5
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            validate_prediction(sample_prediction)

    def test_validate_invalid_timestamp(self, sample_prediction):
        """Test that invalid timestamp format raises error."""
        sample_prediction["timestamp"] = "not-a-timestamp"
        with pytest.raises(ValueError, match="timestamp must be valid ISO format"):
            validate_prediction(sample_prediction)


class TestSavePrediction:
    """Test saving single predictions."""

    def test_save_prediction_csv(self, temp_dir, sample_prediction):
        """Test saving a single prediction to CSV."""
        output_file = temp_dir / "predictions.csv"

        record = save_prediction(
            text=sample_prediction["text"],
            source=sample_prediction["source"],
            label=sample_prediction["label"],
            confidence=sample_prediction["confidence"],
            timestamp=sample_prediction["timestamp"],
            output_file=output_file,
            format="csv",
        )

        # Check that file was created
        assert output_file.exists()

        # Check that record was returned
        assert isinstance(record, PredictionRecord)
        assert record.text == sample_prediction["text"]

        # Check file contents
        loaded = load_predictions(output_file)
        assert len(loaded) == 1
        assert loaded[0]["text"] == sample_prediction["text"]

    def test_save_prediction_json(self, temp_dir, sample_prediction):
        """Test saving a single prediction to JSON."""
        output_file = temp_dir / "predictions.json"

        save_prediction(
            text=sample_prediction["text"],
            source=sample_prediction["source"],
            label=sample_prediction["label"],
            confidence=sample_prediction["confidence"],
            timestamp=sample_prediction["timestamp"],
            output_file=output_file,
            format="json",
        )

        assert output_file.exists()

        loaded = load_predictions(output_file)
        assert len(loaded) == 1
        assert loaded[0]["label"] == sample_prediction["label"]

    def test_save_prediction_auto_timestamp(self, temp_dir):
        """Test that timestamp is auto-generated."""
        output_file = temp_dir / "predictions.csv"

        record = save_prediction(
            text="Market up",
            source="reddit",
            label="positive",
            confidence=0.9,
            output_file=output_file,
        )

        # Should have a timestamp
        assert record.timestamp is not None
        datetime.fromisoformat(record.timestamp)

    def test_save_prediction_creates_directories(self, temp_dir):
        """Test that nested directories are created."""
        output_file = temp_dir / "data" / "predictions" / "test.csv"

        save_prediction(
            text="Market up",
            source="reddit",
            label="positive",
            confidence=0.9,
            output_file=output_file,
        )

        assert output_file.exists()


class TestSavePredictionsBatch:
    """Test saving multiple predictions."""

    def test_save_batch_csv(self, temp_dir, sample_predictions):
        """Test saving batch of predictions to CSV."""
        output_file = temp_dir / "predictions.csv"

        count = save_predictions_batch(sample_predictions, output_file, format="csv")

        assert count == len(sample_predictions)
        assert output_file.exists()

        # Verify all predictions were saved
        loaded = load_predictions(output_file)
        assert len(loaded) == len(sample_predictions)

    def test_save_batch_json(self, temp_dir, sample_predictions):
        """Test saving batch of predictions to JSON."""
        output_file = temp_dir / "predictions.json"

        count = save_predictions_batch(sample_predictions, output_file, format="json")

        assert count == len(sample_predictions)

        loaded = load_predictions(output_file)
        assert len(loaded) == len(sample_predictions)

    def test_save_batch_append_csv(self, temp_dir, sample_predictions):
        """Test appending to existing CSV file."""
        output_file = temp_dir / "predictions.csv"

        # Save first batch
        save_predictions_batch(sample_predictions[:2], output_file, format="csv")

        # Append third prediction
        save_predictions_batch(
            sample_predictions[2:], output_file, format="csv", append=True
        )

        # Should have all 3 predictions
        loaded = load_predictions(output_file)
        assert len(loaded) == 3

    def test_save_batch_overwrite_csv(self, temp_dir, sample_predictions):
        """Test overwriting CSV file."""
        output_file = temp_dir / "predictions.csv"

        # Save first batch
        save_predictions_batch(sample_predictions, output_file, format="csv")

        # Overwrite with single prediction
        save_predictions_batch(
            sample_predictions[:1], output_file, format="csv", append=False
        )

        # Should only have 1 prediction
        loaded = load_predictions(output_file)
        assert len(loaded) == 1

    def test_save_batch_append_json(self, temp_dir, sample_predictions):
        """Test appending to existing JSON file."""
        output_file = temp_dir / "predictions.json"

        # Save first batch
        save_predictions_batch(sample_predictions[:2], output_file, format="json")

        # Append third prediction
        save_predictions_batch(
            sample_predictions[2:], output_file, format="json", append=True
        )

        # Should have all 3 predictions
        loaded = load_predictions(output_file)
        assert len(loaded) == 3

    def test_save_batch_empty_list(self, temp_dir):
        """Test that saving empty list returns 0."""
        output_file = temp_dir / "predictions.csv"

        count = save_predictions_batch([], output_file)

        assert count == 0

    def test_save_batch_validates_all_predictions(self, temp_dir, sample_predictions):
        """Test that all predictions are validated before saving."""
        # Make second prediction invalid
        sample_predictions[1]["label"] = "invalid"

        output_file = temp_dir / "predictions.csv"

        with pytest.raises(ValueError, match="Prediction 1 validation failed"):
            save_predictions_batch(sample_predictions, output_file)

        # File should not be created if validation fails
        assert not output_file.exists()

    def test_save_batch_unsupported_format(self, temp_dir, sample_predictions):
        """Test that unsupported format raises error."""
        output_file = temp_dir / "predictions.xml"

        with pytest.raises(ValueError, match="Unsupported format: xml"):
            save_predictions_batch(sample_predictions, output_file, format="xml")


class TestLoadPredictions:
    """Test loading predictions from files."""

    def test_load_predictions_csv(self, temp_dir, sample_predictions):
        """Test loading predictions from CSV."""
        output_file = temp_dir / "predictions.csv"
        save_predictions_batch(sample_predictions, output_file, format="csv")

        loaded = load_predictions(output_file)

        assert len(loaded) == len(sample_predictions)
        assert loaded[0]["text"] == sample_predictions[0]["text"]
        assert loaded[0]["confidence"] == sample_predictions[0]["confidence"]

    def test_load_predictions_json(self, temp_dir, sample_predictions):
        """Test loading predictions from JSON."""
        output_file = temp_dir / "predictions.json"
        save_predictions_batch(sample_predictions, output_file, format="json")

        loaded = load_predictions(output_file)

        assert len(loaded) == len(sample_predictions)
        assert loaded[1]["label"] == sample_predictions[1]["label"]

    def test_load_predictions_auto_detect_format(self, temp_dir, sample_predictions):
        """Test that format is auto-detected from file extension."""
        csv_file = temp_dir / "predictions.csv"
        save_predictions_batch(sample_predictions, csv_file, format="csv")

        # Don't specify format
        loaded = load_predictions(csv_file)

        assert len(loaded) == len(sample_predictions)

    def test_load_predictions_file_not_found(self, temp_dir):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_predictions(temp_dir / "nonexistent.csv")

    def test_load_predictions_validation_fails(self, temp_dir):
        """Test that loading invalid data raises error."""
        output_file = temp_dir / "bad_predictions.json"

        # Manually create invalid JSON file
        with open(output_file, "w") as f:
            json.dump(
                [{"text": "Test", "source": "reddit", "label": "invalid"}],
                f,
            )

        with pytest.raises(ValueError):
            load_predictions(output_file, validate=True)

    def test_load_predictions_skip_validation(self, temp_dir):
        """Test loading without validation."""
        output_file = temp_dir / "predictions.json"

        # Create file with missing fields
        with open(output_file, "w") as f:
            json.dump([{"text": "Test", "label": "positive"}], f)

        # Should not raise error when validation is disabled
        loaded = load_predictions(output_file, validate=False)
        assert len(loaded) == 1


class TestGetStorageStats:
    """Test statistics retrieval."""

    def test_get_stats_basic(self, temp_dir, sample_predictions):
        """Test getting basic statistics."""
        output_file = temp_dir / "predictions.csv"
        save_predictions_batch(sample_predictions, output_file)

        stats = get_storage_stats(output_file)

        assert stats["total"] == 3
        assert "by_label" in stats
        assert "by_source" in stats
        assert "avg_confidence" in stats
        assert "date_range" in stats

    def test_get_stats_by_label(self, temp_dir, sample_predictions):
        """Test counting by label."""
        output_file = temp_dir / "predictions.csv"
        save_predictions_batch(sample_predictions, output_file)

        stats = get_storage_stats(output_file)

        assert stats["by_label"]["positive"] == 1
        assert stats["by_label"]["negative"] == 1
        assert stats["by_label"]["neutral"] == 1

    def test_get_stats_by_source(self, temp_dir, sample_predictions):
        """Test counting by source."""
        output_file = temp_dir / "predictions.csv"
        save_predictions_batch(sample_predictions, output_file)

        stats = get_storage_stats(output_file)

        assert stats["by_source"]["twitter"] == 1
        assert stats["by_source"]["news"] == 2

    def test_get_stats_avg_confidence(self, temp_dir, sample_predictions):
        """Test calculating average confidence."""
        output_file = temp_dir / "predictions.csv"
        save_predictions_batch(sample_predictions, output_file)

        stats = get_storage_stats(output_file)

        # (0.92 + 0.88 + 0.85) / 3 = 0.8833...
        assert 0.88 < stats["avg_confidence"] < 0.89

    def test_get_stats_date_range(self, temp_dir, sample_predictions):
        """Test getting date range."""
        output_file = temp_dir / "predictions.csv"
        save_predictions_batch(sample_predictions, output_file)

        stats = get_storage_stats(output_file)

        assert stats["date_range"]["first"] == "2026-01-04T09:00:00"
        assert stats["date_range"]["last"] == "2026-01-04T09:30:00"

    def test_get_stats_empty_file(self, temp_dir):
        """Test stats for empty file."""
        output_file = temp_dir / "empty.csv"

        # Create empty file
        with open(output_file, "w") as f:
            f.write("text,source,timestamp,label,confidence\n")

        stats = get_storage_stats(output_file)

        assert stats["total"] == 0
        assert stats["by_label"] == {}
        assert stats["avg_confidence"] == 0.0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow_csv(self, temp_dir):
        """Test complete workflow with CSV."""
        output_file = temp_dir / "workflow.csv"

        # Save single prediction
        save_prediction(
            text="Market rally",
            source="reddit",
            label="positive",
            confidence=0.95,
            output_file=output_file,
        )

        # Append more predictions
        predictions = [
            {
                "text": "Economic crisis",
                "source": "news",
                "timestamp": datetime.utcnow().isoformat(),
                "label": "negative",
                "confidence": 0.88,
            }
        ]
        save_predictions_batch(predictions, output_file, append=True)

        # Load and verify
        loaded = load_predictions(output_file)
        assert len(loaded) == 2

        # Get stats
        stats = get_storage_stats(output_file)
        assert stats["total"] == 2
        assert stats["by_label"]["positive"] == 1
        assert stats["by_label"]["negative"] == 1

    def test_full_workflow_json(self, temp_dir):
        """Test complete workflow with JSON."""
        output_file = temp_dir / "workflow.json"

        # Create batch
        predictions = [
            {
                "text": f"Text {i}",
                "source": "reddit",
                "timestamp": datetime.utcnow().isoformat(),
                "label": "positive",
                "confidence": 0.9,
            }
            for i in range(5)
        ]

        save_predictions_batch(predictions, output_file, format="json")

        loaded = load_predictions(output_file)
        assert len(loaded) == 5

        stats = get_storage_stats(output_file)
        assert stats["total"] == 5
        assert stats["by_label"]["positive"] == 5

    def test_cross_format_conversion(self, temp_dir, sample_predictions):
        """Test converting between CSV and JSON."""
        csv_file = temp_dir / "data.csv"
        json_file = temp_dir / "data.json"

        # Save as CSV
        save_predictions_batch(sample_predictions, csv_file, format="csv")

        # Load and save as JSON
        loaded_from_csv = load_predictions(csv_file)
        save_predictions_batch(loaded_from_csv, json_file, format="json")

        # Load JSON and verify
        loaded_from_json = load_predictions(json_file)

        assert len(loaded_from_json) == len(sample_predictions)
        assert loaded_from_json[0]["text"] == sample_predictions[0]["text"]
