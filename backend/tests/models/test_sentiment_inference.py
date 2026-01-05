"""
Unit tests for sentiment_inference module.

Tests the high-level inference API including single predictions,
batch processing, metadata handling, and summary statistics.
"""

import pytest

from app.models.sentiment_inference import (
    analyze_batch,
    analyze_sentiment,
    analyze_with_metadata,
    get_sentiment_summary,
)


class TestAnalyzeSentiment:
    """Tests for single text sentiment analysis."""

    def test_analyze_positive_sentiment(self):
        """Test analysis of positive financial text."""
        text = "Stock prices surged to record highs today"
        result = analyze_sentiment(text)

        assert "label" in result
        assert "score" in result
        assert result["label"] in ["positive", "negative", "neutral"]
        assert 0 <= result["score"] <= 1
        assert result["label"] == "positive"  # Expected outcome

    def test_analyze_negative_sentiment(self):
        """Test analysis of negative financial text."""
        text = "Company reports massive losses and layoffs"
        result = analyze_sentiment(text)

        assert result["label"] in ["positive", "negative", "neutral"]
        assert 0 <= result["score"] <= 1
        assert result["label"] == "negative"  # Expected outcome

    def test_analyze_neutral_sentiment(self):
        """Test analysis of neutral financial text."""
        text = "The company released quarterly earnings report"
        result = analyze_sentiment(text)

        assert result["label"] in ["positive", "negative", "neutral"]
        assert 0 <= result["score"] <= 1

    def test_analyze_with_all_scores(self):
        """Test that return_all_scores returns all label probabilities."""
        text = "Market outlook remains uncertain"
        result = analyze_sentiment(text, return_all_scores=True)

        assert "scores" in result
        assert isinstance(result["scores"], dict)
        assert "positive" in result["scores"]
        assert "negative" in result["scores"]
        assert "neutral" in result["scores"]

        # Scores should sum to approximately 1
        total = sum(result["scores"].values())
        assert 0.99 <= total <= 1.01

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Text input cannot be blank"):
            analyze_sentiment("")

    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Text input cannot be blank"):
            analyze_sentiment("   ")

    def test_none_input_raises_error(self):
        """Test that None input raises ValueError."""
        with pytest.raises(ValueError, match="Text input must be a non-empty string"):
            analyze_sentiment(None)

    def test_non_string_input_raises_error(self):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="Text input must be a non-empty string"):
            analyze_sentiment(123)


class TestAnalyzeBatch:
    """Tests for batch sentiment analysis."""

    def test_batch_analysis_multiple_texts(self):
        """Test batch analysis with multiple texts."""
        texts = [
            "Stock market rally continues",
            "Company faces bankruptcy",
            "Earnings meet expectations",
        ]
        results = analyze_batch(texts)

        assert len(results) == len(texts)
        assert all(r is not None for r in results)
        assert all("label" in r for r in results)
        assert all("score" in r for r in results)
        assert all(r["label"] in ["positive", "negative", "neutral"] for r in results)

    def test_batch_analysis_single_text(self):
        """Test batch analysis with single text."""
        texts = ["Market conditions improving"]
        results = analyze_batch(texts)

        assert len(results) == 1
        assert results[0]["label"] in ["positive", "negative", "neutral"]

    def test_batch_analysis_custom_batch_size(self):
        """Test batch analysis with custom batch size."""
        texts = ["Text " + str(i) for i in range(10)]
        results = analyze_batch(texts, batch_size=5)

        assert len(results) == len(texts)
        assert all(r is not None for r in results)

    def test_batch_analysis_with_all_scores(self):
        """Test batch analysis returns all scores when requested."""
        texts = ["Good news", "Bad news"]
        results = analyze_batch(texts, return_all_scores=True)

        assert len(results) == 2
        assert all("scores" in r for r in results)
        assert all(isinstance(r["scores"], dict) for r in results)

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Texts must be a non-empty list"):
            analyze_batch([])

    def test_non_list_input_raises_error(self):
        """Test that non-list input raises ValueError."""
        with pytest.raises(ValueError, match="Texts must be a non-empty list"):
            analyze_batch("single text")

    def test_mixed_type_list_raises_error(self):
        """Test that list with non-string items raises ValueError."""
        with pytest.raises(ValueError, match="All items in texts must be strings"):
            analyze_batch(["text", 123, "more text"])

    def test_batch_with_empty_strings(self):
        """Test batch analysis handles empty strings gracefully."""
        texts = ["Valid text", "", "Another valid text", "   "]
        results = analyze_batch(texts)

        assert len(results) == 4
        assert results[0] is not None  # Valid
        assert results[1] is None  # Empty
        assert results[2] is not None  # Valid
        assert results[3] is None  # Whitespace only

    def test_all_empty_strings_raises_error(self):
        """Test that all empty strings raises ValueError."""
        with pytest.raises(ValueError, match="No valid"):
            analyze_batch(["", "  ", ""])


class TestAnalyzeWithMetadata:
    """Tests for sentiment analysis with metadata."""

    def test_analyze_with_metadata_basic(self):
        """Test basic metadata attachment."""
        text = "Stock price increased"
        metadata = {"post_id": "123", "author": "user1"}

        result = analyze_with_metadata(text, metadata)

        assert "label" in result
        assert "score" in result
        assert "text" in result
        assert "text_length" in result
        assert "metadata" in result
        assert result["metadata"] == metadata
        assert result["text_length"] == len(text)

    def test_analyze_with_no_metadata(self):
        """Test analysis without metadata."""
        text = "Market volatility increases"
        result = analyze_with_metadata(text)

        assert "metadata" in result
        assert result["metadata"] == {}

    def test_text_truncation(self):
        """Test that long text is truncated in result."""
        text = "A" * 200  # 200 characters
        result = analyze_with_metadata(text)

        assert len(result["text"]) == 100
        assert result["text_length"] == 200

    def test_metadata_with_complex_types(self):
        """Test metadata with nested structures."""
        text = "Earnings report published"
        metadata = {
            "timestamp": "2026-01-04",
            "source": {"platform": "reddit", "subreddit": "wallstreetbets"},
            "metrics": {"upvotes": 100, "comments": 50},
        }

        result = analyze_with_metadata(text, metadata)

        assert result["metadata"] == metadata
        assert result["metadata"]["source"]["platform"] == "reddit"


class TestGetSentimentSummary:
    """Tests for sentiment summary statistics."""

    def test_summary_basic(self):
        """Test summary generation from results."""
        results = [
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.8},
            {"label": "neutral", "score": 0.7},
        ]

        summary = get_sentiment_summary(results)

        assert summary["total"] == 3
        assert summary["counts"]["positive"] == 1
        assert summary["counts"]["negative"] == 1
        assert summary["counts"]["neutral"] == 1
        assert summary["average_confidence"] > 0

    def test_summary_with_duplicates(self):
        """Test summary with multiple same labels."""
        results = [
            {"label": "positive", "score": 0.9},
            {"label": "positive", "score": 0.85},
            {"label": "negative", "score": 0.7},
        ]

        summary = get_sentiment_summary(results)

        assert summary["total"] == 3
        assert summary["counts"]["positive"] == 2
        assert summary["counts"]["negative"] == 1
        assert summary["counts"]["neutral"] == 0
        assert summary["percentages"]["positive"] == pytest.approx(66.67, rel=0.01)

    def test_summary_empty_results(self):
        """Test summary with empty results list."""
        summary = get_sentiment_summary([])

        assert summary["total"] == 0
        assert all(count == 0 for count in summary["counts"].values())
        assert all(pct == 0.0 for pct in summary["percentages"].values())
        assert summary["average_confidence"] == 0.0

    def test_summary_with_none_results(self):
        """Test summary filters out None results."""
        results = [
            {"label": "positive", "score": 0.9},
            None,
            {"label": "negative", "score": 0.8},
            None,
        ]

        summary = get_sentiment_summary(results)

        assert summary["total"] == 2  # Only non-None results
        assert summary["counts"]["positive"] == 1
        assert summary["counts"]["negative"] == 1

    def test_summary_percentages_sum_to_100(self):
        """Test that percentages sum to 100."""
        results = [
            {"label": "positive", "score": 0.9},
            {"label": "positive", "score": 0.85},
            {"label": "negative", "score": 0.7},
            {"label": "neutral", "score": 0.75},
        ]

        summary = get_sentiment_summary(results)

        total_percentage = sum(summary["percentages"].values())
        assert total_percentage == pytest.approx(100.0, rel=0.01)

    def test_average_confidence_calculation(self):
        """Test average confidence score calculation."""
        results = [
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.6},
            {"label": "neutral", "score": 1.0},
        ]

        summary = get_sentiment_summary(results)

        expected_avg = (0.8 + 0.6 + 1.0) / 3
        assert summary["average_confidence"] == pytest.approx(expected_avg, rel=0.0001)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self):
        """Test complete workflow: batch analysis -> summary."""
        texts = [
            "Company stock soars on earnings beat",
            "Market crash imminent warn analysts",
            "Fed maintains interest rates unchanged",
            "Record profits announced by tech giant",
        ]

        # Analyze batch
        results = analyze_batch(texts)
        assert len(results) == 4
        assert all(r is not None for r in results)

        # Generate summary
        summary = get_sentiment_summary(results)
        assert summary["total"] == 4
        assert sum(summary["counts"].values()) == 4

    def test_workflow_with_metadata(self):
        """Test workflow with metadata tracking."""
        texts_with_meta = [
            ("Stock up 20%", {"source": "twitter", "id": "1"}),
            ("Layoffs announced", {"source": "news", "id": "2"}),
        ]

        results = []
        for text, meta in texts_with_meta:
            result = analyze_with_metadata(text, meta)
            results.append(result)

        assert len(results) == 2
        assert all("metadata" in r for r in results)
        assert results[0]["metadata"]["source"] == "twitter"
        assert results[1]["metadata"]["source"] == "news"
