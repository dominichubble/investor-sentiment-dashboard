"""
Unit tests for SHAP explainer module.

Tests the SHAPExplainer class including single explanations,
batch processing, summary generation, and error handling.
"""

import pytest

from app.explainability.shap_explainer import SHAPExplainer, get_explainer


class TestSHAPExplainer:
    """Tests for SHAPExplainer class."""

    @pytest.fixture(scope="class")
    def explainer(self):
        """Create explainer instance (shared across tests for efficiency)."""
        return get_explainer()

    def test_explainer_initialization(self, explainer):
        """Test that explainer initializes correctly."""
        assert explainer is not None
        assert explainer.model is not None
        assert explainer.explainer is not None
        assert explainer.LABELS == ["positive", "negative", "neutral"]

    def test_explain_positive_text(self, explainer):
        """Test explanation for positive sentiment text."""
        text = "Stock prices surged to record highs today"
        explanation = explainer.explain(text)

        # Check required keys
        assert "text" in explanation
        assert "prediction" in explanation
        assert "tokens" in explanation
        assert "shap_values" in explanation
        assert "target_class" in explanation
        assert "token_contributions" in explanation
        assert "top_contributors" in explanation

        # Check prediction structure
        assert explanation["prediction"]["label"] in ["positive", "negative", "neutral"]
        assert 0 <= explanation["prediction"]["score"] <= 1

        # Check SHAP values structure
        assert "positive" in explanation["shap_values"]
        assert "negative" in explanation["shap_values"]
        assert "neutral" in explanation["shap_values"]

    def test_explain_negative_text(self, explainer):
        """Test explanation for negative sentiment text."""
        text = "Company reports massive losses and announces layoffs"
        explanation = explainer.explain(text)

        assert explanation["text"] == text
        assert len(explanation["tokens"]) > 0
        assert len(explanation["token_contributions"]) > 0

    def test_explain_with_target_class(self, explainer):
        """Test explanation with explicit target class."""
        text = "Market outlook remains uncertain"
        explanation = explainer.explain(text, target_class="negative")

        assert explanation["target_class"] == "negative"
        # SHAP values should be for negative class
        assert len(explanation["shap_values"]["negative"]) == len(explanation["tokens"])

    def test_explain_all_target_classes(self, explainer):
        """Test that all target classes can be explained."""
        text = "Stock market opens flat today"

        for target_class in ["positive", "negative", "neutral"]:
            explanation = explainer.explain(text, target_class=target_class)
            assert explanation["target_class"] == target_class

    def test_token_contributions_sum(self, explainer):
        """Test that token contributions are reasonable."""
        text = "Investors are optimistic about earnings"
        explanation = explainer.explain(text)

        contributions = explanation["token_contributions"]

        # All contributions should be finite numbers
        for token, value in contributions:
            assert isinstance(token, str)
            assert isinstance(value, float)
            assert not (value != value)  # Check for NaN

    def test_top_contributors_sorted(self, explainer):
        """Test that top contributors are sorted by absolute value."""
        text = "Stock crashed after disappointing earnings report"
        explanation = explainer.explain(text)

        top = explanation["top_contributors"]

        # Should be sorted by absolute value (descending)
        for i in range(len(top) - 1):
            assert abs(top[i][1]) >= abs(top[i + 1][1])

    def test_empty_string_raises_error(self, explainer):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            explainer.explain("")

    def test_whitespace_only_raises_error(self, explainer):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            explainer.explain("   ")

    def test_invalid_target_class_raises_error(self, explainer):
        """Test that invalid target class raises ValueError."""
        with pytest.raises(ValueError, match="must be one of"):
            explainer.explain("Test text", target_class="invalid")


class TestExplainBatch:
    """Tests for batch explanation functionality."""

    @pytest.fixture(scope="class")
    def explainer(self):
        """Create explainer instance."""
        return get_explainer()

    def test_batch_multiple_texts(self, explainer):
        """Test batch explanation with multiple texts."""
        texts = [
            "Stock market rally continues",
            "Company faces bankruptcy",
            "Quarterly earnings meet expectations",
        ]
        explanations = explainer.explain_batch(texts)

        assert len(explanations) == len(texts)
        assert all(e is not None for e in explanations)
        assert all("prediction" in e for e in explanations)

    def test_batch_empty_list_raises_error(self, explainer):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            explainer.explain_batch([])

    def test_batch_with_target_class(self, explainer):
        """Test batch explanation with specific target class."""
        texts = ["Market up", "Market down"]
        explanations = explainer.explain_batch(texts, target_class="positive")

        assert all(e["target_class"] == "positive" for e in explanations)


class TestGetSummaryData:
    """Tests for summary data generation."""

    @pytest.fixture(scope="class")
    def explainer(self):
        """Create explainer instance."""
        return get_explainer()

    @pytest.fixture(scope="class")
    def sample_explanations(self, explainer):
        """Generate sample explanations for summary tests."""
        texts = [
            "Stock prices surged after positive earnings",
            "Market crashed on recession fears",
            "Investors remain cautious about outlook",
        ]
        return explainer.explain_batch(texts)

    def test_summary_data_structure(self, explainer, sample_explanations):
        """Test that summary data has correct structure."""
        summary = explainer.get_summary_data(sample_explanations)

        assert "token_importance" in summary
        assert "top_tokens" in summary
        assert "class_token_importance" in summary
        assert "top_positive_tokens" in summary
        assert "top_negative_tokens" in summary
        assert "num_explanations" in summary

    def test_summary_num_explanations(self, explainer, sample_explanations):
        """Test that num_explanations matches input."""
        summary = explainer.get_summary_data(sample_explanations)
        assert summary["num_explanations"] == len(sample_explanations)

    def test_summary_top_tokens_limit(self, explainer, sample_explanations):
        """Test that top_n parameter limits results."""
        summary = explainer.get_summary_data(sample_explanations, top_n=5)
        assert len(summary["top_tokens"]) <= 5

    def test_summary_empty_list_raises_error(self, explainer):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            explainer.get_summary_data([])


class TestGetExplainer:
    """Tests for singleton explainer pattern."""

    def test_get_explainer_returns_instance(self):
        """Test that get_explainer returns an instance."""
        explainer = get_explainer()
        assert isinstance(explainer, SHAPExplainer)

    def test_get_explainer_singleton(self):
        """Test that get_explainer returns same instance."""
        explainer1 = get_explainer()
        explainer2 = get_explainer()
        assert explainer1 is explainer2
