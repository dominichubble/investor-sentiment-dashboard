"""
Tests for LIME Explainer Module

Tests the LIMEExplainer class and its methods for generating
local explanations for FinBERT sentiment predictions.
"""

import pytest

from app.explainability.lime_explainer import LIMEExplainer, get_lime_explainer


class TestLIMEExplainer:
    """Test suite for LIMEExplainer class."""

    @pytest.fixture
    def explainer(self):
        """Create a LIME explainer instance."""
        return LIMEExplainer(num_features=10, num_samples=100)

    def test_initialization(self, explainer):
        """Test that explainer initializes correctly."""
        assert explainer is not None
        assert explainer.model is not None
        assert explainer.explainer is not None
        assert explainer.num_features == 10
        assert explainer.num_samples == 100

    def test_explain_positive_text(self, explainer):
        """Test explanation for positive sentiment text."""
        text = "Stock prices surged to record highs today"
        explanation = explainer.explain(text)

        assert explanation is not None
        assert explanation["text"] == text
        assert "prediction" in explanation
        assert "feature_weights" in explanation
        assert "top_features" in explanation
        assert len(explanation["top_features"]) <= 10
        assert explanation["predicted_class"] in ["positive", "negative", "neutral"]

    def test_explain_negative_text(self, explainer):
        """Test explanation for negative sentiment text."""
        text = "Markets crashed and losses were devastating"
        explanation = explainer.explain(text)

        assert explanation is not None
        assert explanation["text"] == text
        assert "prediction" in explanation
        assert "feature_weights" in explanation
        assert len(explanation["top_features"]) > 0

    def test_explain_neutral_text(self, explainer):
        """Test explanation for neutral sentiment text."""
        text = "The company reported quarterly earnings"
        explanation = explainer.explain(text)

        assert explanation is not None
        assert explanation["text"] == text
        assert "prediction" in explanation

    def test_explain_custom_features(self, explainer):
        """Test explanation with custom num_features."""
        text = "Profit margins increased significantly"
        explanation = explainer.explain(text, num_features=5)

        assert len(explanation["top_features"]) <= 5

    def test_explain_invalid_text(self, explainer):
        """Test that invalid text raises ValueError."""
        with pytest.raises(ValueError):
            explainer.explain("")

        with pytest.raises(ValueError):
            explainer.explain("   ")

    def test_explain_batch(self, explainer):
        """Test batch explanation."""
        texts = [
            "Stock prices increased",
            "Markets are declining",
            "Earnings report released",
        ]

        explanations = explainer.explain_batch(texts)

        assert len(explanations) == 3
        for exp in explanations:
            assert exp is not None
            assert "prediction" in exp
            assert "feature_weights" in exp

    def test_explain_batch_empty_list(self, explainer):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError):
            explainer.explain_batch([])

    def test_feature_weights_structure(self, explainer):
        """Test that feature weights are properly structured."""
        text = "Revenue grew by twenty percent"
        explanation = explainer.explain(text)

        feature_weights = explanation["feature_weights"]
        assert isinstance(feature_weights, dict)

        for feature, weight in feature_weights.items():
            assert isinstance(feature, str)
            assert isinstance(weight, (int, float))

    def test_all_class_weights(self, explainer):
        """Test that all class weights are included."""
        text = "Stocks rallied on positive news"
        explanation = explainer.explain(text)

        all_class_weights = explanation["all_class_weights"]
        assert "positive" in all_class_weights
        assert "negative" in all_class_weights
        assert "neutral" in all_class_weights

    def test_get_summary_data(self, explainer):
        """Test summary data aggregation."""
        texts = [
            "Profits soared significantly",
            "Revenue increased dramatically",
            "Stock prices jumped higher",
        ]

        explanations = explainer.explain_batch(texts, num_features=5)
        summary_data = explainer.get_summary_data(explanations, top_n=10)

        assert summary_data is not None
        assert "feature_importance" in summary_data
        assert "top_features" in summary_data
        assert "class_feature_importance" in summary_data
        assert "top_positive_features" in summary_data
        assert "top_negative_features" in summary_data
        assert summary_data["num_explanations"] == 3

    def test_get_summary_data_empty(self, explainer):
        """Test that empty explanations list raises ValueError."""
        with pytest.raises(ValueError):
            explainer.get_summary_data([])

    def test_get_summary_data_with_failures(self, explainer):
        """Test summary data with some failed explanations."""
        explanations = [
            explainer.explain("Stock prices increased"),
            None,  # Simulated failure
            explainer.explain("Markets are stable"),
        ]

        summary_data = explainer.get_summary_data(explanations, top_n=5)

        assert summary_data["num_explanations"] == 2

    def test_predict_proba(self, explainer):
        """Test the prediction probability function."""
        texts = ["Stocks rallied", "Markets crashed"]
        probs = explainer._predict_proba(texts)

        assert probs.shape == (2, 3)  # 2 texts, 3 classes
        assert all(0 <= p <= 1 for row in probs for p in row)

    def test_predict_proba_empty_text(self, explainer):
        """Test prediction with empty text."""
        texts = ["", "  ", "Valid text"]
        probs = explainer._predict_proba(texts)

        assert probs.shape == (3, 3)

    def test_lime_explanation_object(self, explainer):
        """Test that LIME explanation object is included."""
        text = "Company revenues exceeded expectations"
        explanation = explainer.explain(text)

        assert "lime_explanation" in explanation
        assert explanation["lime_explanation"] is not None

    def test_local_prediction(self, explainer):
        """Test that local prediction is included."""
        text = "Earnings were strong and positive"
        explanation = explainer.explain(text)

        assert "local_prediction" in explanation
        # Local prediction can be None or a float


class TestGetLIMEExplainer:
    """Test suite for get_lime_explainer singleton function."""

    def test_get_lime_explainer(self):
        """Test that get_lime_explainer returns a valid instance."""
        explainer = get_lime_explainer()

        assert explainer is not None
        assert isinstance(explainer, LIMEExplainer)

    def test_singleton_pattern(self):
        """Test that get_lime_explainer returns the same instance."""
        explainer1 = get_lime_explainer()
        explainer2 = get_lime_explainer()

        assert explainer1 is explainer2


class TestLIMEExplainerIntegration:
    """Integration tests for LIME explainer with realistic scenarios."""

    @pytest.fixture
    def explainer(self):
        """Create a LIME explainer instance."""
        return LIMEExplainer(num_features=5, num_samples=100)

    def test_financial_positive_text(self, explainer):
        """Test explanation for positive financial text."""
        text = "The company announced record profits and strong growth"
        explanation = explainer.explain(text)

        # Check that explanation makes sense
        assert explanation is not None
        feature_weights = explanation["feature_weights"]

        # Positive words should have some weight
        positive_words = ["record", "profits", "strong", "growth"]
        found_features = [w for w in positive_words if w in feature_weights]
        assert len(found_features) > 0

    def test_financial_negative_text(self, explainer):
        """Test explanation for negative financial text."""
        text = "Stock prices plummeted amid bankruptcy fears"
        explanation = explainer.explain(text)

        assert explanation is not None
        feature_weights = explanation["feature_weights"]

        # Negative words should have some weight
        negative_words = ["plummeted", "bankruptcy", "fears"]
        found_features = [w for w in negative_words if w in feature_weights]
        assert len(found_features) > 0

    def test_diverse_texts_batch(self, explainer):
        """Test batch explanation with diverse texts."""
        texts = [
            "Revenue exceeded expectations significantly",
            "The market experienced heavy losses",
            "Quarterly report was published today",
            "Investor confidence reached new peaks",
            "Economic outlook remains uncertain",
        ]

        explanations = explainer.explain_batch(texts, num_features=5)

        assert len(explanations) == 5
        successful = sum(1 for e in explanations if e is not None)
        assert successful >= 4  # At least 4 should succeed

    def test_short_text(self, explainer):
        """Test explanation for very short text."""
        text = "Good news"
        explanation = explainer.explain(text, num_features=5)

        assert explanation is not None
        assert len(explanation["top_features"]) > 0

    def test_long_text(self, explainer):
        """Test explanation for longer text."""
        text = (
            "The financial markets showed remarkable resilience today as "
            "major indices climbed higher following the release of positive "
            "economic data. Investors responded enthusiastically to news of "
            "strong corporate earnings and improved economic forecasts."
        )
        explanation = explainer.explain(text, num_features=10)

        assert explanation is not None
        assert len(explanation["top_features"]) <= 10
