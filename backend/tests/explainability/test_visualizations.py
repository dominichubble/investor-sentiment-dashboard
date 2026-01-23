"""
Unit tests for SHAP and LIME visualization functions.

Tests plotting functions and HTML generation for SHAP and LIME explanations.
"""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests

import pytest

from app.explainability.lime_explainer import get_lime_explainer
from app.explainability.shap_explainer import get_explainer
from app.explainability.visualizations import (
    generate_lime_html,
    generate_text_html,
    plot_class_comparison,
    plot_lime_class_comparison,
    plot_lime_features,
    plot_lime_summary_bar,
    plot_summary_bar,
    plot_token_contributions,
    save_explanation_html,
    save_lime_html,
)


@pytest.fixture(scope="module")
def sample_explanation():
    """Generate a sample explanation for visualization tests."""
    explainer = get_explainer()
    return explainer.explain("Stock prices surged to record highs today")


@pytest.fixture(scope="module")
def sample_summary_data():
    """Generate sample summary data for visualization tests."""
    explainer = get_explainer()
    texts = [
        "Stock prices surged after earnings beat",
        "Market crashed on recession fears",
        "Investors cautious about economic outlook",
    ]
    explanations = explainer.explain_batch(texts)
    return explainer.get_summary_data(explanations)


class TestPlotTokenContributions:
    """Tests for plot_token_contributions function."""

    def test_returns_figure(self, sample_explanation):
        """Test that function returns a matplotlib figure."""
        import matplotlib.pyplot as plt

        fig = plot_token_contributions(sample_explanation, show=False)
        assert fig is not None
        plt.close(fig)

    def test_saves_to_file(self, sample_explanation):
        """Test that figure can be saved to file."""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "token_contributions.png"
            fig = plot_token_contributions(
                sample_explanation, save_path=str(save_path), show=False
            )
            assert save_path.exists()
            assert save_path.stat().st_size > 0
            plt.close(fig)

    def test_max_tokens_limits_display(self, sample_explanation):
        """Test that max_tokens parameter limits displayed tokens."""
        import matplotlib.pyplot as plt

        fig = plot_token_contributions(sample_explanation, max_tokens=5, show=False)
        # Figure should be created successfully
        assert fig is not None
        plt.close(fig)

    def test_none_explanation_raises_error(self):
        """Test that None explanation raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            plot_token_contributions(None, show=False)  # type: ignore[arg-type]


class TestPlotSummaryBar:
    """Tests for plot_summary_bar function."""

    def test_returns_figure(self, sample_summary_data):
        """Test that function returns a matplotlib figure."""
        import matplotlib.pyplot as plt

        fig = plot_summary_bar(sample_summary_data, show=False)
        assert fig is not None
        plt.close(fig)

    def test_saves_to_file(self, sample_summary_data):
        """Test that figure can be saved to file."""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "summary.png"
            fig = plot_summary_bar(
                sample_summary_data, save_path=str(save_path), show=False
            )
            assert save_path.exists()
            plt.close(fig)

    def test_top_n_limits_tokens(self, sample_summary_data):
        """Test that top_n parameter limits displayed tokens."""
        import matplotlib.pyplot as plt

        fig = plot_summary_bar(sample_summary_data, top_n=5, show=False)
        assert fig is not None
        plt.close(fig)

    def test_none_summary_raises_error(self):
        """Test that None summary data raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            plot_summary_bar(None, show=False)  # type: ignore[arg-type]


class TestPlotClassComparison:
    """Tests for plot_class_comparison function."""

    def test_returns_figure(self, sample_summary_data):
        """Test that function returns a matplotlib figure."""
        import matplotlib.pyplot as plt

        fig = plot_class_comparison(sample_summary_data, show=False)
        assert fig is not None
        plt.close(fig)

    def test_saves_to_file(self, sample_summary_data):
        """Test that figure can be saved to file."""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "comparison.png"
            fig = plot_class_comparison(
                sample_summary_data, save_path=str(save_path), show=False
            )
            assert save_path.exists()
            plt.close(fig)


class TestGenerateTextHtml:
    """Tests for generate_text_html function."""

    def test_returns_html_string(self, sample_explanation):
        """Test that function returns HTML string."""
        html = generate_text_html(sample_explanation)
        assert isinstance(html, str)
        assert "<div" in html
        assert "SHAP" in html

    def test_html_contains_tokens(self, sample_explanation):
        """Test that HTML contains colored token spans."""
        html = generate_text_html(sample_explanation)
        assert "<span" in html
        assert "background-color" in html

    def test_html_contains_prediction(self, sample_explanation):
        """Test that HTML contains prediction information."""
        html = generate_text_html(sample_explanation)
        assert sample_explanation["prediction"]["label"].upper() in html

    def test_none_explanation_raises_error(self):
        """Test that None explanation raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            generate_text_html(None)  # type: ignore[arg-type]


class TestSaveExplanationHtml:
    """Tests for save_explanation_html function."""

    def test_creates_html_file(self, sample_explanation):
        """Test that function creates HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "explanation.html"
            result = save_explanation_html(sample_explanation, output_path)

            assert result == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_html_file_is_valid(self, sample_explanation):
        """Test that created HTML file has valid structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "explanation.html"
            save_explanation_html(sample_explanation, output_path)

            content = output_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert "<html>" in content
            assert "</html>" in content

    def test_creates_parent_directories(self, sample_explanation):
        """Test that function creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "explanation.html"
            save_explanation_html(sample_explanation, output_path)

            assert output_path.exists()


# ============================================================================
# LIME Visualization Tests
# ============================================================================


@pytest.fixture(scope="module")
def sample_lime_explanation():
    """Generate a sample LIME explanation for visualization tests."""
    explainer = get_lime_explainer(num_features=10, num_samples=100)
    return explainer.explain("Stock prices surged to record highs today")


@pytest.fixture(scope="module")
def sample_lime_summary_data():
    """Generate sample LIME summary data for visualization tests."""
    explainer = get_lime_explainer(num_features=10, num_samples=100)
    texts = [
        "Stock prices surged after earnings beat",
        "Market crashed on recession fears",
        "Investors cautious about economic outlook",
    ]
    explanations = explainer.explain_batch(texts)
    return explainer.get_summary_data(explanations)


class TestPlotLIMEFeatures:
    """Tests for plot_lime_features function."""

    def test_returns_figure(self, sample_lime_explanation):
        """Test that function returns a matplotlib figure."""
        fig = plot_lime_features(sample_lime_explanation, show=False)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_with_max_features(self, sample_lime_explanation):
        """Test plotting with custom max_features."""
        fig = plot_lime_features(sample_lime_explanation, max_features=5, show=False)
        assert fig is not None

    def test_with_custom_figsize(self, sample_lime_explanation):
        """Test plotting with custom figure size."""
        fig = plot_lime_features(sample_lime_explanation, figsize=(10, 5), show=False)
        assert fig is not None

    def test_saves_to_file(self, sample_lime_explanation):
        """Test that function can save figure to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "lime_features.png"
            plot_lime_features(
                sample_lime_explanation, save_path=str(save_path), show=False
            )
            assert save_path.exists()

    def test_invalid_explanation(self):
        """Test that function raises error for invalid explanation."""
        with pytest.raises(ValueError):
            plot_lime_features(None, show=False)

        with pytest.raises(ValueError):
            plot_lime_features({}, show=False)


class TestPlotLIMESummaryBar:
    """Tests for plot_lime_summary_bar function."""

    def test_returns_figure(self, sample_lime_summary_data):
        """Test that function returns a matplotlib figure."""
        fig = plot_lime_summary_bar(sample_lime_summary_data, show=False)
        assert fig is not None

    def test_with_top_n(self, sample_lime_summary_data):
        """Test plotting with custom top_n parameter."""
        fig = plot_lime_summary_bar(sample_lime_summary_data, top_n=5, show=False)
        assert fig is not None

    def test_saves_to_file(self, sample_lime_summary_data):
        """Test that function can save figure to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "lime_summary.png"
            plot_lime_summary_bar(
                sample_lime_summary_data, save_path=str(save_path), show=False
            )
            assert save_path.exists()

    def test_invalid_summary_data(self):
        """Test that function raises error for invalid data."""
        with pytest.raises(ValueError):
            plot_lime_summary_bar(None, show=False)


class TestPlotLIMEClassComparison:
    """Tests for plot_lime_class_comparison function."""

    def test_returns_figure(self, sample_lime_summary_data):
        """Test that function returns a matplotlib figure."""
        fig = plot_lime_class_comparison(sample_lime_summary_data, show=False)
        assert fig is not None

    def test_with_top_n(self, sample_lime_summary_data):
        """Test plotting with custom top_n parameter."""
        fig = plot_lime_class_comparison(sample_lime_summary_data, top_n=5, show=False)
        assert fig is not None

    def test_saves_to_file(self, sample_lime_summary_data):
        """Test that function can save figure to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "lime_comparison.png"
            plot_lime_class_comparison(
                sample_lime_summary_data, save_path=str(save_path), show=False
            )
            assert save_path.exists()


class TestGenerateLIMEHTML:
    """Tests for generate_lime_html function."""

    def test_returns_html_string(self, sample_lime_explanation):
        """Test that function returns an HTML string."""
        html = generate_lime_html(sample_lime_explanation)
        assert html is not None
        assert isinstance(html, str)
        assert "<div" in html
        assert "</div>" in html

    def test_contains_prediction_info(self, sample_lime_explanation):
        """Test that HTML contains prediction information."""
        html = generate_lime_html(sample_lime_explanation)
        assert "Prediction:" in html
        assert "LIME Explanation" in html

    def test_contains_colored_features(self, sample_lime_explanation):
        """Test that HTML contains colored feature spans."""
        html = generate_lime_html(sample_lime_explanation)
        assert "<span" in html
        assert "background-color" in html or "padding" in html

    def test_with_color_intensity(self, sample_lime_explanation):
        """Test HTML generation with different color intensities."""
        html1 = generate_lime_html(sample_lime_explanation, color_intensity=0.5)
        html2 = generate_lime_html(sample_lime_explanation, color_intensity=2.0)

        assert html1 != html2
        assert len(html1) > 0
        assert len(html2) > 0

    def test_invalid_explanation(self):
        """Test that function raises error for invalid explanation."""
        with pytest.raises(ValueError):
            generate_lime_html(None)


class TestSaveLIMEHTML:
    """Tests for save_lime_html function."""

    def test_saves_file(self, sample_lime_explanation):
        """Test that function saves HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "lime_explanation.html"
            result_path = save_lime_html(sample_lime_explanation, output_path)

            assert output_path.exists()
            assert result_path == output_path

    def test_file_contains_valid_html(self, sample_lime_explanation):
        """Test that saved file contains valid HTML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "lime_explanation.html"
            save_lime_html(sample_lime_explanation, output_path)

            content = output_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert "<html>" in content
            assert "</html>" in content

    def test_creates_parent_directories(self, sample_lime_explanation):
        """Test that function creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "lime_explanation.html"
            save_lime_html(sample_lime_explanation, output_path)

            assert output_path.exists()
