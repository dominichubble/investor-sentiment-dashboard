"""
Unit tests for SHAP visualization functions.

Tests plotting functions and HTML generation for SHAP explanations.
"""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests

import pytest

from app.explainability.shap_explainer import get_explainer
from app.explainability.visualizations import (
    generate_text_html,
    plot_class_comparison,
    plot_summary_bar,
    plot_token_contributions,
    save_explanation_html,
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
