"""
SHAP Visualization Module for FinBERT Explanations

Provides functions for generating SHAP summary plots and token-level
visualizations to understand FinBERT sentiment predictions.

Usage:
    from app.explainability import SHAPExplainer
    from app.explainability.visualizations import (
        plot_token_contributions,
        plot_summary_bar,
        save_explanation_html,
    )

    explainer = SHAPExplainer()
    explanation = explainer.explain("Stock prices crashed today")

    # Plot single explanation
    plot_token_contributions(explanation)

    # Generate summary across multiple texts
    explanations = explainer.explain_batch(texts)
    summary_data = explainer.get_summary_data(explanations)
    plot_summary_bar(summary_data)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

logger = logging.getLogger(__name__)

# Color scheme for sentiment classes
SENTIMENT_COLORS = {
    "positive": "#2ecc71",  # Green
    "negative": "#e74c3c",  # Red
    "neutral": "#95a5a6",  # Gray
}

# Color scheme for SHAP contributions
CONTRIBUTION_COLORS = {
    "positive_contribution": "#ff6b6b",  # Warm red (pushes toward label)
    "negative_contribution": "#4ecdc4",  # Teal (pushes away from label)
}


def plot_token_contributions(
    explanation: Dict,
    figsize: Tuple[int, int] = (12, 6),
    max_tokens: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """
    Create a horizontal bar chart showing token contributions for an explanation.

    Args:
        explanation: Explanation dictionary from SHAPExplainer.explain()
        figsize: Figure size (width, height)
        max_tokens: Maximum number of tokens to display
        save_path: If provided, save figure to this path
        show: If True, display the figure

    Returns:
        Matplotlib Figure object
    """
    if not explanation:
        raise ValueError("explanation cannot be None or empty")

    target_class = explanation["target_class"]
    contributions = explanation["token_contributions"]
    prediction = explanation["prediction"]

    # Sort by absolute contribution and limit
    sorted_contributions = sorted(
        contributions, key=lambda x: abs(x[1]), reverse=True
    )[:max_tokens]

    # Reverse for bottom-to-top display
    sorted_contributions = list(reversed(sorted_contributions))

    tokens = [tc[0] for tc in sorted_contributions]
    values = [tc[1] for tc in sorted_contributions]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color based on contribution direction
    colors = [
        CONTRIBUTION_COLORS["positive_contribution"]
        if v > 0
        else CONTRIBUTION_COLORS["negative_contribution"]
        for v in values
    ]

    # Create horizontal bar chart
    bars = ax.barh(tokens, values, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, value in zip(bars, values):
        label_x = value + (0.01 if value >= 0 else -0.01)
        ha = "left" if value >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha=ha,
            fontsize=9,
        )

    # Add vertical line at zero
    ax.axvline(x=0, color="black", linewidth=0.8)

    # Labels and title
    ax.set_xlabel("SHAP Value (contribution to prediction)", fontsize=11)
    ax.set_ylabel("Token", fontsize=11)

    title = (
        f"Token Contributions to '{target_class.upper()}' Prediction\n"
        f"Predicted: {prediction['label']} ({prediction['score']:.1%} confidence)"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=CONTRIBUTION_COLORS["positive_contribution"],
            label=f"Pushes toward {target_class}",
        ),
        Patch(
            facecolor=CONTRIBUTION_COLORS["negative_contribution"],
            label=f"Pushes away from {target_class}",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_summary_bar(
    summary_data: Dict,
    figsize: Tuple[int, int] = (14, 8),
    top_n: int = 15,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """
    Create a summary bar plot showing top tokens by importance.

    Args:
        summary_data: Summary data from SHAPExplainer.get_summary_data()
        figsize: Figure size (width, height)
        top_n: Number of top tokens to display
        save_path: If provided, save figure to this path
        show: If True, display the figure

    Returns:
        Matplotlib Figure object
    """
    if not summary_data:
        raise ValueError("summary_data cannot be None or empty")

    top_tokens = summary_data["top_tokens"][:top_n]

    if not top_tokens:
        raise ValueError("No tokens found in summary data")

    # Prepare data
    tokens = [t[0] for t in reversed(top_tokens)]
    importances = [t[1] for t in reversed(top_tokens)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar chart
    bars = ax.barh(
        tokens,
        importances,
        color="#3498db",
        edgecolor="white",
        linewidth=0.5,
    )

    # Add value labels
    for bar, importance in zip(bars, importances):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{importance:.4f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    # Labels and title
    ax.set_xlabel("Mean |SHAP Value| (average importance)", fontsize=11)
    ax.set_ylabel("Token", fontsize=11)
    ax.set_title(
        f"SHAP Summary: Top {len(tokens)} Most Important Tokens\n"
        f"(Aggregated from {summary_data['num_explanations']} explanations)",
        fontsize=12,
        fontweight="bold",
    )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_class_comparison(
    summary_data: Dict,
    figsize: Tuple[int, int] = (16, 10),
    top_n: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """
    Create a comparison plot showing top tokens for positive vs negative sentiment.

    Args:
        summary_data: Summary data from SHAPExplainer.get_summary_data()
        figsize: Figure size (width, height)
        top_n: Number of top tokens per class
        save_path: If provided, save figure to this path
        show: If True, display the figure

    Returns:
        Matplotlib Figure object
    """
    if not summary_data:
        raise ValueError("summary_data cannot be None or empty")

    top_positive = summary_data["top_positive_tokens"][:top_n]
    top_negative = summary_data["top_negative_tokens"][:top_n]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Positive sentiment tokens
    if top_positive:
        tokens_pos = [t[0] for t in reversed(top_positive)]
        values_pos = [t[1] for t in reversed(top_positive)]

        ax1.barh(
            tokens_pos,
            values_pos,
            color=SENTIMENT_COLORS["positive"],
            edgecolor="white",
        )
        ax1.set_xlabel("Mean SHAP Value", fontsize=11)
        ax1.set_ylabel("Token", fontsize=11)
        ax1.set_title(
            "Top Tokens → POSITIVE Sentiment",
            fontsize=12,
            fontweight="bold",
            color=SENTIMENT_COLORS["positive"],
        )
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(axis="x", alpha=0.3)

    # Negative sentiment tokens
    if top_negative:
        tokens_neg = [t[0] for t in reversed(top_negative)]
        values_neg = [t[1] for t in reversed(top_negative)]

        ax2.barh(
            tokens_neg,
            values_neg,
            color=SENTIMENT_COLORS["negative"],
            edgecolor="white",
        )
        ax2.set_xlabel("Mean SHAP Value", fontsize=11)
        ax2.set_ylabel("Token", fontsize=11)
        ax2.set_title(
            "Top Tokens → NEGATIVE Sentiment",
            fontsize=12,
            fontweight="bold",
            color=SENTIMENT_COLORS["negative"],
        )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"SHAP Class Comparison (n={summary_data['num_explanations']} texts)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def generate_text_html(
    explanation: Dict,
    color_intensity: float = 1.0,
) -> str:
    """
    Generate HTML with colored tokens based on SHAP values.

    Args:
        explanation: Explanation dictionary from SHAPExplainer.explain()
        color_intensity: Multiplier for color intensity (0.0 - 2.0)

    Returns:
        HTML string with colored tokens
    """
    if not explanation:
        raise ValueError("explanation cannot be None or empty")

    tokens = explanation["tokens"]
    target_class = explanation["target_class"]
    shap_values = explanation["shap_values"][target_class]
    prediction = explanation["prediction"]

    # Normalize SHAP values for color intensity
    max_abs = max(abs(min(shap_values)), abs(max(shap_values)))
    if max_abs == 0:
        max_abs = 1

    html_parts = []

    for token, value in zip(tokens, shap_values):
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # Calculate color intensity
        normalized = (value / max_abs) * color_intensity
        normalized = max(-1, min(1, normalized))  # Clamp to [-1, 1]

        if normalized > 0:
            # Red for positive contribution
            r, g, b = 255, int(255 * (1 - normalized)), int(255 * (1 - normalized))
        else:
            # Blue for negative contribution
            r, g, b = int(255 * (1 + normalized)), int(255 * (1 + normalized)), 255

        html_parts.append(
            f'<span style="background-color: rgb({r},{g},{b}); '
            f'padding: 2px 4px; margin: 1px; border-radius: 3px;" '
            f'title="SHAP: {value:+.4f}">{token}</span>'
        )

    # Build complete HTML
    html = f"""
    <div style="font-family: Arial, sans-serif; line-height: 2;">
        <h3>SHAP Explanation</h3>
        <p><strong>Prediction:</strong> {prediction['label'].upper()}
           ({prediction['score']:.1%} confidence)</p>
        <p><strong>Explaining:</strong> {target_class} class</p>
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
            {' '.join(html_parts)}
        </div>
        <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
            <span style="background-color: rgb(255,200,200); padding: 2px 6px;">Red</span>
            = pushes toward {target_class} |
            <span style="background-color: rgb(200,200,255); padding: 2px 6px;">Blue</span>
            = pushes away from {target_class}
        </p>
    </div>
    """

    return html


def save_explanation_html(
    explanation: Dict,
    output_path: Union[str, Path],
    color_intensity: float = 1.0,
) -> Path:
    """
    Save explanation as an HTML file with colored tokens.

    Args:
        explanation: Explanation dictionary from SHAPExplainer.explain()
        output_path: Path to save the HTML file
        color_intensity: Multiplier for color intensity

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>SHAP Explanation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
        </style>
    </head>
    <body>
        {generate_text_html(explanation, color_intensity)}
    </body>
    </html>
    """

    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"HTML explanation saved to {output_path}")

    return output_path
