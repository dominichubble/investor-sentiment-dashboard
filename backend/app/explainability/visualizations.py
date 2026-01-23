"""
Visualization Module for FinBERT Explanations (SHAP & LIME)

Provides functions for generating SHAP and LIME summary plots and token-level
visualizations to understand FinBERT sentiment predictions.

Usage:
    from app.explainability import SHAPExplainer, LIMEExplainer
    from app.explainability.visualizations import (
        plot_token_contributions,
        plot_summary_bar,
        plot_lime_features,
        save_explanation_html,
        save_lime_html,
    )

    # SHAP
    explainer = SHAPExplainer()
    explanation = explainer.explain("Stock prices crashed today")
    plot_token_contributions(explanation)

    # LIME
    lime_explainer = LIMEExplainer()
    lime_explanation = lime_explainer.explain("Stock prices crashed today")
    plot_lime_features(lime_explanation)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

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
    sorted_contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[
        :max_tokens
    ]

    # Reverse for bottom-to-top display
    sorted_contributions = list(reversed(sorted_contributions))

    tokens = [tc[0] for tc in sorted_contributions]
    values = [tc[1] for tc in sorted_contributions]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color based on contribution direction
    colors = [
        (
            CONTRIBUTION_COLORS["positive_contribution"]
            if v > 0
            else CONTRIBUTION_COLORS["negative_contribution"]
        )
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


# ============================================================================
# LIME Visualization Functions
# ============================================================================


def plot_lime_features(
    explanation: Dict,
    figsize: Tuple[int, int] = (12, 6),
    max_features: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """
    Create a horizontal bar chart showing LIME feature weights.

    Args:
        explanation: Explanation dictionary from LIMEExplainer.explain()
        figsize: Figure size (width, height)
        max_features: Maximum number of features to display
        save_path: If provided, save figure to this path
        show: If True, display the figure

    Returns:
        Matplotlib Figure object
    """
    if not explanation:
        raise ValueError("explanation cannot be None or empty")

    predicted_class = explanation["predicted_class"]
    top_features = explanation["top_features"][:max_features]
    prediction = explanation["prediction"]

    # Reverse for bottom-to-top display
    top_features = list(reversed(top_features))

    features = [f[0] for f in top_features]
    weights = [f[1] for f in top_features]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color based on weight direction
    colors = [
        (
            CONTRIBUTION_COLORS["positive_contribution"]
            if w > 0
            else CONTRIBUTION_COLORS["negative_contribution"]
        )
        for w in weights
    ]

    # Create horizontal bar chart
    bars = ax.barh(features, weights, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, weight in zip(bars, weights):
        label_x = weight + (0.01 if weight >= 0 else -0.01)
        ha = "left" if weight >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{weight:+.3f}",
            va="center",
            ha=ha,
            fontsize=9,
        )

    # Add vertical line at zero
    ax.axvline(x=0, color="black", linewidth=0.8)

    # Labels and title
    ax.set_xlabel("LIME Weight (contribution to prediction)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)

    title = (
        f"LIME Feature Contributions to '{predicted_class.upper()}' Prediction\n"
        f"Predicted: {prediction['label']} ({prediction['score']:.1%} confidence)"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=CONTRIBUTION_COLORS["positive_contribution"],
            label=f"Pushes toward {predicted_class}",
        ),
        Patch(
            facecolor=CONTRIBUTION_COLORS["negative_contribution"],
            label=f"Pushes away from {predicted_class}",
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


def plot_lime_summary_bar(
    summary_data: Dict,
    figsize: Tuple[int, int] = (14, 8),
    top_n: int = 15,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """
    Create a summary bar plot showing top features by LIME importance.

    Args:
        summary_data: Summary data from LIMEExplainer.get_summary_data()
        figsize: Figure size (width, height)
        top_n: Number of top features to display
        save_path: If provided, save figure to this path
        show: If True, display the figure

    Returns:
        Matplotlib Figure object
    """
    if not summary_data:
        raise ValueError("summary_data cannot be None or empty")

    top_features = summary_data["top_features"][:top_n]

    if not top_features:
        raise ValueError("No features found in summary data")

    # Prepare data
    features = [f[0] for f in reversed(top_features)]
    importances = [f[1] for f in reversed(top_features)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar chart
    bars = ax.barh(
        features,
        importances,
        color="#9b59b6",  # Purple for LIME
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
    ax.set_xlabel("Mean |LIME Weight| (average importance)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(
        f"LIME Summary: Top {len(features)} Most Important Features\n"
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


def plot_lime_class_comparison(
    summary_data: Dict,
    figsize: Tuple[int, int] = (16, 10),
    top_n: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """
    Create a comparison plot showing top LIME features for positive vs negative.

    Args:
        summary_data: Summary data from LIMEExplainer.get_summary_data()
        figsize: Figure size (width, height)
        top_n: Number of top features per class
        save_path: If provided, save figure to this path
        show: If True, display the figure

    Returns:
        Matplotlib Figure object
    """
    if not summary_data:
        raise ValueError("summary_data cannot be None or empty")

    top_positive = summary_data["top_positive_features"][:top_n]
    top_negative = summary_data["top_negative_features"][:top_n]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Positive sentiment features
    if top_positive:
        features_pos = [f[0] for f in reversed(top_positive)]
        values_pos = [f[1] for f in reversed(top_positive)]

        ax1.barh(
            features_pos,
            values_pos,
            color=SENTIMENT_COLORS["positive"],
            edgecolor="white",
        )
        ax1.set_xlabel("Mean LIME Weight", fontsize=11)
        ax1.set_ylabel("Feature", fontsize=11)
        ax1.set_title(
            "Top Features → POSITIVE Sentiment",
            fontsize=12,
            fontweight="bold",
            color=SENTIMENT_COLORS["positive"],
        )
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(axis="x", alpha=0.3)

    # Negative sentiment features
    if top_negative:
        features_neg = [f[0] for f in reversed(top_negative)]
        values_neg = [f[1] for f in reversed(top_negative)]

        ax2.barh(
            features_neg,
            values_neg,
            color=SENTIMENT_COLORS["negative"],
            edgecolor="white",
        )
        ax2.set_xlabel("Mean LIME Weight", fontsize=11)
        ax2.set_ylabel("Feature", fontsize=11)
        ax2.set_title(
            "Top Features → NEGATIVE Sentiment",
            fontsize=12,
            fontweight="bold",
            color=SENTIMENT_COLORS["negative"],
        )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"LIME Class Comparison (n={summary_data['num_explanations']} texts)",
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


def generate_lime_html(
    explanation: Dict,
    color_intensity: float = 1.0,
) -> str:
    """
    Generate HTML with colored features based on LIME weights.

    Args:
        explanation: Explanation dictionary from LIMEExplainer.explain()
        color_intensity: Multiplier for color intensity (0.0 - 2.0)

    Returns:
        HTML string with colored features
    """
    if not explanation:
        raise ValueError("explanation cannot be None or empty")

    text = explanation["text"]
    predicted_class = explanation["predicted_class"]
    feature_weights = explanation["feature_weights"]
    prediction = explanation["prediction"]

    # Split text into words
    words = text.split()

    # Normalize weights for color intensity
    if feature_weights:
        max_abs = max(abs(w) for w in feature_weights.values())
    else:
        max_abs = 1

    if max_abs == 0:
        max_abs = 1

    html_parts = []

    for word in words:
        word_lower = word.lower().strip()
        weight = feature_weights.get(word_lower, 0)

        # Calculate color intensity
        normalized = (weight / max_abs) * color_intensity
        normalized = max(-1, min(1, normalized))  # Clamp to [-1, 1]

        if abs(normalized) < 0.01:
            # Neutral/no contribution
            html_parts.append(f'<span style="padding: 2px 4px; margin: 1px;">{word}</span>')
        elif normalized > 0:
            # Green for positive contribution
            r, g, b = int(255 * (1 - normalized)), 255, int(255 * (1 - normalized))
        else:
            # Red for negative contribution
            r, g, b = 255, int(255 * (1 + normalized)), int(255 * (1 + normalized))

        if abs(normalized) >= 0.01:
            html_parts.append(
                f'<span style="background-color: rgb({r},{g},{b}); '
                f'padding: 2px 4px; margin: 1px; border-radius: 3px;" '
                f'title="LIME: {weight:+.4f}">{word}</span>'
            )

    # Build complete HTML
    html = f"""
    <div style="font-family: Arial, sans-serif; line-height: 2;">
        <h3>LIME Explanation</h3>
        <p><strong>Prediction:</strong> {prediction['label'].upper()}
           ({prediction['score']:.1%} confidence)</p>
        <p><strong>Explaining:</strong> {predicted_class} class</p>
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
            {' '.join(html_parts)}
        </div>
        <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
            <span style="background-color: rgb(200,255,200); padding: 2px 6px;">Green</span>
            = pushes toward {predicted_class} |
            <span style="background-color: rgb(255,200,200); padding: 2px 6px;">Red</span>
            = pushes away from {predicted_class}
        </p>
    </div>
    """

    return html


def save_lime_html(
    explanation: Dict,
    output_path: Union[str, Path],
    color_intensity: float = 1.0,
) -> Path:
    """
    Save LIME explanation as an HTML file with colored features.

    Args:
        explanation: Explanation dictionary from LIMEExplainer.explain()
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
        <title>LIME Explanation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
        </style>
    </head>
    <body>
        {generate_lime_html(explanation, color_intensity)}
    </body>
    </html>
    """

    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"HTML explanation saved to {output_path}")

    return output_path
