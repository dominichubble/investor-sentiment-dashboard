"""
Explainability Package for FinBERT Sentiment Analysis

This package provides Explainable AI (XAI) tools for interpreting
FinBERT sentiment predictions using SHAP and LIME.

Modules:
    - shap_explainer: SHAP-based token importance analysis
    - visualizations: Plotting functions for SHAP explanations
    - lime_explainer: LIME-based local explanations (coming soon)

Usage:
    from app.explainability import SHAPExplainer, get_explainer
    from app.explainability.visualizations import (
        plot_token_contributions,
        plot_summary_bar,
        plot_class_comparison,
    )

    # Quick start
    explainer = get_explainer()
    explanation = explainer.explain("Stock prices surged today")
    plot_token_contributions(explanation)
"""

from app.explainability.shap_explainer import SHAPExplainer, get_explainer
from app.explainability.visualizations import (
    generate_text_html,
    plot_class_comparison,
    plot_summary_bar,
    plot_token_contributions,
    save_explanation_html,
)

__all__ = [
    # Core explainer
    "SHAPExplainer",
    "get_explainer",
    # Visualization functions
    "plot_token_contributions",
    "plot_summary_bar",
    "plot_class_comparison",
    "generate_text_html",
    "save_explanation_html",
]
