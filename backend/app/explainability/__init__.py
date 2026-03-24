"""
Explainability Package for FinBERT Sentiment Analysis

This package provides Explainable AI (XAI) tools for interpreting
FinBERT sentiment predictions using SHAP and LIME.

Modules:
    - shap_explainer: SHAP-based token importance analysis
    - lime_explainer: LIME-based local explanations
    - visualizations: Plotting functions for SHAP and LIME explanations

Usage:
    # SHAP
    from app.explainability import SHAPExplainer, get_explainer
    from app.explainability.visualizations import (
        plot_token_contributions,
        plot_summary_bar,
        plot_class_comparison,
    )

    explainer = get_explainer()
    explanation = explainer.explain("Stock prices surged today")
    plot_token_contributions(explanation)

    # LIME
    from app.explainability import LIMEExplainer, get_lime_explainer
    from app.explainability.visualizations import (
        plot_lime_features,
        plot_lime_summary_bar,
        plot_lime_class_comparison,
    )

    lime_explainer = get_lime_explainer()
    lime_explanation = lime_explainer.explain("Stock prices surged today")
    plot_lime_features(lime_explanation)
"""

from app.explainability.lime_explainer import LIMEExplainer, get_lime_explainer
from app.explainability.shap_explainer import SHAPExplainer, get_explainer
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

__all__ = [
    # SHAP explainer
    "SHAPExplainer",
    "get_explainer",
    # LIME explainer
    "LIMEExplainer",
    "get_lime_explainer",
    # SHAP visualization functions
    "plot_token_contributions",
    "plot_summary_bar",
    "plot_class_comparison",
    "generate_text_html",
    "save_explanation_html",
    # LIME visualization functions
    "plot_lime_features",
    "plot_lime_summary_bar",
    "plot_lime_class_comparison",
    "generate_lime_html",
    "save_lime_html",
]
