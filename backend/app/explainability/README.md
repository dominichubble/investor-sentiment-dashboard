# Explainability Module

This module provides Explainable AI (XAI) capabilities for the Investor Sentiment Dashboard using SHAP (SHapley Additive exPlanations).

## Overview

The explainability module helps users understand **why** FinBERT made a particular sentiment prediction by identifying which tokens (words) contributed most to the classification.

## Components

### SHAPExplainer (`shap_explainer.py`)

Core explainer class that computes SHAP values for FinBERT predictions.

```python
from app.explainability import SHAPExplainer, get_explainer

# Using singleton pattern (recommended)
explainer = get_explainer()

# Single text explanation
explanation = explainer.explain("Stock prices crashed after earnings miss")

# Batch explanation
texts = ["Market rally continues", "Recession fears grow"]
explanations = explainer.explain_batch(texts)

# Get summary data for multiple explanations
summary = explainer.get_summary_data(explanations)
```

### Visualizations (`visualizations.py`)

Plotting functions for SHAP explanations.

```python
from app.explainability import (
    plot_token_contributions,
    plot_summary_bar,
    plot_class_comparison,
    save_explanation_html,
)

# Plot single explanation
fig = plot_token_contributions(explanation, save_path="token_contrib.png")

# Plot summary across multiple explanations
fig = plot_summary_bar(summary_data, save_path="summary.png")

# Compare positive vs negative tokens
fig = plot_class_comparison(summary_data, save_path="comparison.png")

# Export as interactive HTML
save_explanation_html(explanation, "explanation.html")
```

## Explanation Output Format

```python
{
    "text": "Stock prices crashed today",
    "prediction": {
        "label": "negative",
        "score": 0.94,
        "scores": {"positive": 0.02, "negative": 0.94, "neutral": 0.04}
    },
    "tokens": ["Stock", "prices", "crashed", "today"],
    "shap_values": {
        "positive": [-0.05, -0.02, -0.42, -0.01],
        "negative": [0.03, 0.05, 0.45, 0.02],
        "neutral": [0.02, -0.03, -0.03, -0.01]
    },
    "target_class": "negative",
    "token_contributions": [
        ("Stock", 0.03),
        ("prices", 0.05),
        ("crashed", 0.45),
        ("today", 0.02)
    ],
    "top_contributors": [
        ("crashed", 0.45),
        ("prices", 0.05),
        ...
    ],
    "base_value": 0.33
}
```

## Performance Considerations

- SHAP computation is CPU/GPU intensive (~1-5 seconds per text)
- For batch processing, consider using `explain_batch()` 
- The explainer uses a singleton pattern to avoid reloading the model
- Consider caching explanations for frequently analyzed texts

## Dependencies

- `shap>=0.44.0`
- `matplotlib>=3.8.0`
- `torch` and `transformers` (via FinBERT model)
