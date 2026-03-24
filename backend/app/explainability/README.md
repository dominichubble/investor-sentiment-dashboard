# Explainability Module

This module provides Explainable AI (XAI) capabilities for the Investor Sentiment Dashboard using both SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).

## Overview

The explainability module helps users understand **why** FinBERT made a particular sentiment prediction by identifying which tokens/features contributed most to the classification.

### Methods

- **SHAP**: Global explanations based on game theory, showing token-level contributions
- **LIME**: Local explanations that approximate the model's behavior around specific predictions

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

### LIMEExplainer (`lime_explainer.py`)

Local explanation class that computes feature importance for FinBERT predictions.

```python
from app.explainability import LIMEExplainer, get_lime_explainer

# Using singleton pattern (recommended)
lime_explainer = get_lime_explainer(num_features=10, num_samples=1000)

# Single text explanation
explanation = lime_explainer.explain("Stock prices crashed after earnings miss")

# Batch explanation
texts = ["Market rally continues", "Recession fears grow"]
explanations = lime_explainer.explain_batch(texts)

# Get summary data for multiple explanations
summary = lime_explainer.get_summary_data(explanations)
```

### Visualizations (`visualizations.py`)

Plotting functions for both SHAP and LIME explanations.

#### SHAP Visualizations

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

#### LIME Visualizations

```python
from app.explainability import (
    plot_lime_features,
    plot_lime_summary_bar,
    plot_lime_class_comparison,
    save_lime_html,
)

# Plot single LIME explanation
fig = plot_lime_features(lime_explanation, save_path="lime_features.png")

# Plot summary across multiple LIME explanations
fig = plot_lime_summary_bar(lime_summary_data, save_path="lime_summary.png")

# Compare positive vs negative features
fig = plot_lime_class_comparison(lime_summary_data, save_path="lime_comparison.png")

# Export as interactive HTML
save_lime_html(lime_explanation, "lime_explanation.html")
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

## LIME Explanation Output Format

```python
{
    "text": "Stock prices crashed today",
    "prediction": {
        "label": "negative",
        "score": 0.94,
        "scores": {"positive": 0.02, "negative": 0.94, "neutral": 0.04}
    },
    "predicted_class": "negative",
    "feature_weights": {
        "crashed": 0.42,
        "prices": 0.08,
        "stock": 0.03,
        "today": 0.01
    },
    "all_class_weights": {
        "positive": {...},
        "negative": {...},
        "neutral": {...}
    },
    "top_features": [
        ("crashed", 0.42),
        ("prices", 0.08),
        ...
    ],
    "lime_explanation": <LimeExplanation object>,
    "local_prediction": 0.94
}
```

## Performance Considerations

### SHAP
- SHAP computation is CPU/GPU intensive (~1-5 seconds per text)
- More accurate but slower than LIME
- Better for understanding global model behavior

### LIME
- LIME is faster (~2-3 seconds per text with 1000 samples)
- Good for local explanations and quick analysis
- Adjustable `num_samples` parameter trades off speed vs accuracy

### Best Practices
- For batch processing, use `explain_batch()` methods
- Both explainers use singleton patterns to avoid reloading the model
- Consider caching explanations for frequently analyzed texts
- Use SHAP for comprehensive analysis, LIME for quick insights

## Choosing Between SHAP and LIME

| Criterion | SHAP | LIME |
|-----------|------|------|
| Speed | Slower | Faster |
| Accuracy | More accurate | Approximate |
| Consistency | Consistent | May vary slightly |
| Interpretability | Token-level | Feature-level |
| Use Case | Deep analysis | Quick insights |

## Dependencies

- `shap>=0.44.0`
- `lime>=0.2.0.1`
- `matplotlib>=3.8.0`
- `torch` and `transformers` (via FinBERT model)

## Example Usage: Comparing SHAP and LIME

```python
from app.explainability import get_explainer, get_lime_explainer
from app.explainability.visualizations import (
    plot_token_contributions,
    plot_lime_features,
)

text = "Stock prices surged on positive earnings"

# Get both explanations
shap_explainer = get_explainer()
lime_explainer = get_lime_explainer()

shap_exp = shap_explainer.explain(text)
lime_exp = lime_explainer.explain(text)

# Visualize both
plot_token_contributions(shap_exp, save_path="shap_analysis.png")
plot_lime_features(lime_exp, save_path="lime_analysis.png")
```
