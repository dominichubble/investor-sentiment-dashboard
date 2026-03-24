"""
LIME Explainability Module for FinBERT

This module provides LIME (Local Interpretable Model-agnostic Explanations)
integration for interpreting FinBERT sentiment predictions. LIME explains
individual predictions by learning an interpretable model locally around
the prediction.

Usage:
    from app.explainability import LIMEExplainer

    explainer = LIMEExplainer()
    explanation = explainer.explain("Stock prices surged today")
    explainer.plot_text_explanation(explanation)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from lime.lime_text import LimeTextExplainer
from numpy.typing import NDArray

from app.models.finbert_model import FinBERTModel, get_model

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    LIME-based explainer for FinBERT sentiment predictions.

    Uses LIME's text explainer to compute token-level importance scores
    for sentiment classifications by perturbing the input text and
    observing prediction changes.
    """

    LABELS = ["positive", "negative", "neutral"]
    CLASS_NAMES = ["positive", "negative", "neutral"]

    def __init__(
        self,
        num_features: int = 10,
        num_samples: int = 5000,
    ):
        """
        Initialize the LIME explainer.

        Args:
            num_features: Number of features/tokens to include in explanation.
            num_samples: Number of perturbed samples for LIME to generate.
                        Higher values give more accurate explanations but take longer.
        """
        self.num_features = num_features
        self.num_samples = num_samples
        self.model: Optional[FinBERTModel] = None
        self.explainer: Optional[LimeTextExplainer] = None
        self._initialize()

    def _initialize(self):
        """Initialize the FinBERT model and LIME explainer."""
        logger.info("Initializing LIME explainer for FinBERT")

        # Get the FinBERT model
        self.model = get_model()

        # Create a custom tokenizer function that uses word-level splitting
        # This works better with LIME's perturbation approach while still being
        # interpretable. FinBERT will handle subword tokenization internally.
        def tokenize_text(text: str) -> List[str]:
            """Tokenize text into words for LIME perturbations."""
            import re

            # Split on whitespace and punctuation, keeping words
            tokens = re.findall(r"\b\w+\b", text)
            return tokens if tokens else [text]

        # Create LIME text explainer with word-level tokenization
        # Using word-level splitting works better than subword-level for interpretability
        self.explainer = LimeTextExplainer(
            class_names=self.CLASS_NAMES,
            split_expression=r"\s+",  # Split on whitespace (keeps punctuation with words)
            bow=False,  # Don't use bag-of-words (position matters)
            random_state=42,
        )

        logger.info("✓ LIME explainer initialized successfully")

    def _predict_proba(self, texts: List[str]) -> NDArray[np.floating[Any]]:
        """
        Prediction function for LIME that returns class probabilities.

        Args:
            texts: List of text strings to analyze

        Returns:
            NumPy array of shape (n_texts, n_classes) with probabilities
        """
        if isinstance(texts, str):
            texts = [texts]

        # Handle empty texts from LIME perturbations
        processed_texts: List[str] = []
        for text in texts:
            if text is None or (isinstance(text, str) and not text.strip()):
                # Use a neutral placeholder for empty texts
                processed_texts.append("neutral")
            else:
                processed_texts.append(str(text))

        # Model must be initialized
        assert self.model is not None, "Model not initialized"

        try:
            # Use batch prediction for efficiency
            results = []
            batch_size = 32  # Process in batches

            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i : i + batch_size]
                batch_results = self.model.predict_batch(
                    batch, batch_size=len(batch), return_all_scores=True
                )

                for result in batch_results:
                    if isinstance(result, dict) and "scores" in result:
                        scores = result["scores"]
                        # Ensure order is [positive, negative, neutral]
                        probs = [
                            scores.get("positive", 0.33),
                            scores.get("negative", 0.33),
                            scores.get("neutral", 0.33),
                        ]
                        results.append(probs)
                    else:
                        # Fallback uniform distribution
                        results.append([0.33, 0.33, 0.33])

            return cast(NDArray[np.floating[Any]], np.array(results))

        except Exception as e:
            logger.error(f"Prediction failed during LIME computation: {e}")
            # Return uniform distribution on error
            return cast(
                NDArray[np.floating[Any]], np.ones((len(processed_texts), 3)) / 3
            )

    def explain(
        self,
        text: str,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None,
    ) -> Dict:
        """
        Generate LIME explanation for a single text.

        Args:
            text: Text to explain
            num_features: Number of features to include (overrides instance default)
            num_samples: Number of samples for LIME (overrides instance default)

        Returns:
            Dictionary containing:
                - text: Original input text
                - prediction: Predicted label and score
                - lime_explanation: LIME explanation object
                - feature_weights: Dict mapping features to their weights per class
                - top_features: Top contributing features for predicted class
                - local_prediction: LIME's local model prediction

        Raises:
            ValueError: If text is invalid
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")

        num_features = num_features or self.num_features
        num_samples = num_samples or self.num_samples

        logger.info(f"Generating LIME explanation for: '{text[:50]}...'")

        # Model and explainer must be initialized
        assert self.model is not None, "Model not initialized"
        assert self.explainer is not None, "Explainer not initialized"

        # Get prediction first
        prediction_result = self.model.predict(text, return_all_scores=True)
        prediction = cast(Dict[str, Any], prediction_result)

        # Get predicted class index
        predicted_label = str(prediction["label"])
        predicted_idx = self.LABELS.index(predicted_label)

        # Generate LIME explanation
        lime_exp = self.explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=(predicted_idx,),  # Explain predicted class
        )

        # Extract feature weights for the predicted class
        try:
            feature_weights = dict(lime_exp.as_list(label=predicted_idx))
        except (KeyError, IndexError) as e:
            logger.warning(
                f"Failed to extract feature weights for class {predicted_idx}: {e}"
            )
            feature_weights = {}

        # Get all features for all classes
        all_class_weights = {}
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            try:
                class_features = dict(lime_exp.as_list(label=class_idx))
                all_class_weights[class_name] = class_features
            except (KeyError, IndexError):
                # If class wasn't explained, use empty dict
                all_class_weights[class_name] = {}

        # Sort features by absolute weight
        sorted_features = sorted(
            feature_weights.items(), key=lambda x: float(abs(x[1])), reverse=True
        )

        # Warn if all weights are zero (indicates LIME couldn't find meaningful features)
        if feature_weights and all(abs(w) < 1e-6 for w in feature_weights.values()):
            logger.warning(
                f"All LIME feature weights are near zero. This may indicate:\n"
                f"  1. Model predictions are too confident (not enough variation)\n"
                f"  2. Tokenization mismatch between LIME and FinBERT\n"
                f"  3. Need more samples (current: {num_samples})\n"
                f"Consider increasing num_samples or checking model predictions."
            )

        # Get local prediction (LIME's interpretation)
        # When explaining a single class, local_pred is a 1D array with one element
        if lime_exp.local_pred is not None:
            # Since we only explained one class (predicted_idx), local_pred has shape (1,)
            local_pred = (
                float(lime_exp.local_pred[0]) if len(lime_exp.local_pred) > 0 else None
            )
        else:
            local_pred = None

        explanation = {
            "text": text,
            "prediction": prediction,
            "predicted_class": predicted_label,
            "predicted_class_idx": predicted_idx,
            "lime_explanation": lime_exp,
            "feature_weights": feature_weights,
            "all_class_weights": all_class_weights,
            "top_features": sorted_features[:num_features],
            "local_prediction": local_pred,
            "num_features": num_features,
            "num_samples": num_samples,
        }

        # Log completion with top feature if available
        if sorted_features:
            logger.info(
                f"Explanation complete. Top feature: '{sorted_features[0][0]}' "
                f"({sorted_features[0][1]:+.4f})"
            )
        else:
            logger.warning("Explanation complete but no features found")

        return explanation

    def explain_batch(
        self,
        texts: List[str],
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate LIME explanations for multiple texts.

        Args:
            texts: List of texts to explain
            num_features: Number of features to include
            num_samples: Number of samples for LIME

        Returns:
            List of explanation dictionaries
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        logger.info(f"Generating LIME explanations for {len(texts)} texts")

        explanations = []
        for i, text in enumerate(texts):
            try:
                explanation = self.explain(text, num_features, num_samples)
                explanations.append(explanation)
                logger.info(f"✓ Explained {i+1}/{len(texts)}")
            except Exception as e:
                logger.warning(f"Failed to explain text {i}: {e}")
                explanations.append(None)

        successful = sum(1 for e in explanations if e is not None)
        logger.info(f"Completed {successful}/{len(texts)} explanations")

        return explanations

    def get_summary_data(
        self,
        explanations: List[Dict],
        top_n: int = 20,
    ) -> Dict:
        """
        Aggregate LIME feature weights across multiple explanations.

        Args:
            explanations: List of explanation dictionaries from explain()
            top_n: Number of top features to include in summary

        Returns:
            Dictionary with aggregated data for summary plots:
                - feature_importance: Dict mapping features to average absolute weights
                - class_feature_importance: Per-class feature importance
                - top_positive_features: Features most associated with positive sentiment
                - top_negative_features: Features most associated with negative sentiment
        """
        if not explanations:
            raise ValueError("explanations list cannot be empty")

        # Filter out failed explanations
        valid_explanations = [e for e in explanations if e is not None]

        if not valid_explanations:
            raise ValueError("No valid explanations to summarize")

        # Aggregate feature importance across all explanations
        feature_importance: Dict[str, List[float]] = {}
        class_feature_importance: Dict[str, Dict[str, List[float]]] = {
            label: {} for label in self.LABELS
        }

        for explanation in valid_explanations:
            all_class_weights = explanation["all_class_weights"]

            # Aggregate per-class feature weights
            for class_name in self.LABELS:
                if class_name not in all_class_weights:
                    continue

                class_weights = all_class_weights[class_name]
                for feature, weight in class_weights.items():
                    feature_lower = feature.lower().strip()

                    # Overall importance (absolute value)
                    if feature_lower not in feature_importance:
                        feature_importance[feature_lower] = []
                    feature_importance[feature_lower].append(abs(weight))

                    # Per-class importance
                    if feature_lower not in class_feature_importance[class_name]:
                        class_feature_importance[class_name][feature_lower] = []
                    class_feature_importance[class_name][feature_lower].append(weight)

        # Calculate average importance for each feature
        avg_importance = {
            feature: float(np.mean(values))
            for feature, values in feature_importance.items()
        }

        # Calculate per-class average
        class_avg_importance = {}
        for label in self.LABELS:
            class_avg_importance[label] = {
                feature: float(np.mean(values))
                for feature, values in class_feature_importance[label].items()
            }

        # Get top features by importance
        sorted_features = sorted(
            avg_importance.items(), key=lambda x: float(x[1]), reverse=True
        )
        top_features = sorted_features[:top_n]

        # Get features most associated with each class
        top_positive = sorted(
            class_avg_importance["positive"].items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        top_negative = sorted(
            class_avg_importance["negative"].items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        return {
            "feature_importance": avg_importance,
            "top_features": top_features,
            "class_feature_importance": class_avg_importance,
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "num_explanations": len(valid_explanations),
        }


# Global explainer instance (singleton pattern)
_explainer_instance = None


def get_lime_explainer(
    num_features: int = 10, num_samples: int = 5000
) -> LIMEExplainer:
    """
    Get or create the global LIME explainer instance.

    Args:
        num_features: Number of features to include in explanations
        num_samples: Number of samples for LIME

    Returns:
        LIMEExplainer instance
    """
    global _explainer_instance

    if _explainer_instance is None:
        logger.info("Initializing LIME explainer (first load)")
        _explainer_instance = LIMEExplainer(
            num_features=num_features, num_samples=num_samples
        )

    return _explainer_instance
