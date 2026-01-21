"""
SHAP Explainability Module for FinBERT

This module provides SHAP (SHapley Additive exPlanations) integration for
interpreting FinBERT sentiment predictions. It identifies which tokens
contributed most to positive, negative, or neutral classifications.

Usage:
    from app.explainability import SHAPExplainer

    explainer = SHAPExplainer()
    explanation = explainer.explain("Stock prices surged today")
    explainer.plot_text_explanation(explanation)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import shap
import torch
from numpy.typing import NDArray

from app.models.finbert_model import FinBERTModel, get_model

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainer for FinBERT sentiment predictions.

    Uses SHAP's Partition explainer optimized for transformer models to compute
    token-level importance scores for each sentiment class.
    """

    LABELS = ["positive", "negative", "neutral"]

    def __init__(self, max_evals: int = 500):
        """
        Initialize the SHAP explainer.

        Args:
            max_evals: Maximum number of model evaluations for SHAP.
                      Higher values give more accurate explanations but take longer.
        """
        self.max_evals = max_evals
        self.model: Optional[FinBERTModel] = None
        self.explainer: Optional[shap.Explainer] = None
        self._initialize()

    def _initialize(self):
        """Initialize the FinBERT model and SHAP explainer."""
        logger.info("Initializing SHAP explainer for FinBERT")

        # Get the FinBERT model
        self.model = get_model()

        # Create prediction function for SHAP
        self.explainer = shap.Explainer(
            self._predict_proba,
            self.model.tokenizer,
            output_names=self.LABELS,
        )

        logger.info("✓ SHAP explainer initialized successfully")

    def _predict_proba(self, texts: List[str]) -> NDArray[np.floating[Any]]:
        """
        Prediction function for SHAP that returns class probabilities.

        Args:
            texts: List of text strings to analyze

        Returns:
            NumPy array of shape (n_texts, n_classes) with probabilities
        """
        if isinstance(texts, str):
            texts = [texts]

        # Handle masked/empty texts from SHAP perturbations
        processed_texts: List[str] = []
        for text in texts:
            if text is None or (isinstance(text, str) and not text.strip()):
                processed_texts.append("[PAD]")
            else:
                processed_texts.append(str(text))

        # Model must be initialized
        assert self.model is not None, "Model not initialized"
        assert self.model.tokenizer is not None, "Tokenizer not loaded"
        assert self.model.model is not None, "Model not loaded"

        # Local references for type checker
        tokenizer = self.model.tokenizer
        model_device = self.model.device
        model_inner = self.model.model

        try:
            # Tokenize
            inputs = tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = model_inner(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            return cast(NDArray[np.floating[Any]], probs.cpu().numpy())

        except Exception as e:
            logger.error(f"Prediction failed during SHAP computation: {e}")
            # Return uniform distribution on error
            return cast(NDArray[np.floating[Any]], np.ones((len(processed_texts), 3)) / 3)

    def explain(
        self,
        text: str,
        target_class: Optional[str] = None,
    ) -> Dict:
        """
        Generate SHAP explanation for a single text.

        Args:
            text: Text to explain
            target_class: Specific class to explain ('positive', 'negative', 'neutral').
                         If None, explains the predicted class.

        Returns:
            Dictionary containing:
                - text: Original input text
                - prediction: Predicted label and score
                - tokens: List of tokens
                - shap_values: Token-level SHAP values for each class
                - target_class: The class being explained
                - token_contributions: List of (token, contribution) tuples for target class

        Raises:
            ValueError: If target_class is invalid
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")

        if target_class and target_class not in self.LABELS:
            raise ValueError(f"target_class must be one of {self.LABELS}")

        logger.info(f"Generating SHAP explanation for: '{text[:50]}...'")

        # Model and explainer must be initialized
        assert self.model is not None, "Model not initialized"
        assert self.explainer is not None, "Explainer not initialized"

        # Get prediction first
        prediction_result = self.model.predict(text, return_all_scores=True)
        prediction = cast(Dict[str, Any], prediction_result)

        # Determine target class
        if target_class is None:
            target_class = str(prediction["label"])

        target_idx = self.LABELS.index(target_class)

        # Compute SHAP values
        shap_values = self.explainer([text], fixed_context=1)

        # Extract tokens and values
        tokens = shap_values.data[0]
        values = shap_values.values[0]

        # Get contributions for target class
        target_contributions = values[:, target_idx]
        token_contributions = [
            (str(token), float(contrib))
            for token, contrib in zip(tokens, target_contributions)
        ]

        # Sort by absolute contribution
        sorted_contributions = sorted(
            token_contributions, key=lambda x: abs(x[1]), reverse=True
        )

        explanation = {
            "text": text,
            "prediction": prediction,
            "tokens": [str(t) for t in tokens],
            "shap_values": {
                label: values[:, i].tolist() for i, label in enumerate(self.LABELS)
            },
            "target_class": target_class,
            "token_contributions": token_contributions,
            "top_contributors": sorted_contributions[:10],
            "base_value": float(shap_values.base_values[0][target_idx]),
        }

        logger.info(
            f"Explanation complete. Top contributor: '{sorted_contributions[0][0]}' "
            f"({sorted_contributions[0][1]:+.4f})"
        )

        return explanation

    def explain_batch(
        self,
        texts: List[str],
        target_class: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate SHAP explanations for multiple texts.

        Args:
            texts: List of texts to explain
            target_class: Specific class to explain (applies to all texts)

        Returns:
            List of explanation dictionaries
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        logger.info(f"Generating SHAP explanations for {len(texts)} texts")

        explanations = []
        for i, text in enumerate(texts):
            try:
                explanation = self.explain(text, target_class)
                explanations.append(explanation)
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
        Aggregate SHAP values across multiple explanations for summary visualization.

        Args:
            explanations: List of explanation dictionaries from explain() or explain_batch()
            top_n: Number of top tokens to include in summary

        Returns:
            Dictionary with aggregated data for summary plots:
                - token_importance: Dict mapping tokens to average absolute SHAP values
                - class_token_importance: Per-class token importance
                - top_positive_tokens: Tokens most associated with positive sentiment
                - top_negative_tokens: Tokens most associated with negative sentiment
        """
        if not explanations:
            raise ValueError("explanations list cannot be empty")

        # Filter out failed explanations
        valid_explanations = [e for e in explanations if e is not None]

        if not valid_explanations:
            raise ValueError("No valid explanations to summarize")

        # Aggregate token importance across all explanations
        token_importance: Dict[str, List[float]] = {}
        class_token_importance: Dict[str, Dict[str, List[float]]] = {
            label: {} for label in self.LABELS
        }

        for explanation in valid_explanations:
            tokens = explanation["tokens"]
            shap_values = explanation["shap_values"]

            for i, token in enumerate(tokens):
                token = token.lower().strip()
                if not token or token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue

                # Overall importance (mean absolute across all classes)
                mean_abs = float(np.mean(
                    [abs(shap_values[label][i]) for label in self.LABELS]
                ))

                if token not in token_importance:
                    token_importance[token] = []
                token_importance[token].append(mean_abs)

                # Per-class importance
                for label in self.LABELS:
                    if token not in class_token_importance[label]:
                        class_token_importance[label][token] = []
                    class_token_importance[label][token].append(shap_values[label][i])

        # Calculate average importance for each token
        avg_importance = {
            token: np.mean(values) for token, values in token_importance.items()
        }

        # Calculate per-class average
        class_avg_importance = {}
        for label in self.LABELS:
            class_avg_importance[label] = {
                token: np.mean(values)
                for token, values in class_token_importance[label].items()
            }

        # Get top tokens by importance
        sorted_tokens = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[:top_n]

        # Get tokens most associated with each class
        top_positive = sorted(
            class_avg_importance["positive"].items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        top_negative = sorted(
            class_avg_importance["negative"].items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        return {
            "token_importance": avg_importance,
            "top_tokens": top_tokens,
            "class_token_importance": class_avg_importance,
            "top_positive_tokens": top_positive,
            "top_negative_tokens": top_negative,
            "num_explanations": len(valid_explanations),
        }


# Global explainer instance (singleton pattern)
_explainer_instance = None


def get_explainer(max_evals: int = 500) -> SHAPExplainer:
    """
    Get or create the global SHAP explainer instance.

    Args:
        max_evals: Maximum number of model evaluations for SHAP

    Returns:
        SHAPExplainer instance
    """
    global _explainer_instance

    if _explainer_instance is None:
        logger.info("Initializing SHAP explainer (first load)")
        _explainer_instance = SHAPExplainer(max_evals=max_evals)

    return _explainer_instance
