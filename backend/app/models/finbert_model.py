"""
FinBERT Sentiment Analysis Model

This module provides a wrapper for the ProsusAI/finbert model with automatic
GPU/CPU fallback and model caching.

Usage:
    from app.models.finbert_model import FinBERTModel
    
    model = FinBERTModel()
    sentiment = model.predict("The stock market is performing well today")
    # Returns: {'label': 'positive', 'score': 0.95, 'scores': {...}}
"""

import logging
from typing import Dict, List, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class FinBERTModel:
    """
    FinBERT sentiment analysis model wrapper with GPU/CPU fallback.
    
    The model classifies financial text into three categories:
    - positive: Optimistic/bullish sentiment
    - negative: Pessimistic/bearish sentiment
    - neutral: Objective/balanced sentiment
    """

    MODEL_NAME = "ProsusAI/finbert"
    LABELS = ["positive", "negative", "neutral"]

    def __init__(self, cache_dir: str = None):
        """
        Initialize the FinBERT model with automatic device selection.
        
        Args:
            cache_dir: Optional directory to cache the model. If None, uses
                      the default transformers cache directory.
        """
        self.cache_dir = cache_dir
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _get_device(self) -> str:
        """
        Determine the best available device (GPU/CPU) for inference.
        
        Returns:
            Device string: 'cuda' if GPU available, else 'cpu'
        """
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")
        
        return device

    def _load_model(self):
        """
        Load the FinBERT model and tokenizer from Hugging Face Hub.
        
        The model is cached locally after first download for faster subsequent loads.
        Implements GPU/CPU fallback automatically.
        """
        logger.info(f"Loading FinBERT model: {self.MODEL_NAME}")
        logger.info(f"Device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                cache_dir=self.cache_dir
            )
            logger.info("✓ Tokenizer loaded successfully")

            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                cache_dir=self.cache_dir
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info("✓ Model loaded successfully")
            logger.info(f"✓ Model cached at: {self.cache_dir or 'default cache'}")
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise

    def predict(
        self,
        text: Union[str, List[str]],
        return_all_scores: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment for given text(s).
        
        Args:
            text: Single text string or list of text strings
            return_all_scores: If True, return scores for all labels
            
        Returns:
            For single text: Dict with 'label', 'score', and optionally 'scores'
            For multiple texts: List of dicts
            
        Example:
            >>> model.predict("Stock prices are rising")
            {'label': 'positive', 'score': 0.92, 'scores': {'positive': 0.92, ...}}
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Handle single string input
        single_input = isinstance(text, str)
        if single_input:
            text = [text]
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            results = []
            for pred in predictions:
                scores_dict = {
                    label: float(score)
                    for label, score in zip(self.LABELS, pred)
                }
                
                # Get top prediction
                top_label_idx = torch.argmax(pred).item()
                top_label = self.LABELS[top_label_idx]
                top_score = float(pred[top_label_idx])
                
                result = {
                    "label": top_label,
                    "score": top_score
                }
                
                if return_all_scores:
                    result["scores"] = scores_dict
                
                results.append(result)
            
            # Return single result if single input
            return results[0] if single_input else results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_all_scores: bool = False
    ) -> List[Dict]:
        """
        Predict sentiment for a large batch of texts efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            return_all_scores: If True, return scores for all labels
            
        Returns:
            List of prediction dicts
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.predict(batch, return_all_scores=return_all_scores)
            results.extend(batch_results)
        
        return results

    def get_device_info(self) -> Dict[str, str]:
        """
        Get information about the device being used.
        
        Returns:
            Dict with device information
        """
        info = {
            "device": self.device,
            "device_name": torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "model_loaded": self.model is not None
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        
        return info


# Global model instance (singleton pattern)
_model_instance = None


def get_model(cache_dir: str = None) -> FinBERTModel:
    """
    Get or create the global FinBERT model instance.
    
    This implements a singleton pattern to ensure the model is only loaded once
    and cached for subsequent use.
    
    Args:
        cache_dir: Optional cache directory for model files
        
    Returns:
        FinBERTModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        logger.info("Initializing FinBERT model (first load)")
        _model_instance = FinBERTModel(cache_dir=cache_dir)
    
    return _model_instance
