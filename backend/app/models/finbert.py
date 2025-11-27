#!/usr/bin/env python3
"""
FinBERT Sentiment Analysis Module

This module provides a wrapper for the ProsusAI/finbert model for
financial sentiment analysis with GPU/CPU fallback and caching.

Usage:
    from app.models.finbert import FinBERTSentiment
    
    # Initialize model (auto-downloads and caches)
    finbert = FinBERTSentiment()
    
    # Analyze sentiment
    result = finbert.predict("Stock prices rallied after strong earnings.")
    print(result)  # {'label': 'positive', 'score': 0.95}
    
    # Batch prediction
    texts = ["Market crashed today", "Strong earnings beat expectations"]
    results = finbert.predict_batch(texts)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "finbert"
MAX_LENGTH = 512  # FinBERT's maximum sequence length


class FinBERTSentiment:
    """
    FinBERT sentiment analysis wrapper with GPU/CPU fallback and caching.
    
    This class handles:
    - Automatic model download and caching
    - GPU/CPU device selection with fallback
    - Single and batch prediction
    - Proper resource management
    
    Attributes:
        model_name: Hugging Face model identifier
        device: Current device (cuda/cpu)
        model: Loaded FinBERT model
        tokenizer: Loaded tokenizer
        pipeline: Hugging Face sentiment analysis pipeline
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize FinBERT model with GPU/CPU fallback.
        
        Args:
            model_name: Hugging Face model identifier (default: ProsusAI/finbert)
            cache_dir: Directory to cache model files (default: ~/.cache/finbert)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        
        Raises:
            RuntimeError: If model fails to load
            OSError: If cache directory cannot be created
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model cache directory: {self.cache_dir}")
        
        # Determine device (GPU/CPU fallback)
        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        self._load_model()
        logger.info("✓ FinBERT model loaded successfully")
    
    def _get_device(self, device: Optional[str] = None) -> str:
        """
        Determine the best available device with GPU/CPU fallback.
        
        Args:
            device: Preferred device ('cuda', 'cpu', or None for auto)
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if device:
            # User specified device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            return device
        
        # Auto-detect device
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
            return "cuda"
        else:
            logger.info("No GPU detected, using CPU")
            return "cpu"
    
    def _load_model(self):
        """
        Load FinBERT model, tokenizer, and create pipeline.
        
        Downloads and caches the model if not already present.
        Implements automatic fallback to CPU if GPU fails.
        
        Raises:
            RuntimeError: If model loading fails on both GPU and CPU
        """
        try:
            logger.info(f"Loading FinBERT model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
            )
            logger.info("✓ Tokenizer loaded")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
            )
            
            # Move model to device
            try:
                self.model = self.model.to(self.device)
                logger.info(f"✓ Model loaded on {self.device}")
            except RuntimeError as e:
                if self.device == "cuda":
                    logger.warning(f"Failed to load on GPU: {e}")
                    logger.info("Falling back to CPU...")
                    self.device = "cpu"
                    self.model = self.model.to(self.device)
                    logger.info("✓ Model loaded on CPU (fallback)")
                else:
                    raise
            
            # Create sentiment analysis pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_length=MAX_LENGTH,
                truncation=True,
            )
            logger.info("✓ Pipeline created")
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary with 'label' and 'score' keys
            - label: 'positive', 'negative', or 'neutral'
            - score: Confidence score (0-1)
        
        Example:
            >>> finbert = FinBERTSentiment()
            >>> result = finbert.predict("Stock prices soared today")
            >>> print(result)
            {'label': 'positive', 'score': 0.95}
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for prediction")
            return {"label": "neutral", "score": 0.0}
        
        try:
            result = self.pipeline(text)[0]
            return {
                "label": result["label"].lower(),
                "score": float(result["score"]),
            }
        except Exception as e:
            logger.error(f"Prediction failed for text: {text[:50]}... Error: {e}")
            return {"label": "neutral", "score": 0.0}
    
    def predict_batch(
        self, texts: List[str], batch_size: int = 8
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sentiment for multiple texts efficiently.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
        
        Returns:
            List of dictionaries with 'label' and 'score' keys
        
        Example:
            >>> finbert = FinBERTSentiment()
            >>> texts = ["Stocks up", "Market crashed", "Earnings beat expectations"]
            >>> results = finbert.predict_batch(texts)
            >>> print(results)
            [
                {'label': 'positive', 'score': 0.89},
                {'label': 'negative', 'score': 0.92},
                {'label': 'positive', 'score': 0.94}
            ]
        """
        if not texts:
            logger.warning("Empty text list provided for batch prediction")
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) < len(texts):
            logger.warning(
                f"Filtered out {len(texts) - len(valid_texts)} empty texts"
            )
        
        if not valid_texts:
            return [{"label": "neutral", "score": 0.0} for _ in texts]
        
        try:
            results = self.pipeline(valid_texts, batch_size=batch_size)
            return [
                {"label": r["label"].lower(), "score": float(r["score"])}
                for r in results
            ]
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [{"label": "neutral", "score": 0.0} for _ in texts]
    
    def get_model_info(self) -> Dict[str, Union[str, int, bool]]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata:
            - model_name: Model identifier
            - device: Current device (cuda/cpu)
            - cuda_available: Whether CUDA is available
            - cache_dir: Model cache directory
            - max_length: Maximum sequence length
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cache_dir": str(self.cache_dir),
            "max_length": MAX_LENGTH,
        }
    
    def warm_up(self, sample_text: str = "The market is performing well."):
        """
        Warm up the model by running a sample prediction.
        
        This helps ensure the model is fully loaded and ready, especially
        useful when using GPU to initialize CUDA kernels.
        
        Args:
            sample_text: Text to use for warm-up (default: generic positive text)
        """
        logger.info("Warming up model...")
        _ = self.predict(sample_text)
        logger.info("✓ Model warm-up complete")
    
    def __repr__(self) -> str:
        """String representation of the model instance."""
        return (
            f"FinBERTSentiment("
            f"model='{self.model_name}', "
            f"device='{self.device}'"
            f")"
        )


def initialize_finbert(
    cache_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> FinBERTSentiment:
    """
    Initialize and cache FinBERT model (convenience function).
    
    This function can be called at startup to download and cache the model
    before the first prediction request.
    
    Args:
        cache_dir: Directory to cache model files
        device: Device to use ('cuda', 'cpu', or None for auto)
    
    Returns:
        Initialized FinBERTSentiment instance
    
    Example:
        >>> finbert = initialize_finbert()
        >>> result = finbert.predict("Strong quarterly earnings")
    """
    logger.info("Initializing FinBERT model...")
    finbert = FinBERTSentiment(cache_dir=cache_dir, device=device)
    finbert.warm_up()
    logger.info("✓ FinBERT initialization complete")
    return finbert


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("FinBERT Sentiment Analysis Demo")
    print("=" * 60)
    
    # Initialize model
    finbert = initialize_finbert()
    
    # Show model info
    print("\nModel Info:")
    info = finbert.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Single prediction
    print("\n" + "=" * 60)
    print("Single Prediction:")
    print("=" * 60)
    test_text = "The stock market rallied today with tech stocks leading gains."
    result = finbert.predict(test_text)
    print(f"\nText: {test_text}")
    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['score']:.2%}")
    
    # Batch prediction
    print("\n" + "=" * 60)
    print("Batch Prediction:")
    print("=" * 60)
    test_texts = [
        "Earnings beat expectations significantly.",
        "Market crashed due to recession fears.",
        "Stock prices remained stable throughout the day.",
    ]
    
    results = finbert.predict_batch(test_texts)
    for text, result in zip(test_texts, results):
        print(f"\nText: {text}")
        print(f"  → {result['label']} ({result['score']:.2%})")
