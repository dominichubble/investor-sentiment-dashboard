#!/usr/bin/env python3
"""
Unit tests for FinBERT sentiment analysis module.

Tests cover:
- Model initialization
- GPU/CPU fallback
- Single and batch predictions
- Error handling
- Model caching
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from app.models.finbert import (
    DEFAULT_MODEL_NAME,
    FinBERTSentiment,
    initialize_finbert,
)


class TestFinBERTSentiment:
    """Test suite for FinBERTSentiment class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_device_detection_auto(self):
        """Test automatic device detection."""
        finbert = FinBERTSentiment.__new__(FinBERTSentiment)
        
        # Test with CUDA available
        with patch("torch.cuda.is_available", return_value=True):
            device = finbert._get_device(None)
            assert device == "cuda"
        
        # Test with CUDA unavailable
        with patch("torch.cuda.is_available", return_value=False):
            device = finbert._get_device(None)
            assert device == "cpu"
    
    def test_device_selection_explicit(self):
        """Test explicit device selection."""
        finbert = FinBERTSentiment.__new__(FinBERTSentiment)
        
        # Explicit CPU
        device = finbert._get_device("cpu")
        assert device == "cpu"
        
        # Explicit CUDA (with fallback if not available)
        with patch("torch.cuda.is_available", return_value=False):
            device = finbert._get_device("cuda")
            assert device == "cpu"  # Should fallback
    
    def test_cache_directory_creation(self, temp_cache_dir):
        """Test cache directory is created if it doesn't exist."""
        cache_path = temp_cache_dir / "test_cache"
        assert not cache_path.exists()
        
        # Initialize with custom cache dir
        # Note: This will actually load the model - use with caution
        # In a real test, you'd mock the model loading
        with patch.object(FinBERTSentiment, "_load_model"):
            finbert = FinBERTSentiment(cache_dir=cache_path)
            assert cache_path.exists()
    
    @pytest.mark.slow
    def test_model_initialization(self):
        """Test model loads without errors (slow test - loads real model)."""
        finbert = FinBERTSentiment()
        
        assert finbert.model is not None
        assert finbert.tokenizer is not None
        assert finbert.pipeline is not None
        assert finbert.device in ["cuda", "cpu"]
    
    @pytest.mark.slow
    def test_single_prediction(self):
        """Test single text prediction."""
        finbert = FinBERTSentiment()
        
        # Positive sentiment
        result = finbert.predict("Stock prices soared to new highs today.")
        assert isinstance(result, dict)
        assert "label" in result
        assert "score" in result
        assert result["label"] in ["positive", "negative", "neutral"]
        assert 0 <= result["score"] <= 1
    
    @pytest.mark.slow
    def test_batch_prediction(self):
        """Test batch prediction."""
        finbert = FinBERTSentiment()
        
        texts = [
            "Earnings beat expectations.",
            "Market crashed today.",
            "Stock prices remained stable.",
        ]
        
        results = finbert.predict_batch(texts)
        assert len(results) == len(texts)
        
        for result in results:
            assert isinstance(result, dict)
            assert "label" in result
            assert "score" in result
            assert result["label"] in ["positive", "negative", "neutral"]
    
    def test_empty_text_prediction(self):
        """Test prediction with empty text."""
        with patch.object(FinBERTSentiment, "_load_model"):
            finbert = FinBERTSentiment()
            finbert.pipeline = MagicMock()
            
            result = finbert.predict("")
            assert result["label"] == "neutral"
            assert result["score"] == 0.0
    
    def test_empty_batch_prediction(self):
        """Test batch prediction with empty list."""
        with patch.object(FinBERTSentiment, "_load_model"):
            finbert = FinBERTSentiment()
            finbert.pipeline = MagicMock()
            
            results = finbert.predict_batch([])
            assert results == []
    
    def test_batch_with_empty_texts(self):
        """Test batch prediction filters out empty texts."""
        with patch.object(FinBERTSentiment, "_load_model"):
            finbert = FinBERTSentiment()
            
            # Mock pipeline to return results for valid texts only
            finbert.pipeline = MagicMock(
                return_value=[
                    {"label": "positive", "score": 0.9},
                    {"label": "negative", "score": 0.8},
                ]
            )
            
            texts = ["Valid text", "", "Another valid text", None]
            results = finbert.predict_batch(texts)
            
            assert len(results) == 2
            assert results[0]["label"] == "positive"
            assert results[1]["label"] == "negative"
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        with patch.object(FinBERTSentiment, "_load_model"):
            finbert = FinBERTSentiment()
            
            info = finbert.get_model_info()
            assert isinstance(info, dict)
            assert "model_name" in info
            assert "device" in info
            assert "cuda_available" in info
            assert "cache_dir" in info
            assert "max_length" in info
    
    @pytest.mark.slow
    def test_warm_up(self):
        """Test model warm-up."""
        finbert = FinBERTSentiment()
        
        # Should not raise any errors
        finbert.warm_up()
        finbert.warm_up("Custom warm-up text")
    
    def test_repr(self):
        """Test string representation."""
        with patch.object(FinBERTSentiment, "_load_model"):
            finbert = FinBERTSentiment()
            
            repr_str = repr(finbert)
            assert "FinBERTSentiment" in repr_str
            assert finbert.model_name in repr_str
            assert finbert.device in repr_str
    
    @pytest.mark.slow
    def test_initialize_finbert_function(self):
        """Test convenience initialization function."""
        finbert = initialize_finbert()
        
        assert isinstance(finbert, FinBERTSentiment)
        assert finbert.model is not None
    
    def test_prediction_error_handling(self):
        """Test prediction handles errors gracefully."""
        with patch.object(FinBERTSentiment, "_load_model"):
            finbert = FinBERTSentiment()
            
            # Mock pipeline to raise exception
            finbert.pipeline = MagicMock(side_effect=Exception("Test error"))
            
            result = finbert.predict("Test text")
            assert result["label"] == "neutral"
            assert result["score"] == 0.0
    
    def test_batch_prediction_error_handling(self):
        """Test batch prediction handles errors gracefully."""
        with patch.object(FinBERTSentiment, "_load_model"):
            finbert = FinBERTSentiment()
            
            # Mock pipeline to raise exception
            finbert.pipeline = MagicMock(side_effect=Exception("Test error"))
            
            results = finbert.predict_batch(["Text 1", "Text 2"])
            assert len(results) == 2
            assert all(r["label"] == "neutral" for r in results)


class TestModelLoading:
    """Test suite for model loading scenarios."""
    
    def test_model_loading_failure(self):
        """Test model loading failure handling."""
        with patch(
            "app.models.finbert.AutoTokenizer.from_pretrained",
            side_effect=Exception("Network error"),
        ):
            with pytest.raises(RuntimeError, match="Model loading failed"):
                FinBERTSentiment()
    
    @pytest.mark.slow
    def test_cpu_fallback_on_gpu_failure(self):
        """Test CPU fallback when GPU loading fails."""
        # This test would need a more complex setup to simulate GPU failure
        # while having CUDA available - skipping detailed implementation
        pass


# Configuration for pytest
pytest_plugins = []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
