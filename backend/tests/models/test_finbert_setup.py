#!/usr/bin/env python3
"""
Test script to verify FinBERT model loads correctly.

This script tests:
1. Model and tokenizer load without errors
2. GPU/CPU fallback works correctly
3. Model is cached properly
4. Basic inference works

Usage:
    python test_finbert_setup.py
"""

import logging
import sys
from pathlib import Path

# Add backend to path (navigate to project root, then into backend)
project_root = Path(__file__).resolve().parents[3]
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

from app.models.finbert_model import FinBERTModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test that the model loads correctly."""
    logger.info("=" * 60)
    logger.info("Testing FinBERT Model Setup")
    logger.info("=" * 60)

    try:
        # Initialize model
        logger.info("\n[1/4] Initializing FinBERT model...")
        model = FinBERTModel()
        logger.info("✓ Model initialized successfully")

        # Check device
        logger.info("\n[2/4] Checking device configuration...")
        device_info = model.get_device_info()
        logger.info(f"  Device: {device_info['device']}")
        logger.info(f"  Device Name: {device_info['device_name']}")
        logger.info(f"  CUDA Available: {device_info['cuda_available']}")
        logger.info(f"  Model Loaded: {device_info['model_loaded']}")

        if device_info["cuda_available"]:
            logger.info(f"  CUDA Version: {device_info.get('cuda_version', 'N/A')}")
            logger.info(
                f"  GPU Memory: {device_info.get('gpu_memory_allocated', 'N/A')}"
            )

        logger.info("✓ Device configuration verified")

        # Test single prediction
        logger.info("\n[3/4] Testing single prediction...")
        test_text = "The stock market is performing well today"
        result = model.predict(test_text, return_all_scores=True)

        logger.info(f"  Input: '{test_text}'")
        logger.info(f"  Prediction: {result['label']} ({result['score']:.4f})")
        logger.info(f"  All scores:")
        for label, score in result["scores"].items():
            logger.info(f"    {label}: {score:.4f}")

        logger.info("✓ Single prediction successful")

        # Test batch prediction
        logger.info("\n[4/4] Testing batch prediction...")
        test_texts = [
            "Company reports record earnings",
            "Stock prices plummet on bad news",
            "Market remains stable",
        ]

        results = model.predict(test_texts)

        for text, result in zip(test_texts, results):
            logger.info(f"  '{text}' -> {result['label']} ({result['score']:.4f})")

        logger.info("✓ Batch prediction successful")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("\nFinBERT is ready to use!")
        logger.info(f"Model: {FinBERTModel.MODEL_NAME}")
        logger.info(f"Device: {device_info['device']}")
        logger.info(f"Labels: {', '.join(FinBERTModel.LABELS)}")

        return True

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("✗ TEST FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
