#!/usr/bin/env python3
"""
FinBERT Model Initialization Script

This script downloads and caches the FinBERT model at startup.
Run this script to pre-download the model before first use.

Usage:
    python -m app.models.init_finbert
    python -m app.models.init_finbert --device cuda
    python -m app.models.init_finbert --cache-dir /path/to/cache
"""

import argparse
import logging
import sys
from pathlib import Path

from app.models.finbert import initialize_finbert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize and cache FinBERT model"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache model files (default: ~/.cache/finbert)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use for model (default: auto)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("FinBERT Model Initialization")
    logger.info("=" * 60)
    
    try:
        # Initialize model
        device = None if args.device == "auto" else args.device
        finbert = initialize_finbert(
            cache_dir=args.cache_dir,
            device=device,
        )
        
        # Display model info
        logger.info("\n" + "=" * 60)
        logger.info("Model Information:")
        logger.info("=" * 60)
        info = finbert.get_model_info()
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        
        # Run test prediction
        logger.info("\n" + "=" * 60)
        logger.info("Running Test Prediction:")
        logger.info("=" * 60)
        test_text = "The company reported strong quarterly earnings."
        result = finbert.predict(test_text)
        logger.info(f"  Text: {test_text}")
        logger.info(f"  Sentiment: {result['label']}")
        logger.info(f"  Confidence: {result['score']:.2%}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ FinBERT initialization successful!")
        logger.info("=" * 60)
        logger.info("The model is now cached and ready for use.")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Initialization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
