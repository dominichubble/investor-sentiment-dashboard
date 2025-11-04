#!/usr/bin/env python3
"""
Text Preprocessing Pipeline

Preprocesses collected data from Reddit, Twitter, and News APIs by applying
tokenization, stopword removal, and lemmatization. Outputs preprocessed
JSON files ready for sentiment analysis.

Usage:
    python preprocess_data.py --input data/raw/reddit --output data/processed/reddit
    python preprocess_data.py --source all --remove-stopwords --lemmatize
    python preprocess_data.py --input data/raw/news/2025-11-02 --config minimal
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from app.preprocessing import TextProcessor
except ImportError:
    print("Error: Could not import preprocessing module.")
    print("Make sure you're running from the backend directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Preprocessing configurations
CONFIGS = {
    "minimal": {
        "lowercase": True,
        "remove_urls": True,
        "remove_stopwords": False,
        "lemmatize": False,
        "preserve_financial": True,
        "preserve_financial_punctuation": False,
        "handle_negations": False,
    },
    "standard": {
        "lowercase": True,
        "remove_urls": True,
        "remove_stopwords": True,
        "lemmatize": False,
        "preserve_financial": True,
        "preserve_financial_punctuation": False,
        "handle_negations": False,
    },
    "full": {
        "lowercase": True,
        "remove_urls": True,
        "remove_stopwords": True,
        "lemmatize": True,
        "preserve_financial": True,
        "preserve_financial_punctuation": False,
        "handle_negations": False,
    },
    "finbert": {
        "lowercase": False,  # Preserve case for FinBERT
        "remove_urls": True,
        "remove_stopwords": False,  # Keep all words
        "lemmatize": False,  # No lemmatization for transformers
        "preserve_financial": True,
        "preserve_financial_punctuation": True,  # Keep %, $, decimals
        "handle_negations": True,  # Critical for sentiment
    },
}

# Text fields by data source
SOURCE_TEXT_FIELDS = {
    "reddit": ["title", "selftext"],
    "twitter": ["text"],
    "news": ["clean_title", "clean_description", "clean_content"],
}


def find_json_files(input_path: Path) -> List[Path]:
    """
    Find all JSON files in input directory.

    Args:
        input_path: Directory to search

    Returns:
        List of JSON file paths
    """
    if input_path.is_file() and input_path.suffix == ".json":
        return [input_path]

    json_files = list(input_path.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {input_path}")
    return json_files


def detect_source_type(data: List[Dict[str, Any]], filepath: Path) -> Optional[str]:
    """
    Detect data source type from file path or content.

    Args:
        data: List of data records
        filepath: Path to data file

    Returns:
        Source type: 'reddit', 'twitter', or 'news'
    """
    # Try to detect from path
    path_str = str(filepath).lower()
    if "reddit" in path_str:
        return "reddit"
    elif "twitter" in path_str or "tweet" in path_str:
        return "twitter"
    elif "news" in path_str:
        return "news"

    # Try to detect from data fields
    if data:
        fields = set(data[0].keys())
        if "selftext" in fields and "subreddit" in fields:
            return "reddit"
        elif "retweet_count" in fields or "author_id" in fields:
            return "twitter"
        elif "source_id" in fields or "url_to_image" in fields:
            return "news"

    return None


def preprocess_record(
    record: Dict[str, Any],
    source_type: str,
    processor: TextProcessor,
) -> Dict[str, Any]:
    """
    Preprocess a single data record.

    Args:
        record: Data record dictionary
        source_type: Type of data source
        processor: Configured TextProcessor instance

    Returns:
        Record with added preprocessed fields
    """
    preprocessed = record.copy()
    text_fields = SOURCE_TEXT_FIELDS.get(source_type, [])

    for field in text_fields:
        if field in record and record[field]:
            # Generate preprocessed field names
            tokens_field = f"{field}_tokens"
            processed_field = f"{field}_processed"

            # Process text
            tokens = processor.process(record[field], return_string=False)
            processed_text = " ".join(tokens)

            # Add to record
            preprocessed[tokens_field] = tokens
            preprocessed[processed_field] = processed_text

            logger.debug(
                f"Preprocessed {field}: {len(record[field])} chars -> "
                f"{len(tokens)} tokens"
            )

    return preprocessed


def preprocess_file(
    input_file: Path,
    output_dir: Path,
    processor: TextProcessor,
    source_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Preprocess a single JSON file.

    Args:
        input_file: Input JSON file path
        output_dir: Output directory
        processor: Configured TextProcessor
        source_type: Optional source type override

    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing {input_file.name}...")

    # Load data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading {input_file}: {e}")
        return {"error": str(e), "file": str(input_file)}

    # Handle different JSON structures
    if isinstance(data, dict):
        if "data" in data:
            records = data["data"]
            metadata = {k: v for k, v in data.items() if k != "data"}
        else:
            records = [data]
            metadata = {}
    else:
        records = data
        metadata = {}

    if not records:
        logger.warning(f"No records found in {input_file}")
        return {"file": str(input_file), "records": 0}

    # Detect source type
    if not source_type:
        source_type = detect_source_type(records, input_file)
        if not source_type:
            logger.error(f"Could not detect source type for {input_file}")
            return {"error": "Unknown source type", "file": str(input_file)}

    logger.info(f"Detected source type: {source_type}")

    # Preprocess all records
    preprocessed_records = []
    total_tokens = 0

    for i, record in enumerate(records):
        try:
            preprocessed = preprocess_record(record, source_type, processor)
            preprocessed_records.append(preprocessed)

            # Count tokens
            for field in SOURCE_TEXT_FIELDS.get(source_type, []):
                tokens_field = f"{field}_tokens"
                if tokens_field in preprocessed:
                    total_tokens += len(preprocessed[tokens_field])

        except Exception as e:
            logger.error(f"Error preprocessing record {i} in {input_file}: {e}")
            preprocessed_records.append(record)  # Keep original on error

    # Prepare output
    output_data = {
        "data": preprocessed_records,
        "metadata": {
            **metadata,
            "preprocessing": {
                "processed_at": datetime.now().isoformat(),
                "source_file": str(input_file.name),
                "source_type": source_type,
                "config": {
                    "lowercase": processor.lowercase,
                    "remove_urls": processor.remove_urls,
                    "remove_stopwords": processor.remove_stopwords,
                    "lemmatize": processor.lemmatize,
                    "preserve_financial": processor.preserve_financial,
                    "preserve_financial_punctuation": processor.preserve_financial_punctuation,
                    "handle_negations": processor.handle_negations,
                },
                "stats": {
                    "total_records": len(preprocessed_records),
                    "total_tokens": total_tokens,
                    "avg_tokens_per_record": (
                        total_tokens / len(preprocessed_records)
                        if preprocessed_records
                        else 0
                    ),
                },
            },
        },
    }

    # Save output
    output_file = output_dir / input_file.name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved preprocessed data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving {output_file}: {e}")
        return {"error": str(e), "file": str(input_file)}

    return {
        "file": str(input_file.name),
        "source_type": source_type,
        "records": len(preprocessed_records),
        "total_tokens": total_tokens,
        "output": str(output_file),
    }


def run_preprocessing(
    input_path: Path,
    output_path: Path,
    config_name: str = "standard",
    source_type: Optional[str] = None,
    **processor_kwargs,
) -> Dict[str, Any]:
    """
    Run preprocessing pipeline on input data.

    Args:
        input_path: Input directory or file
        output_path: Output directory
        config_name: Preprocessing configuration name
        source_type: Optional source type override
        **processor_kwargs: Override processor configuration

    Returns:
        Dictionary with processing summary
    """
    logger.info("="*60)
    logger.info("Text Preprocessing Pipeline")
    logger.info("="*60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Config: {config_name}")
    logger.info("="*60)

    # Get configuration
    config = CONFIGS.get(config_name, CONFIGS["standard"]).copy()
    config.update(processor_kwargs)

    logger.info("Preprocessing configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Filter config to only include valid TextProcessor parameters
    valid_params = {
        'lowercase', 'remove_urls', 'remove_stopwords',
        'lemmatize', 'preserve_financial', 'preserve_financial_punctuation',
        'handle_negations', 'custom_stopwords'
    }
    processor_config = {k: v for k, v in config.items() if k in valid_params}

    # Create processor
    processor = TextProcessor(**processor_config)  # type: ignore

    # Find input files
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return {"error": "Input path not found"}

    json_files = find_json_files(input_path)
    if not json_files:
        logger.error(f"No JSON files found in {input_path}")
        return {"error": "No JSON files found"}

    # Process each file
    results = []
    for json_file in json_files:
        # Maintain directory structure in output
        if input_path.is_dir():
            rel_path = json_file.parent.relative_to(input_path)
            file_output_dir = output_path / rel_path
        else:
            file_output_dir = output_path

        result = preprocess_file(json_file, file_output_dir, processor, source_type)
        results.append(result)

    # Summary
    logger.info("=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)

    total_records = sum(r.get("records", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    errors = sum(1 for r in results if "error" in r)

    logger.info(f"Files processed: {len(results)}")
    logger.info(f"Total records: {total_records}")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Avg tokens/record: {total_tokens/total_records if total_records else 0:.1f}")
    if errors:
        logger.warning(f"Errors: {errors}")

    return {
        "files_processed": len(results),
        "total_records": total_records,
        "total_tokens": total_tokens,
        "results": results,
        "config": config,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess text data for sentiment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess Reddit data with standard config
  python preprocess_data.py --input ../data/processed/reddit --output ../data/preprocessed/reddit

  # Preprocess with full pipeline (stopwords + lemmatization)
  python preprocess_data.py --input ../data/processed/news --config full

  # Custom preprocessing
  python preprocess_data.py --input data.json --remove-stopwords --lemmatize

  # Process specific source type
  python preprocess_data.py --input data.json --source twitter
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input directory or JSON file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory (default: input_dir + '_preprocessed')",
    )

    parser.add_argument(
        "--config",
        "-c",
        choices=list(CONFIGS.keys()),
        default="standard",
        help="Preprocessing configuration (default: standard)",
    )

    parser.add_argument(
        "--source",
        "-s",
        choices=["reddit", "twitter", "news"],
        help="Force source type (auto-detected if not specified)",
    )

    # Preprocessing options
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove stopwords (overrides config)",
    )

    parser.add_argument(
        "--no-remove-stopwords",
        action="store_true",
        help="Don't remove stopwords (overrides config)",
    )

    parser.add_argument(
        "--lemmatize",
        action="store_true",
        help="Apply lemmatization (overrides config)",
    )

    parser.add_argument(
        "--no-lemmatize",
        action="store_true",
        help="Don't apply lemmatization (overrides config)",
    )

    parser.add_argument(
        "--no-preserve-financial",
        action="store_true",
        help="Don't preserve financial terms when removing stopwords",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Determine output path
    if not args.output:
        if args.input.is_dir():
            args.output = args.input.parent / f"{args.input.name}_preprocessed"
        else:
            args.output = args.input.parent / "preprocessed"

    # Build processor kwargs from args
    processor_kwargs = {}
    if args.remove_stopwords:
        processor_kwargs["remove_stopwords"] = True
    if args.no_remove_stopwords:
        processor_kwargs["remove_stopwords"] = False
    if args.lemmatize:
        processor_kwargs["lemmatize"] = True
    if args.no_lemmatize:
        processor_kwargs["lemmatize"] = False
    if args.no_preserve_financial:
        processor_kwargs["preserve_financial"] = False

    # Run preprocessing
    try:
        summary = run_preprocessing(
            input_path=args.input,
            output_path=args.output,
            config_name=args.config,
            source_type=args.source,
            **processor_kwargs,
        )

        if "error" in summary:
            logger.error(f"Preprocessing failed: {summary['error']}")
            return 1

        logger.info(f"\n✓ Preprocessed data saved to: {args.output}")
        return 0

    except KeyboardInterrupt:
        logger.warning("\nPreprocessing interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Preprocessing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
