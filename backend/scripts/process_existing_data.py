"""
Process Existing Data with Sentiment Analysis

This script reads existing data from your data collection pipelines
(Reddit, Twitter, News) and processes it through sentiment analysis,
then saves the predictions using the storage module.

Features:
- Comprehensive error handling and logging
- Tracks failed items and saves to failed_items.json
- Robust batch processing with individual fallback
- Progress tracking and detailed reporting

Usage:
    python process_existing_data.py --input data/twitter_finance_*.csv --source twitter
    python process_existing_data.py --input data/reddit_*.csv --source reddit
    python process_existing_data.py --input data/news_*.csv --source news
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.logging_config import get_logger, log_exception, setup_logging
from app.models import analyze_batch
from app.storage import save_predictions_batch

# Setup logging
setup_logging()
logger = get_logger(__name__)


def process_csv_file(input_file: Path, source: str, text_column: str = "text") -> dict:
    """
    Process a CSV file through sentiment analysis and save predictions.

    Args:
        input_file: Path to input CSV file
        source: Data source identifier (reddit, twitter, news)
        text_column: Name of column containing text to analyze

    Returns:
        Dictionary with processing statistics:
            - total: Total records in file
            - processed: Successfully processed records
            - failed: Failed records
            - skipped: Skipped records (empty text)
            - saved: Successfully saved predictions
    """
    logger.info(f"Processing file: {input_file}")
    
    stats = {
        "total": 0,
        "processed": 0,
        "failed": 0,
        "skipped": 0,
        "saved": 0,
        "file": str(input_file)
    }

    # Read CSV
    try:
        df = pd.read_csv(input_file)
        stats["total"] = len(df)
        logger.info(f"Loaded {len(df)} records from {input_file}")
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
        return stats
    except Exception as e:
        log_exception(logger, e, f"Error reading file {input_file}")
        return stats

    # Check if text column exists
    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' not found in CSV")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        return stats

    # Get texts (drop nulls and empty strings)
    texts = df[text_column].dropna().tolist()
    original_count = len(texts)
    texts = [str(t).strip() for t in texts if t and str(t).strip()]
    stats["skipped"] = original_count - len(texts)

    if not texts:
        logger.warning("No valid texts found in file")
        return stats

    logger.info(f"Found {len(texts)} valid texts to analyze ({stats['skipped']} skipped)")

    # Analyze sentiment in batches
    logger.info("Starting sentiment analysis...")
    try:
        results, failures = analyze_batch(
            texts, 
            batch_size=32, 
            skip_errors=True, 
            track_failures=True
        )
        
        # Count successful analyses
        successful_results = [r for r in results if r is not None]
        stats["processed"] = len(successful_results)
        stats["failed"] = failures.count() if failures else 0
        
        logger.info(
            f"Analysis complete: {stats['processed']} successful, "
            f"{stats['failed']} failed"
        )
        
    except Exception as e:
        log_exception(logger, e, "Critical error during sentiment analysis")
        # Save failures if any were tracked
        if 'failures' in locals() and failures and failures.count() > 0:
            failed_file = Path("data") / "failed_items" / f"{source}_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                count = failures.save(failed_file)
                logger.info(f"Saved {count} failed items to {failed_file}")
            except Exception as save_error:
                logger.error(f"Could not save failures: {save_error}")
        return stats

    # Save failures if any
    if failures and failures.count() > 0:
        failed_file = Path("data") / "failed_items" / f"{source}_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            failed_file.parent.mkdir(parents=True, exist_ok=True)
            count = failures.save(failed_file)
            logger.info(f"✓ Saved {count} failed items to {failed_file}")
        except Exception as e:
            log_exception(logger, e, f"Error saving failed items to {failed_file}")

    # Prepare predictions for storage
    logger.info("Preparing predictions for storage...")
    timestamp = datetime.utcnow().isoformat()
    predictions = []

    for text, result in zip(texts, results):
        if result is not None:  # Only save successful predictions
            try:
                predictions.append({
                    "text": text[:500],  # Limit text length for storage
                    "source": source,
                    "timestamp": timestamp,
                    "label": result["label"],
                    "confidence": result["score"],
                })
            except KeyError as e:
                logger.error(f"Missing key in result: {e}")
                stats["failed"] += 1

    if not predictions:
        logger.warning("No predictions to save")
        return stats

    # Save predictions
    output_dir = Path("data") / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{source}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    logger.info(f"Saving {len(predictions)} predictions to {output_file}...")

    try:
        count = save_predictions_batch(
            predictions, 
            output_file, 
            format="csv", 
            append=False
        )
        stats["saved"] = count
        logger.info(f"✓ Successfully saved {count} predictions!")
        
    except Exception as e:
        log_exception(logger, e, f"Error saving predictions to {output_file}")
        return stats

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Process existing data with sentiment analysis"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input CSV file (supports wildcards)"
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["reddit", "twitter", "news"],
        help="Data source",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of column containing text (default: text)",
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Starting sentiment analysis processing")
    logger.info(f"Source: {args.source}")
    logger.info(f"Input pattern: {args.input}")
    logger.info(f"Text column: {args.text_column}")
    logger.info("="*60)

    # Handle wildcards
    input_path = Path(args.input)
    if "*" in args.input:
        # Glob pattern
        parent = input_path.parent
        pattern = input_path.name
        files = list(parent.glob(pattern))
    else:
        files = [input_path]

    if not files:
        logger.error(f"No files found matching: {args.input}")
        print(f"ERROR: No files found matching: {args.input}")
        return

    logger.info(f"Found {len(files)} file(s) to process")
    
    # Track overall statistics
    overall_stats = {
        "files_processed": 0,
        "files_failed": 0,
        "total_records": 0,
        "total_processed": 0,
        "total_failed": 0,
        "total_skipped": 0,
        "total_saved": 0,
    }

    for file_idx, file in enumerate(files, 1):
        logger.info("")
        logger.info(f"Processing file {file_idx}/{len(files)}: {file}")
        logger.info("-" * 60)
        
        if not file.exists():
            logger.warning(f"File does not exist, skipping: {file}")
            overall_stats["files_failed"] += 1
            continue

        try:
            stats = process_csv_file(file, args.source, args.text_column)
            
            # Update overall stats
            overall_stats["files_processed"] += 1
            overall_stats["total_records"] += stats["total"]
            overall_stats["total_processed"] += stats["processed"]
            overall_stats["total_failed"] += stats["failed"]
            overall_stats["total_skipped"] += stats["skipped"]
            overall_stats["total_saved"] += stats["saved"]
            
            # Log file summary
            logger.info("-" * 60)
            logger.info(f"File summary for {file.name}:")
            logger.info(f"  Total records: {stats['total']}")
            logger.info(f"  Processed: {stats['processed']}")
            logger.info(f"  Failed: {stats['failed']}")
            logger.info(f"  Skipped: {stats['skipped']}")
            logger.info(f"  Saved: {stats['saved']}")
            
        except Exception as e:
            log_exception(logger, e, f"Unexpected error processing {file}")
            overall_stats["files_failed"] += 1

    # Final summary
    logger.info("")
    logger.info("="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Files processed: {overall_stats['files_processed']}/{len(files)}")
    logger.info(f"Files failed: {overall_stats['files_failed']}")
    logger.info(f"Total records: {overall_stats['total_records']}")
    logger.info(f"Successfully processed: {overall_stats['total_processed']}")
    logger.info(f"Failed: {overall_stats['total_failed']}")
    logger.info(f"Skipped (empty): {overall_stats['total_skipped']}")
    logger.info(f"Predictions saved: {overall_stats['total_saved']}")
    logger.info("="*60)
    
    # Also print summary to console for visibility
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Files processed: {overall_stats['files_processed']}/{len(files)}")
    print(f"Total predictions saved: {overall_stats['total_saved']}")
    print(f"Failed items: {overall_stats['total_failed']}")
    print(f"Check logs/ directory for detailed logs and failed_items.json")
    print("="*60)


if __name__ == "__main__":
    main()
