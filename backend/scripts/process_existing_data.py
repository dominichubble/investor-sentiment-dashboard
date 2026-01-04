"""
Process Existing Data with Sentiment Analysis

This script reads existing data from your data collection pipelines
(Reddit, Twitter, News) and processes it through sentiment analysis,
then saves the predictions using the storage module.

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

from app.models import analyze_batch
from app.storage import save_predictions_batch


def process_csv_file(input_file: Path, source: str, text_column: str = "text") -> int:
    """
    Process a CSV file through sentiment analysis and save predictions.

    Args:
        input_file: Path to input CSV file
        source: Data source identifier (reddit, twitter, news)
        text_column: Name of column containing text to analyze

    Returns:
        Number of predictions saved
    """
    print(f"Reading {input_file}...")

    # Read CSV
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0

    # Check if text column exists
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in CSV")
        print(f"Available columns: {', '.join(df.columns)}")
        return 0

    # Get texts (drop nulls and empty strings)
    texts = df[text_column].dropna().tolist()
    texts = [str(t).strip() for t in texts if t and str(t).strip()]

    if not texts:
        print("No valid texts found in file")
        return 0

    print(f"Found {len(texts)} texts to analyze")

    # Analyze sentiment in batches
    print("Analyzing sentiment...")
    try:
        results = analyze_batch(texts, batch_size=32)
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return 0

    # Prepare predictions for storage
    print("Preparing predictions...")
    timestamp = datetime.utcnow().isoformat()
    predictions = []

    for text, result in zip(texts, results):
        predictions.append(
            {
                "text": text[:500],  # Limit text length for storage
                "source": source,
                "timestamp": timestamp,
                "label": result["label"],
                "confidence": result["score"],
            }
        )

    # Save predictions
    output_file = Path("data") / "predictions" / f"{source}_predictions.csv"
    print(f"Saving {len(predictions)} predictions to {output_file}...")

    try:
        count = save_predictions_batch(
            predictions, output_file, format="csv", append=True
        )
        print(f"✓ Saved {count} predictions successfully!")
        return count
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return 0


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
        print(f"No files found matching: {args.input}")
        return

    print(f"Found {len(files)} file(s) to process")
    total_saved = 0

    for file in files:
        if not file.exists():
            print(f"Skipping {file} (does not exist)")
            continue

        count = process_csv_file(file, args.source, args.text_column)
        total_saved += count
        print()

    print(f"Total predictions saved: {total_saved}")


if __name__ == "__main__":
    main()
