"""
Process Existing Data with Sentiment Analysis

This script reads existing data from your data collection pipelines
(Reddit, Twitter, News) and processes it through sentiment analysis,
then saves the predictions to SQLite storage.

Usage:
    python process_existing_data.py --input data/twitter_finance_*.csv --source twitter
    python process_existing_data.py --input data/reddit_*.csv --source reddit
    python process_existing_data.py --input data/news_*.csv --source news
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import analyze_batch
from app.storage import StockSentimentStorage
from app.storage.record_ids import make_record_id


def _coerce_timestamp(value) -> str | None:
    """Best-effort parse of timestamps into ISO-8601 (UTC) with Z suffix."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    # Numeric timestamps (assume Unix seconds or milliseconds)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            ts_val = float(value)
            if ts_val > 1e12:  # likely milliseconds
                ts_val = ts_val / 1000.0
            dt = datetime.utcfromtimestamp(ts_val)
            return dt.isoformat() + "Z"
        except (ValueError, OSError, OverflowError):
            return None

    # String timestamps
    try:
        dt = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        dt = dt.to_pydatetime()
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt.isoformat() + "Z"
    except Exception:
        return None


def _extract_timestamp(row: pd.Series) -> str:
    """Extract a stable timestamp from a row, with fallback to now."""
    candidates = [
        "timestamp",
        "created_at",
        "created_utc",
        "published_at",
        "date",
        "datetime",
    ]
    for key in candidates:
        if key in row:
            value = row[key]
            if value is None:
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            parsed = _coerce_timestamp(value)
            if parsed:
                return parsed
    return datetime.utcnow().isoformat() + "Z"


def _extract_source_id(row: pd.Series) -> str:
    """Extract a stable source identifier from a row if available."""
    candidates = [
        "id",
        "source_id",
        "url",
        "permalink",
        "reddit_id",
        "tweet_id",
        "post_id",
    ]
    for key in candidates:
        if key in row:
            value = row[key]
            if value is None:
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            value = str(value).strip()
            if value:
                return value
    return ""


def process_csv_file(
    input_file: Path,
    source: str,
    storage: StockSentimentStorage,
    text_column: str = "text",
) -> int:
    """
    Process a CSV file through sentiment analysis and save predictions.

    Args:
        input_file: Path to input CSV file
        source: Data source identifier (reddit, twitter, news)
        storage: SQLite storage instance
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

    # Get valid rows (drop nulls and empty strings)
    valid_rows = df[df[text_column].notna()].copy()
    valid_rows["__text__"] = valid_rows[text_column].astype(str).str.strip()
    valid_rows = valid_rows[valid_rows["__text__"] != ""].reset_index(drop=True)
    texts = valid_rows["__text__"].tolist()

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

    # Prepare records for SQLite storage
    print("Preparing records for SQLite...")
    records = []

    for idx, (text, result) in enumerate(zip(texts, results)):
        row = valid_rows.iloc[idx]
        timestamp = _extract_timestamp(row)
        source_id = _extract_source_id(row)
        record_id = make_record_id("doc", source, source_id, timestamp, text[:200])
        records.append(
            {
                "id": record_id,
                "record_type": "document",
                "document_id": record_id,
                "text": text[:2000],
                "ticker": None,
                "mentioned_as": "",
                "sentiment_label": result["label"],
                "sentiment_score": result["score"],
                "context": "",
                "source": source,
                "source_id": source_id,
                "position_start": None,
                "position_end": None,
                "timestamp": timestamp,
                "sentiment_mode": "finbert",
            }
        )

    print(f"Saving {len(records)} predictions to SQLite...")

    try:
        count = storage.save_records_batch(records)
        print(f"âœ“ Saved {count} predictions successfully!")
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

    storage = StockSentimentStorage()
    storage.load()

    print(f"Found {len(files)} file(s) to process")
    total_saved = 0

    for file in files:
        if not file.exists():
            print(f"Skipping {file} (does not exist)")
            continue

        count = process_csv_file(file, args.source, storage, args.text_column)
        total_saved += count
        print()

    print(f"Total predictions saved: {total_saved}")


if __name__ == "__main__":
    main()
