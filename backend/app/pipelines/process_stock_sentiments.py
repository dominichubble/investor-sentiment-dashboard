"""
Stock-sentiment processing pipeline.

Reads raw Reddit and News data, computes document-level and per-stock sentiment
using StockSentimentAnalyzer, and stores unified sentiment records in SQLite
(`data/db/sentiments.db`).

Usage:
    cd backend
    python -m app.pipelines.process_stock_sentiments
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ValidationError

from app.schemas.sentiment import SentimentRecord
from app.stocks import StockSentimentAnalyzer
from app.storage import StockSentimentStorage
from app.storage.record_ids import make_record_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_TEXT_LEN = 2000
MAX_CONTEXT_LEN = 500


def _date_from_filename(filepath: Path) -> Optional[str]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filepath.name)
    if match:
        return f"{match.group(1)}T12:00:00"
    return None


def _find_source_files(data_dir: Path, source: str, pattern: str) -> List[Path]:
    """Find data files for a source in data/raw/."""
    source_dir = data_dir / "raw" / source
    if not source_dir.exists():
        logger.warning("Directory not found: %s", source_dir)
        return []

    found = sorted(source_dir.rglob(pattern))
    found = [f for f in found if "_meta" not in f.name]
    if found:
        logger.info("  Found %d %s files in raw/%s/", len(found), source, source)
    return found


def load_reddit_file(filepath: Path) -> List[Dict]:
    """Load a Reddit data file and extract records."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("Error loading %s: %s", filepath, e)
        return []

    records = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(records, list):
        return []

    results = []
    for item in records:
        title = item.get("title", "")
        selftext = item.get("selftext", "")

        text = title
        if selftext and len(selftext) > 10:
            text = f"{title}. {selftext}" if title else selftext

        if not text or len(text) < 15:
            continue

        created_utc = item.get("created_utc")
        timestamp = None
        if created_utc:
            try:
                timestamp = datetime.fromtimestamp(created_utc).isoformat()
            except (ValueError, OSError, OverflowError):
                timestamp = None

        if not timestamp:
            timestamp = _date_from_filename(filepath)

        if not timestamp:
            continue

        results.append(
            {
                "text": text[:MAX_TEXT_LEN],
                "source": "reddit",
                "timestamp": timestamp,
                "source_id": item.get("id", ""),
                "subreddit": item.get("subreddit", ""),
                "permalink": item.get("permalink", ""),
            }
        )

    return results


def load_news_file(filepath: Path) -> List[Dict]:
    """Load a news data file and extract records."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("Error loading %s: %s", filepath, e)
        return []

    records = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(records, list):
        return []

    results = []
    for item in records:
        title = item.get("clean_title") or item.get("title", "")
        description = item.get("clean_description") or item.get("description", "")
        content = item.get("clean_content") or item.get("content", "")

        text = title
        if description and len(description) > 20:
            text = f"{title}. {description}" if title else description
        elif content and len(content) > 20:
            text = f"{title}. {content}" if title else content

        if not text or len(text) < 15:
            continue

        published_at = item.get("published_at") or item.get("publishedAt")
        timestamp = None
        if published_at:
            try:
                parsed = published_at.replace("Z", "+00:00")
                dt = datetime.fromisoformat(parsed)
                timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S")
            except (ValueError, TypeError):
                timestamp = None

        if not timestamp:
            timestamp = _date_from_filename(filepath)

        if not timestamp:
            continue

        source_id = item.get("url") or item.get("source_id") or ""

        results.append(
            {
                "text": text[:MAX_TEXT_LEN],
                "source": "news",
                "timestamp": timestamp,
                "source_id": source_id,
                "source_name": item.get("source_name", ""),
            }
        )

    return results


def process_all_data(data_dir: Path, max_records: Optional[int] = None) -> Dict:
    """
    Process all available data files and generate unified sentiment records.

    Args:
        data_dir: Path to the data directory.
        max_records: Optional cap for debugging.

    Returns:
        Statistics about processing.
    """
    all_records: List[Dict] = []

    # 1. Load Reddit data
    reddit_files = _find_source_files(data_dir, "reddit", "reddit_finance_*.json")
    logger.info("Found %d Reddit data files total", len(reddit_files))
    for filepath in reddit_files:
        records = load_reddit_file(filepath)
        all_records.extend(records)
        if records:
            logger.info("  Loaded %d records from %s", len(records), filepath.name)

    # 2. Load News data
    news_files = _find_source_files(data_dir, "news", "news_finance_*.json")
    logger.info("Found %d News data files total", len(news_files))
    for filepath in news_files:
        records = load_news_file(filepath)
        all_records.extend(records)
        if records:
            logger.info("  Loaded %d records from %s", len(records), filepath.name)

    logger.info("\nTotal records loaded: %d", len(all_records))

    # Initialize analyzer and storage
    try:
        analyzer = StockSentimentAnalyzer()
    except Exception as e:
        logger.error("Failed to initialize StockSentimentAnalyzer: %s", e)
        sys.exit(1)

    storage = StockSentimentStorage()
    storage.load()

    unified_records: List[Dict] = []
    ticker_stats: Dict[str, int] = {}
    processed = 0
    doc_records = 0
    stock_records = 0
    validation_errors = 0

    for record in all_records:
        if max_records is not None and processed >= max_records:
            break

        text = record["text"]

        try:
            analysis = analyzer.analyze(
                text=text,
                extract_context=True,
                include_movements=True,
            )
        except OSError as e:
            logger.error(
                "spaCy model not available (%s). Run: python -m spacy download en_core_web_sm",
                e,
            )
            sys.exit(1)
        except Exception as e:
            logger.warning("Failed analysis for record: %s", e)
            continue

        overall = analysis.get("overall_sentiment", {}) or {}
        timestamp = record.get("timestamp") or datetime.utcnow().isoformat()
        source = record.get("source", "")
        source_id = record.get("source_id", "")

        document_id = make_record_id(
            "doc",
            source,
            source_id,
            timestamp,
            text[:200],
        )

        document_record = {
            "id": document_id,
            "record_type": "document",
            "document_id": document_id,
            "text": text[:MAX_TEXT_LEN],
            "ticker": None,
            "mentioned_as": "",
            "sentiment_label": overall.get("label", "neutral"),
            "sentiment_score": float(overall.get("score", 0.5)),
            "context": "",
            "source": source,
            "source_id": source_id,
            "position_start": None,
            "position_end": None,
            "timestamp": timestamp,
            "sentiment_mode": "finbert",
        }

        try:
            validated = SentimentRecord(**document_record)
            unified_records.append(validated.model_dump())
        except ValidationError:
            validation_errors += 1
            unified_records.append(document_record)
        doc_records += 1

        for stock in analysis.get("stocks", []):
            ticker = stock.get("ticker")
            if not ticker:
                continue

            position = stock.get("position") or {}
            context = (stock.get("context") or "")[:MAX_CONTEXT_LEN]

            record_data = {
                "id": make_record_id(
                    "stock",
                    document_id,
                    ticker,
                    stock.get("mentioned_as", ""),
                    str(position.get("start")),
                    str(position.get("end")),
                ),
                "record_type": "stock",
                "document_id": document_id,
                "text": text[:MAX_TEXT_LEN],
                "ticker": ticker,
                "mentioned_as": stock.get("mentioned_as", ""),
                "sentiment_label": stock.get("sentiment", {}).get("label", "neutral"),
                "sentiment_score": float(stock.get("sentiment", {}).get("score", 0.5)),
                "context": context,
                "source": source,
                "source_id": source_id,
                "position_start": position.get("start"),
                "position_end": position.get("end"),
                "timestamp": timestamp,
                "sentiment_mode": "finbert",
            }

            try:
                validated = SentimentRecord(**record_data)
                unified_records.append(validated.model_dump())
            except ValidationError:
                validation_errors += 1
                unified_records.append(record_data)

            ticker_stats[ticker] = ticker_stats.get(ticker, 0) + 1
            stock_records += 1

        processed += 1

        if len(unified_records) >= 1000:
            storage.save_records_batch(unified_records)
            unified_records.clear()

    if validation_errors:
        logger.warning("Validation errors (records kept as-is): %d", validation_errors)

    if unified_records:
        storage.save_records_batch(unified_records)

    top_tickers = sorted(ticker_stats.items(), key=lambda x: x[1], reverse=True)[:20]
    logger.info("\nProcessing complete")
    logger.info("  Documents stored: %d", doc_records)
    logger.info("  Stock mentions stored: %d", stock_records)
    logger.info("  Unique tickers: %d", len(ticker_stats))

    if top_tickers:
        logger.info("\nTop 20 tickers by mentions:")
        for ticker, count in top_tickers:
            logger.info("  %s: %d mentions", ticker, count)

    return {
        "total_records": len(all_records),
        "documents": doc_records,
        "stock_mentions": stock_records,
        "unique_tickers": len(ticker_stats),
        "top_tickers": dict(top_tickers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process stock sentiment from collected data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the data directory (default: project-root/data)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit the number of records processed (debugging).",
    )
    parser.add_argument(
        "--use-finbert",
        action="store_true",
        help="(Deprecated) FinBERT is now always used.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="(Deprecated) Keyword mode has been removed.",
    )

    args = parser.parse_args()

    if args.use_finbert or args.fast:
        logger.warning("Legacy flags --use-finbert/--fast are ignored.")

    backend_dir = Path(__file__).parent.parent.parent
    data_dir = args.data_dir or (backend_dir.parent / "data")

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        return 1

    logger.info("=" * 60)
    logger.info("Stock Sentiment Processing Pipeline")
    logger.info("  Mode: StockSentimentAnalyzer (FinBERT)")
    logger.info("=" * 60)
    logger.info("Data directory: %s", data_dir)

    process_all_data(data_dir, max_records=args.max_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
