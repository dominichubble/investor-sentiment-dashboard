#!/usr/bin/env python3
"""
Import stock_tweets.csv into SQLite with FinBERT sentiment classification.

Features:
  - Batch processing with configurable batch size
  - Progress file for resume capability (restarts from last completed batch)
  - Per-tweet ticker extraction from the CSV's 'Stock Name' column
  - Both document-level and stock-level records are created

Usage:
    cd backend
    python -m scripts.import_tweets                           # full import
    python -m scripts.import_tweets --limit 1000              # import first 1000
    python -m scripts.import_tweets --batch-size 64           # larger batches
    python -m scripts.import_tweets --csv ../stock_tweets.csv # custom path
    python -m scripts.import_tweets --reset                   # ignore progress, restart
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the backend package is importable when invoked as a script.
# ---------------------------------------------------------------------------
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.storage.database import get_engine
from app.storage.record_ids import make_record_id
from app.storage.sqlite_storage import SQLiteSentimentStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = Path(__file__).resolve().parent / ".import_tweets_progress.json"


def _load_progress(reset: bool) -> dict:
    if reset or not PROGRESS_FILE.exists():
        return {"completed_batches": 0, "total_inserted": 0}
    try:
        return json.loads(PROGRESS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"completed_batches": 0, "total_inserted": 0}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress))


def _parse_timestamp(raw: str) -> str:
    """Normalise the CSV 'Date' field to UTC ISO-8601."""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except (ValueError, TypeError):
        return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _read_csv(csv_path: Path, limit: int | None) -> list[dict]:
    """Read the stock_tweets CSV into dicts."""
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            tweet = (row.get("Tweet") or "").strip()
            if not tweet:
                continue
            rows.append(
                {
                    "text": tweet,
                    "ticker": (row.get("Stock Name") or "").strip().upper() or None,
                    "company_name": (row.get("Company Name") or "").strip(),
                    "timestamp": _parse_timestamp(row.get("Date", "")),
                    "source": "twitter",
                    "source_id": hashlib.sha1(
                        f"{row.get('Date','')}{tweet[:80]}".encode()
                    ).hexdigest()[:16],
                }
            )
    return rows


def _build_db_rows(
    rows: list[dict], sentiments: list[dict | None]
) -> list[dict]:
    """Build DB-ready dicts from tweet rows and their FinBERT predictions."""
    db_rows: list[dict] = []
    for row, sentiment in zip(rows, sentiments):
        if not sentiment:
            continue
        ts = row["timestamp"]
        src = row["source"]
        src_id = row["source_id"]
        text = row["text"]
        ticker = row["ticker"]

        doc_id = make_record_id("doc", src, src_id, ts, text[:120])

        # Document-level record
        db_rows.append(
            {
                "id": doc_id,
                "record_type": "document",
                "document_id": doc_id,
                "text": text,
                "ticker": ticker,
                "mentioned_as": f"${ticker}" if ticker else "",
                "sentiment_label": sentiment["label"],
                "sentiment_score": float(sentiment["score"]),
                "context": "",
                "source": src,
                "source_id": src_id,
                "position_start": None,
                "position_end": None,
                "timestamp": ts,
                "sentiment_mode": "finbert",
            }
        )

        # Stock-level record (if ticker present)
        if ticker:
            stock_id = make_record_id("stock", doc_id, ticker, ts)
            db_rows.append(
                {
                    "id": stock_id,
                    "record_type": "stock",
                    "document_id": doc_id,
                    "text": text,
                    "ticker": ticker,
                    "mentioned_as": f"${ticker}",
                    "sentiment_label": sentiment["label"],
                    "sentiment_score": float(sentiment["score"]),
                    "context": text[:200],
                    "source": src,
                    "source_id": src_id,
                    "position_start": None,
                    "position_end": None,
                    "timestamp": ts,
                    "sentiment_mode": "finbert",
                }
            )

    return db_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import stock_tweets.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "stock_tweets.csv",
        help="Path to stock_tweets.csv",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="FinBERT batch size")
    parser.add_argument("--limit", type=int, default=None, help="Max tweets to import")
    parser.add_argument(
        "--reset", action="store_true", help="Ignore progress file, restart"
    )
    args = parser.parse_args()

    if not args.csv.exists():
        logger.error("CSV not found: %s", args.csv)
        sys.exit(1)

    # --- Load CSV ---
    logger.info("Reading %s ...", args.csv)
    rows = _read_csv(args.csv, args.limit)
    logger.info("Loaded %d tweets with text", len(rows))

    if not rows:
        logger.warning("No valid tweets found. Exiting.")
        return

    # --- Resume support ---
    progress = _load_progress(args.reset)
    skip_batches = progress["completed_batches"]
    total_inserted = progress["total_inserted"]

    batch_size = args.batch_size
    total_batches = (len(rows) + batch_size - 1) // batch_size

    if skip_batches >= total_batches:
        logger.info("All %d batches already completed (%d inserted). Done.", total_batches, total_inserted)
        return

    if skip_batches > 0:
        logger.info("Resuming from batch %d / %d  (%d already inserted)", skip_batches + 1, total_batches, total_inserted)

    # --- Lazy-load FinBERT (heavy) ---
    logger.info("Loading FinBERT model (this may take a minute on first run) ...")
    from app.models.sentiment_inference import analyze_batch

    # --- Init storage ---
    get_engine()
    storage = SQLiteSentimentStorage()

    t0 = time.time()
    for batch_idx in range(skip_batches, total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(rows))
        batch_rows = rows[batch_start:batch_end]

        texts = [r["text"] for r in batch_rows]
        sentiments = analyze_batch(texts, batch_size=batch_size, return_all_scores=False)

        db_rows = _build_db_rows(batch_rows, sentiments)
        inserted = storage.save_records_batch(db_rows)
        total_inserted += inserted

        progress["completed_batches"] = batch_idx + 1
        progress["total_inserted"] = total_inserted
        _save_progress(progress)

        elapsed = time.time() - t0
        rate = (batch_idx - skip_batches + 1) / elapsed if elapsed > 0 else 0
        eta = (total_batches - batch_idx - 1) / rate if rate > 0 else 0

        logger.info(
            "Batch %d/%d  | +%d rows | Total inserted: %d | %.1f batch/s | ETA: %s",
            batch_idx + 1,
            total_batches,
            inserted,
            total_inserted,
            rate,
            f"{int(eta // 60)}m {int(eta % 60)}s" if eta < 36000 else "calculating...",
        )

    elapsed_total = time.time() - t0
    logger.info(
        "Import complete. %d records inserted in %dm %ds.",
        total_inserted,
        int(elapsed_total // 60),
        int(elapsed_total % 60),
    )

    # Clean up progress file on successful completion
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        logger.info("Progress file cleaned up.")


if __name__ == "__main__":
    main()
