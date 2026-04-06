#!/usr/bin/env python3
"""
Stream English financial news headlines from Hugging Face → FinBERT → Neon.

Default dataset ``ashraq/financial-news`` (~1.85M rows) includes ``headline``,
``publisher``, ``stock`` (ticker), ``date``, and ``url``. Rows are streamed so
disk use stays modest; use ``--max-rows`` to cap each run.

Requires: ``pip install datasets`` (see backend/requirements.txt).

Usage (from repo root, PYTHONPATH=backend):
  python -m app.pipelines.ingest_huggingface_financial_news --max-rows 5000 --store-db
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from app.pipelines.ingest_news import clean_text
from app.services.import_service import ImportService
from app.utils.ticker_detection import TickerDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET = "ashraq/financial-news"


def _parse_published(raw: Any) -> str:
    if raw is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(raw, (int, float)):
        ts = float(raw)
        if ts > 1e12:
            ts /= 1000.0
        return (
            datetime.fromtimestamp(ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    s = str(raw).strip()
    if not s:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%m/%d/%Y",
    ):
        try:
            dt = datetime.strptime(s[:32], fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except ValueError:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_id(url: str, headline: str) -> str:
    h = hashlib.sha256(f"{url}\n{headline[:200]}".encode("utf-8", errors="ignore"))
    return h.hexdigest()[:32]


def _row_to_record(
    ex: Dict[str, Any],
    detector: TickerDetector,
) -> Optional[Dict[str, Any]]:
    headline = ex.get("headline") or ex.get("title") or ex.get("text") or ""
    if not isinstance(headline, str):
        headline = str(headline)
    headline = clean_text(headline)
    if len(headline) < 12:
        return None

    url = ex.get("url") or ""
    if not isinstance(url, str):
        url = str(url)
    publisher = ex.get("publisher") or ex.get("source") or "hf_financial_news"
    if not isinstance(publisher, str):
        publisher = str(publisher)
    publisher = re.sub(r"\s+", " ", publisher).strip()[:80]

    raw_stock = ex.get("stock") or ex.get("ticker") or ""
    if not isinstance(raw_stock, str):
        raw_stock = str(raw_stock)
    sym = raw_stock.strip().lstrip("$").upper()
    hint_tickers: List[str] = []
    if sym and detector.is_valid_ticker(sym):
        hint_tickers = [sym]

    sid = _stable_id(url, headline)
    return {
        "data_source": "news",
        "source_name": publisher or "hf_financial_news",
        "source_id": sid,
        "title": headline,
        "published_at": _parse_published(ex.get("date") or ex.get("published_at")),
        "url": url[:500] if url else None,
        **({"hint_tickers": hint_tickers} if hint_tickers else {}),
    }


def _stream_examples(
    dataset_id: str,
    split: str,
) -> Iterator[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The `datasets` package is required. Install with: pip install datasets"
        ) from e

    logger.info("Opening streaming dataset %s (split=%s)", dataset_id, split)
    ds = load_dataset(dataset_id, split=split, streaming=True)
    for ex in ds:
        yield ex


def run_ingest(
    dataset_id: str,
    split: str,
    max_rows: int,
    flush_batch: int,
    store_db: bool,
) -> Dict[str, Any]:
    detector = TickerDetector.get_instance()
    service = ImportService()

    pending: List[Dict[str, Any]] = []
    total_seen = 0
    total_loaded = 0
    total_inserted = 0

    try:
        for ex in _stream_examples(dataset_id, split):
            if total_seen >= max_rows:
                break
            total_seen += 1
            rec = _row_to_record(ex, detector)
            if rec is None:
                continue
            pending.append(rec)
            total_loaded += 1

            if store_db and len(pending) >= flush_batch:
                r = service.import_from_records(pending)
                total_inserted += int(r.get("records_inserted", 0))
                pending.clear()
                if total_seen % (flush_batch * 5) == 0:
                    logger.info(
                        "Progress: seen=%d usable=%d inserted_rows=%d",
                        total_seen,
                        total_loaded,
                        total_inserted,
                    )

        if store_db and pending:
            r = service.import_from_records(pending)
            total_inserted += int(r.get("records_inserted", 0))
            pending.clear()
    except Exception:
        logger.exception("HF financial news ingest failed")
        raise

    summary = {
        "dataset": dataset_id,
        "rows_seen": total_seen,
        "rows_with_text": total_loaded,
        "records_inserted": total_inserted,
        "store_db": store_db,
    }
    logger.info("Done: %s", summary)
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF financial news → FinBERT → DB")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id")
    p.add_argument("--split", default="train", help="Dataset split")
    p.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Max streaming rows to scan (includes skipped short headlines)",
    )
    p.add_argument(
        "--flush-batch",
        type=int,
        default=200,
        help="Accumulate this many records per ImportService call",
    )
    p.add_argument(
        "--store-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist via ImportService (default True)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore[misc, assignment]

    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[3]
    if load_dotenv:
        load_dotenv(repo / ".env")
        load_dotenv(repo / "backend" / ".env")

    if not os.environ.get("DATABASE_URL", "").strip():
        logger.error("DATABASE_URL is not set")
        return 1

    try:
        run_ingest(
            dataset_id=args.dataset,
            split=args.split,
            max_rows=args.max_rows,
            flush_batch=args.flush_batch,
            store_db=args.store_db,
        )
        return 0
    except RuntimeError as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
