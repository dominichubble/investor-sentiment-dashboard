#!/usr/bin/env python3
"""
``StephanAkkerman/financial-tweets-stocks`` → FinBERT → Neon.

~28k rows: ``description`` (tweet text), ``timestamp``, ``url``, optional
``financial_info`` (ticker metadata as a stringified list). Dataset includes a
``sentiment`` label; this pipeline **recomputes** sentiment with FinBERT for
consistency with the rest of ``sentiment_records``.

Rows without any resolvable ticker are **skipped** (no ``ticker=NULL`` rows).
Use ``--document-fallback`` only if you explicitly want document-level rows.

Usage (PYTHONPATH=backend):
  python -m app.pipelines.ingest_huggingface_financial_tweets --store-db
  python -m app.pipelines.ingest_huggingface_financial_tweets --max-rows 500 --store-db
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.pipelines.ingest_news import clean_text
from app.services.import_service import ImportService
from app.utils.ticker_detection import TickerDetector, normalize_ticker_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_ID = "StephanAkkerman/financial-tweets-stocks"

# financial_info is usually a Python repr; literal_eval can fail on edge cases.
_FIN_TICKER_KV_RE = re.compile(
    r"""['\"]ticker['\"]\s*:\s*['\"]?\$?([A-Za-z][A-Za-z0-9.-]{0,7})""",
    re.IGNORECASE,
)


def _parse_financial_info_tickers(raw: Any, detector: TickerDetector) -> List[str]:
    """Tickers from structured financial_info + regex fallbacks on the raw string."""
    out: List[str] = []
    seen: set[str] = set()

    def add(sym: str) -> None:
        n = normalize_ticker_symbol(sym)
        if not n or n in seen or not detector.is_valid_ticker(n):
            return
        seen.add(n)
        out.append(n)

    if raw is None:
        return out

    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, dict):
                add(str(it.get("ticker", "")))
        return out

    if not isinstance(raw, str):
        return out

    s = raw.strip()
    if not s:
        return out

    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, list):
        for it in parsed:
            if isinstance(it, dict):
                add(str(it.get("ticker", "")))

    for m in _FIN_TICKER_KV_RE.finditer(s):
        add(m.group(1))
    for m in re.finditer(
        r"\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)\b", s, flags=re.IGNORECASE
    ):
        add(m.group(1))

    return out


def _tweet_id_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"/status/(\d+)", str(url))
    return m.group(1) if m else None


def _parse_timestamp(raw: Any) -> str:
    if raw is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    s = str(raw).strip()
    if not s:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except ValueError:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _row_to_record(ex: Dict[str, Any], detector: TickerDetector) -> Optional[Dict[str, Any]]:
    body = ex.get("description") or ""
    if not isinstance(body, str):
        body = str(body)
    body = clean_text(body)
    if len(body) < 5:
        return None

    title = ex.get("embed_title") or ""
    if isinstance(title, str) and len(title.strip()) > 8:
        title = clean_text(title)
        text = f"{body}\n\n[{title}]" if title else body
    else:
        text = body

    url = str(ex.get("url") or "")
    tid = _tweet_id_from_url(url)
    if tid:
        source_id = tid
    else:
        source_id = hashlib.sha256(f"{url}\n{text[:160]}".encode("utf-8", errors="ignore")).hexdigest()[:32]

    hints = _parse_financial_info_tickers(ex.get("financial_info"), detector)

    return {
        "data_source": "twitter",
        "source": "hf_financial_tweets",
        "source_id": source_id,
        "text": text,
        "published_at": _parse_timestamp(ex.get("timestamp")),
        "url": url[:500] if url else None,
        **({"hint_tickers": hints} if hints else {}),
    }


def run_ingest(
    max_rows: Optional[int],
    flush_batch: int,
    store_db: bool,
    document_fallback_without_ticker: bool,
) -> Dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install datasets: pip install datasets") from e

    logger.info("Loading %s (may download ~26MB on first run)", DATASET_ID)
    ds = load_dataset(DATASET_ID, split="train")
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    detector = TickerDetector.get_instance()
    service = ImportService()

    pending: List[Dict[str, Any]] = []
    total_loaded = 0
    total_inserted = 0

    for i, ex in enumerate(ds):
        rec = _row_to_record(ex, detector)
        if rec is None:
            continue
        pending.append(rec)
        total_loaded += 1

        if store_db and len(pending) >= flush_batch:
            r = service.import_from_records(
                pending,
                document_fallback_without_ticker=document_fallback_without_ticker,
            )
            total_inserted += int(r.get("records_inserted", 0))
            pending.clear()
            if (i + 1) % (flush_batch * 10) == 0:
                logger.info(
                    "Progress: scanned=%d usable=%d inserted_rows=%d",
                    i + 1,
                    total_loaded,
                    total_inserted,
                )

    if store_db and pending:
        r = service.import_from_records(
            pending,
            document_fallback_without_ticker=document_fallback_without_ticker,
        )
        total_inserted += int(r.get("records_inserted", 0))

    summary = {
        "dataset": DATASET_ID,
        "rows_scanned": len(ds),
        "rows_with_text": total_loaded,
        "records_inserted": total_inserted,
        "store_db": store_db,
        "document_fallback_without_ticker": document_fallback_without_ticker,
    }
    logger.info("Done: %s", summary)
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF financial tweets → FinBERT → DB")
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap rows (default: all ~28k)",
    )
    p.add_argument("--flush-batch", type=int, default=200)
    p.add_argument(
        "--store-db",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--document-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, store rows with no ticker as document-level (ticker NULL). Default False.",
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
            max_rows=args.max_rows,
            flush_batch=args.flush_batch,
            store_db=args.store_db,
            document_fallback_without_ticker=args.document_fallback,
        )
        return 0
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
