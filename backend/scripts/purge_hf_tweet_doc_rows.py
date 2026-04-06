#!/usr/bin/env python3
"""
Remove document-level (ticker NULL) rows created for Hugging Face financial tweets.

Safe filter: data_source=twitter AND source=hf_financial_tweets AND ticker IS NULL.

Usage (from backend/, DATABASE_URL in .env):
  python scripts/purge_hf_tweet_doc_rows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

from sqlalchemy import text


def main() -> int:
    repo = _backend.parent
    if load_dotenv:
        load_dotenv(repo / ".env")
        load_dotenv(_backend / ".env")

    from app.storage.database import get_engine

    engine = get_engine()
    sql = text(
        """
        DELETE FROM sentiment_records
        WHERE ticker IS NULL
          AND data_source = :ds
          AND source = :src
        """
    )
    with engine.begin() as conn:
        result = conn.execute(
            sql, {"ds": "twitter", "src": "hf_financial_tweets"}
        )
        n = result.rowcount
    print(f"Deleted {n} rows (ticker NULL, hf_financial_tweets).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
