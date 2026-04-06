#!/usr/bin/env python3
"""
Delete the oldest sentiment_records (by published_at, then id) to free Neon space.

Usage (from backend/, DATABASE_URL in .env):
  python scripts/prune_sentiment_oldest.py --count 10000
  python scripts/prune_sentiment_oldest.py --count 5000 --dry-run
  python scripts/prune_sentiment_oldest.py --count 2000 --data-source news

Auto-prune during ingest (optional):
  set SENTIMENT_AUTO_PRUNE_ON_STORAGE_ERROR=1
  optional SENTIMENT_AUTO_PRUNE_BATCH (default 5000)
"""

from __future__ import annotations

import argparse
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


def _preview_sql(where_sql: str) -> str:
    return f"""
    SELECT COUNT(*)::int AS cnt,
           MIN(published_at) AS oldest,
           MAX(published_at) AS newest_in_slice
    FROM (
        SELECT published_at FROM sentiment_records
        WHERE {where_sql}
        ORDER BY published_at ASC, id ASC
        LIMIT :lim
    ) sub
    """


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune oldest sentiment_records by published_at."
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Maximum rows to delete (oldest published_at first).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows would be removed and the published_at range.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Only consider rows with this source column value.",
    )
    parser.add_argument(
        "--data-source",
        default=None,
        dest="data_source",
        help="Only consider rows with this data_source column value.",
    )
    args = parser.parse_args()

    if args.count <= 0:
        print("--count must be positive.", file=sys.stderr)
        return 2

    repo = _backend.parent
    if load_dotenv:
        load_dotenv(repo / ".env")
        load_dotenv(_backend / ".env")

    from app.storage.database import get_engine

    parts = ["1=1"]
    params: dict[str, object] = {"lim": args.count}
    if args.source is not None:
        parts.append("source = :source")
        params["source"] = args.source
    if args.data_source is not None:
        parts.append("data_source = :data_source")
        params["data_source"] = args.data_source
    where_sql = " AND ".join(parts)

    engine = get_engine()
    preview = text(_preview_sql(where_sql))

    with engine.connect() as conn:
        row = conn.execute(preview, params).mappings().first()
        assert row is not None
        cnt = int(row["cnt"] or 0)
        oldest = row["oldest"]
        newest_in_slice = row["newest_in_slice"]
        print(
            f"Would affect up to {cnt} row(s); slice published_at from {oldest} to {newest_in_slice}."
        )

    if args.dry_run:
        return 0

    from app.storage.sqlite_storage import SentimentStorage

    storage = SentimentStorage()
    deleted = storage.prune_oldest_published(
        limit=args.count,
        source=args.source,
        data_source=args.data_source,
    )
    print(f"Deleted {deleted} row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
