"""One-time migration: delete legacy rows, drop unused columns, drop old table.

Run:
    cd backend
    python -m app.storage.migrate_drop_columns

Idempotent — safe to run multiple times.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

COLUMNS_TO_DROP = [
    "record_type",
    "document_id",
    "context",
    "position_start",
    "position_end",
    "sentiment_mode",
]


def _get_db_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "db" / "sentiments.db"


def _existing_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _table_exists(cur: sqlite3.Cursor, table: str) -> bool:
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def migrate(db_path: Path | None = None) -> None:
    db_path = db_path or _get_db_path()
    if not db_path.exists():
        logger.warning("Database not found at %s — nothing to migrate.", db_path)
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # 1. Delete legacy keyword-matched stock rows.
    cur.execute(
        "SELECT COUNT(*) FROM sentiment_records WHERE sentiment_mode = 'keyword'"
    )
    legacy_count = cur.fetchone()[0]
    if legacy_count:
        cur.execute("DELETE FROM sentiment_records WHERE sentiment_mode = 'keyword'")
        conn.commit()
        logger.info("Deleted %d legacy keyword rows.", legacy_count)
    else:
        logger.info("No legacy keyword rows to delete.")

    # 2. Drop unused columns (SQLite 3.35+).
    existing = _existing_columns(cur, "sentiment_records")
    for col in COLUMNS_TO_DROP:
        if col in existing:
            cur.execute(f"ALTER TABLE sentiment_records DROP COLUMN {col}")
            conn.commit()
            logger.info("Dropped column: %s", col)
        else:
            logger.info("Column already absent: %s", col)

    # 3. Drop legacy stock_sentiments table.
    if _table_exists(cur, "stock_sentiments"):
        cur.execute("DROP TABLE stock_sentiments")
        conn.commit()
        logger.info("Dropped legacy table stock_sentiments.")
    else:
        logger.info("Legacy table stock_sentiments already absent.")

    # 4. VACUUM to reclaim space.
    cur.execute("VACUUM")
    conn.commit()
    logger.info("VACUUM complete.")

    # Summary.
    cur.execute("SELECT COUNT(*) FROM sentiment_records")
    total = cur.fetchone()[0]
    remaining = _existing_columns(cur, "sentiment_records")
    logger.info("Done. %d rows remain. Columns: %s", total, sorted(remaining))

    conn.close()


if __name__ == "__main__":
    migrate()
