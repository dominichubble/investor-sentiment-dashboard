"""Add data_source column (reddit | news | twitter) to sentiment_records."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def migrate(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "ALTER TABLE sentiment_records ADD COLUMN data_source VARCHAR(20)"
            )
            print("Added column data_source")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print("Skip data_source (exists)")
            else:
                raise
        # Composite index for filtering by platform over time
        try:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_sentiment_records_data_source_timestamp "
                "ON sentiment_records (data_source, timestamp)"
            )
            print("Created index ix_sentiment_records_data_source_timestamp")
        except sqlite3.OperationalError as e:
            print(f"Index note: {e}")
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    db = root / "data" / "db" / "sentiments.db"
    if not db.exists():
        print(f"No database at {db}", file=sys.stderr)
        return 1
    migrate(db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
