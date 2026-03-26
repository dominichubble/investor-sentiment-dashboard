"""Add source_meta_json for Reddit/Twitter/News extra fields (permalink, score, etc.)."""

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
                "ALTER TABLE sentiment_records ADD COLUMN source_meta_json TEXT"
            )
            print("Added column source_meta_json")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print("Skip source_meta_json (exists)")
            else:
                raise
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
