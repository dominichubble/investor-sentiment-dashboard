"""One-off migration: add richer FinBERT metadata columns to sentiment_records."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def migrate(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cols = [
            ("score_positive", "REAL"),
            ("score_negative", "REAL"),
            ("score_neutral", "REAL"),
            ("sentiment_uncertainty", "REAL"),
            ("rationale", "TEXT"),
            ("aspects_json", "TEXT"),
        ]
        for name, typ in cols:
            try:
                cur.execute(f"ALTER TABLE sentiment_records ADD COLUMN {name} {typ}")
                print(f"Added column {name}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    print(f"Skip {name} (exists)")
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
