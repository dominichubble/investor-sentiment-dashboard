#!/usr/bin/env python3
"""Import local raw/processed datasets into SQLite with FinBERT sentiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.import_service import ImportService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import records into SQLite database")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Path to the data root containing raw/ and processed/",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "raw", "processed"],
        default="all",
        help="Which local data folders to import",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = ImportService()

    include_raw = args.mode in {"all", "raw"}
    include_processed = args.mode in {"all", "processed"}
    result = service.import_from_data_dirs(
        data_root=args.data_root,
        include_raw=include_raw,
        include_processed=include_processed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

