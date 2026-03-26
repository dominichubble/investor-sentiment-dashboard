#!/usr/bin/env python3
"""
Focused Reddit pipeline: collect → FinBERT (local GPU/CPU) → Neon Postgres.

Loads repo-root .env (REDDIT_*, DATABASE_URL). Chdir to backend for imports.

Defaults target a small ticker set and a handful of subs for in-depth evaluation.
Override tickers/subs via files or CLI.

Usage (from anywhere):
  python backend/scripts/focused_reddit_pipeline.py
  python backend/scripts/focused_reddit_pipeline.py --quick
  python backend/scripts/focused_reddit_pipeline.py --tickers NVDA PLTR --write-files
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Repo layout: .../investor-sentiment-dashboard/backend/scripts/this_file.py
_SCRIPT = Path(__file__).resolve()
_BACKEND_ROOT = _SCRIPT.parents[1]
_REPO_ROOT = _BACKEND_ROOT.parent

_DEFAULT_TICKERS = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD"]
_DEFAULT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
    "options",
]

_TICKERS_FILE = _REPO_ROOT / "data" / "config" / "focus_tickers.txt"
_SUBS_FILE = _REPO_ROOT / "data" / "config" / "focus_subreddits.txt"


def _bootstrap() -> None:
    os.chdir(_BACKEND_ROOT)
    if str(_BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(_BACKEND_ROOT))
    try:
        from dotenv import load_dotenv

        env_path = _REPO_ROOT / ".env"
        load_dotenv(env_path)
        if not env_path.exists():
            logging.warning("No .env at %s — set REDDIT_* and DATABASE_URL in environment", env_path)
    except ImportError:
        logging.warning("python-dotenv not installed; relying on existing environment variables")


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.split()[0])
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _bootstrap()

    p = argparse.ArgumentParser(
        description="Focused Reddit ingest + FinBERT + Neon (narrow tickers for evaluation)."
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Override ticker list (else focus_tickers.txt or built-in default)",
    )
    p.add_argument(
        "--tickers-file",
        type=Path,
        default=None,
        help=f"Ticker file (default: {_TICKERS_FILE})",
    )
    p.add_argument(
        "--subreddits",
        nargs="+",
        default=None,
        help="Override subreddit list (else focus_subreddits.txt or built-in default)",
    )
    p.add_argument(
        "--subreddits-file",
        type=Path,
        default=None,
        help=f"Subreddit file (default: {_SUBS_FILE})",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller limits for a fast sanity run on this PC",
    )
    p.add_argument(
        "--write-files",
        action="store_true",
        help="Also write JSON under data/artifacts/reddit_bulk/",
    )
    args = p.parse_args()

    tickers_file = args.tickers_file or _TICKERS_FILE
    subs_file = args.subreddits_file or _SUBS_FILE

    if args.tickers:
        tickers = args.tickers
    else:
        tickers = _read_lines(tickers_file)
        if not tickers:
            tickers = list(_DEFAULT_TICKERS)
            logging.info("Using built-in default tickers: %s", tickers)

    if args.subreddits:
        subreddits = args.subreddits
    else:
        subreddits = _read_lines(subs_file)
        if not subreddits:
            subreddits = list(_DEFAULT_SUBREDDITS)
            logging.info("Using built-in default subreddits (%d subs)", len(subreddits))

    if args.quick:
        limit_per_feed = 40
        search_limit = 35
        top_filters = ["day", "week"]
        search_time = "week"
        search_sorts = ["new", "relevance"]
        sleep_s = 0.35
    else:
        limit_per_feed = 300
        search_limit = 250
        top_filters = ["week", "month", "year"]
        search_time = "year"
        search_sorts = ["new", "relevance", "hot"]
        sleep_s = 1.0

    run_id = datetime.utcnow().strftime("%Y-%m-%d-%H%M") + "-focus"
    output_dir = _REPO_ROOT / "data" / "artifacts" / "reddit_bulk"

    if not os.environ.get("DATABASE_URL"):
        logging.error("DATABASE_URL is not set. Add it to %s", _REPO_ROOT / ".env")
        return 1
    if not os.environ.get("REDDIT_CLIENT_ID"):
        logging.error("REDDIT_CLIENT_ID not set — add Reddit API credentials to .env")
        return 1

    from app.pipelines.reddit_bulk_ingest import keyword_groups_for_tickers, run_bulk_ingest

    seen_sym: set[str] = set()
    deduped: List[str] = []
    for t in tickers:
        sym = t.strip().lstrip("$").upper()
        if not sym or sym in seen_sym:
            continue
        seen_sym.add(sym)
        deduped.append(t.strip())
    kw_groups = keyword_groups_for_tickers(deduped)
    if not kw_groups:
        logging.error("No valid tickers after parsing")
        return 1

    logging.info(
        "Focused run: %d tickers %s | %d subreddits | quick=%s",
        len(deduped),
        deduped,
        len(subreddits),
        args.quick,
    )

    summary = run_bulk_ingest(
        subreddits=subreddits,
        keyword_groups=kw_groups,
        limit_per_feed=limit_per_feed,
        top_time_filters=top_filters,
        search_time_filter=search_time,
        search_limit_per_group=search_limit,
        search_sorts=search_sorts,
        sleep_seconds=sleep_s,
        skip_search=False,
        store_db=True,
        write_files=args.write_files,
        output_dir=output_dir,
        run_id=run_id,
    )
    print(json.dumps(summary, indent=2))
    return 0 if not summary.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
