#!/usr/bin/env python3
"""
Run high-volume ingestion back-to-back until a wall-clock budget elapses.

Each step loads credentials from the repo-root ``.env`` and runs FinBERT inside
``ImportService`` when writing to Neon.

Default round (high row count):
  1. Reddit bulk (expanded sub list + keyword groups + listings/search)
  2. Hugging Face ``ashraq/financial-news`` stream (headlines + ticker hints)
  3. NewsAPI
  4. Optional Twitter (skipped by default: snscrape breaks on Python 3.13)

Usage:
  cd backend
  python scripts/hour_ingest_loop.py --seconds 3600
  python scripts/hour_ingest_loop.py --seconds 7200 --hf-max-rows 15000 --twitter
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="High-volume multi-source ingest loop")
    parser.add_argument(
        "--seconds",
        type=int,
        default=3600,
        help="Stop starting new rounds after this many seconds (default 3600)",
    )
    parser.add_argument(
        "--hf-max-rows",
        type=int,
        default=8000,
        help="HF financial-news rows to scan per round (default 8000)",
    )
    parser.add_argument(
        "--hf-flush-batch",
        type=int,
        default=250,
        help="Records per ImportService batch for HF ingest (default 250)",
    )
    parser.add_argument(
        "--reddit-limit-feed",
        type=int,
        default=320,
        help="reddit_bulk_ingest --limit-per-feed (default 320)",
    )
    parser.add_argument(
        "--reddit-search-limit",
        type=int,
        default=130,
        help="reddit_bulk_ingest --search-limit (default 130)",
    )
    parser.add_argument(
        "--reddit-sleep",
        type=float,
        default=0.65,
        help="Seconds between Reddit API blocks (default 0.65)",
    )
    parser.add_argument(
        "--reddit-search-time-filter",
        default="month",
        choices=["hour", "day", "week", "month", "year", "all"],
        help="Reddit search time window (default month)",
    )
    parser.add_argument(
        "--news-max-articles",
        type=int,
        default=280,
        help="NewsAPI max articles per round (default 280)",
    )
    parser.add_argument(
        "--news-days-back",
        type=int,
        default=3,
        help="NewsAPI lookback days (default 3)",
    )
    parser.add_argument(
        "--twitter-max",
        type=int,
        default=150,
        help="Twitter max tweets when --twitter is set (default 150)",
    )
    parser.add_argument(
        "--twitter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Twitter ingest (default False; use CSV/API backends on 3.13)",
    )
    parser.add_argument(
        "--skip-hf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip Hugging Face financial-news step",
    )
    args = parser.parse_args()

    backend = Path(__file__).resolve().parents[1]
    repo_root = backend.parent
    os.chdir(backend)
    os.environ["PYTHONPATH"] = str(backend)

    try:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env")
        load_dotenv(backend / ".env")
    except ImportError:
        pass

    py = sys.executable
    deadline = time.time() + args.seconds
    roundn = 0

    while time.time() < deadline:
        roundn += 1
        remaining = int(max(0, deadline - time.time()))
        print(
            f"\n{'=' * 60}\nRound {roundn} (~{remaining}s remaining)\n{'=' * 60}\n",
            flush=True,
        )

        subprocess.run(
            [
                py,
                "-m",
                "app.pipelines.reddit_bulk_ingest",
                "--sleep-seconds",
                str(args.reddit_sleep),
                "--limit-per-feed",
                str(args.reddit_limit_feed),
                "--search-limit",
                str(args.reddit_search_limit),
                "--search-time-filter",
                args.reddit_search_time_filter,
                "--store-db",
                "--no-write-files",
            ],
            check=False,
        )
        if time.time() >= deadline:
            break

        if not args.skip_hf:
            subprocess.run(
                [
                    py,
                    "-m",
                    "app.pipelines.ingest_huggingface_financial_news",
                    "--max-rows",
                    str(args.hf_max_rows),
                    "--flush-batch",
                    str(args.hf_flush_batch),
                    "--store-db",
                ],
                check=False,
            )
        if time.time() >= deadline:
            break

        subprocess.run(
            [
                py,
                "-m",
                "app.pipelines.ingest_news",
                "--days-back",
                str(args.news_days_back),
                "--max-articles",
                str(args.news_max_articles),
                "--store-db",
                "--no-write-files",
            ],
            check=False,
        )
        if time.time() >= deadline:
            break

        subprocess.run(
            [
                py,
                "-m",
                "app.pipelines.ingest_reddit",
                "--time-filter",
                "month",
                "--limit",
                "55",
                "--search-sorts",
                "new",
                "relevance",
                "hot",
                "--store-db",
                "--no-write-files",
            ],
            check=False,
        )
        if time.time() >= deadline:
            break

        if args.twitter:
            subprocess.run(
                [
                    py,
                    "-m",
                    "app.pipelines.ingest_twitter",
                    "--max-tweets",
                    str(args.twitter_max),
                    "--store-db",
                    "--no-write-files",
                ],
                check=False,
            )

    print("\nIngest loop finished (time budget elapsed).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
