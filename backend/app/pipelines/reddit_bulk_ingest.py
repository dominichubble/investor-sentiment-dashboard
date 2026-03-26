#!/usr/bin/env python3
"""
High-volume Reddit collection for sentiment DB.

Pulls far more posts than the daily workflow by combining:
  - /new, /hot, /rising
  - /top for several time windows
  - Multiple keyword search batches (OR within each batch)

Respects Reddit OAuth rate limits via configurable pauses between blocks.
Use a descriptive REDDIT_USER_AGENT and stay under ~60 requests/minute.

Usage (from repo root, PYTHONPATH=backend):
  python -m app.pipelines.reddit_bulk_ingest --store-db --no-write-files
  python -m app.pipelines.reddit_bulk_ingest --limit-per-feed 300 --sleep-seconds 1.2
  python -m app.pipelines.reddit_bulk_ingest --subreddits-file path/to/subs.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from praw.exceptions import PRAWException, RedditAPIException

from app.pipelines.ingest_reddit import (
    build_query,
    initialize_reddit_client,
    normalize_post,
)
from app.services.import_service import ImportService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Finance / markets ecosystem (add/remove as needed; some subs may 403).
DEFAULT_BULK_SUBREDDITS: List[str] = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
    "options",
    "pennystocks",
    "ValueInvesting",
    "SecurityAnalysis",
    "dividends",
    "FinancialIndependence",
    "Economics",
    "SPACs",
    "CanadianInvestor",
    "UKInvesting",
    "IndianStreetBets",
    "cryptocurrency",
    "Bitcoin",
    "ethereum",
    "Superstonk",
    "amcstock",
    "GME",
    "thetagang",
    "Daytrading",
    "SwingTrading",
    "RealEstateInvesting",
    "Bogleheads",
    "personalfinance",
    "CryptoMarkets",
]

# Smaller OR-queries = different search result sets than one giant OR.
DEFAULT_KEYWORD_GROUPS: List[List[str]] = [
    ["stock", "stocks", "market", "trading", "investor"],
    ["earnings", "eps", "revenue", "guidance", "quarter"],
    ["fed", "rates", "inflation", "recession", "tariff"],
    ["nvda", "tsla", "aapl", "msft", "amzn", "meta", "googl", "amd"],
    ["bull", "bear", "rally", "crash", "dip", "short", "long"],
    ["crypto", "bitcoin", "btc", "eth", "etf"],
    ["bankruptcy", "debt", "lawsuit", "sec", "merger"],
]


def _sleep(seconds: float, reason: str) -> None:
    if seconds > 0:
        logger.debug("Sleep %.2fs (%s)", seconds, reason)
        time.sleep(seconds)


def _safe_collect(
    label: str,
    iterator_factory: Callable[[], Iterable[Any]],
    add: Callable[[Any], None],
    sleep_after: float,
) -> int:
    n = 0
    try:
        for item in iterator_factory():
            add(item)
            n += 1
    except (PRAWException, RedditAPIException) as e:
        logger.warning("%s failed: %s", label, e)
    _sleep(sleep_after, f"after {label}")
    return n


def fetch_subreddit_bulk(
    sr_name: str,
    reddit: Any,
    *,
    limit_per_feed: int,
    top_time_filters: List[str],
    search_time_filter: str,
    search_limit_per_group: int,
    keyword_groups: List[List[str]],
    sleep_seconds: float,
    skip_search: bool,
) -> List[Dict[str, Any]]:
    """Collect unique submissions from r/sr_name using multiple feeds + searches."""
    sr = reddit.subreddit(sr_name)
    seen: Set[str] = set()
    rows: List[Dict[str, Any]] = []

    def add_submission(sub: Any) -> None:
        sid = sub.id
        if sid in seen:
            return
        seen.add(sid)
        rows.append(normalize_post(sub))

    _safe_collect(
        f"r/{sr_name} new",
        lambda: sr.new(limit=limit_per_feed),
        add_submission,
        sleep_seconds,
    )
    _safe_collect(
        f"r/{sr_name} hot",
        lambda: sr.hot(limit=limit_per_feed),
        add_submission,
        sleep_seconds,
    )
    for tf in top_time_filters:
        _safe_collect(
            f"r/{sr_name} top:{tf}",
            lambda tf=tf: sr.top(time_filter=tf, limit=limit_per_feed),
            add_submission,
            sleep_seconds,
        )
    _safe_collect(
        f"r/{sr_name} rising",
        lambda: sr.rising(limit=min(100, limit_per_feed)),
        add_submission,
        sleep_seconds,
    )

    if not skip_search:
        for group in keyword_groups:
            if not group:
                continue
            q = build_query(group)
            _safe_collect(
                f"r/{sr_name} search({len(group)} terms)",
                lambda q=q: sr.search(
                    q,
                    sort="new",
                    time_filter=search_time_filter,
                    limit=search_limit_per_group,
                ),
                add_submission,
                sleep_seconds,
            )

    logger.info("r/%s → %d unique posts", sr_name, len(rows))
    return rows


def run_bulk_ingest(
    subreddits: List[str],
    keyword_groups: List[List[str]],
    limit_per_feed: int,
    top_time_filters: List[str],
    search_time_filter: str,
    search_limit_per_group: int,
    sleep_seconds: float,
    skip_search: bool,
    store_db: bool,
    write_files: bool,
    output_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    reddit = initialize_reddit_client()
    all_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for name in subreddits:
        name = name.strip().lstrip("r/").lower()
        if not name:
            continue
        try:
            chunk = fetch_subreddit_bulk(
                name,
                reddit,
                limit_per_feed=limit_per_feed,
                top_time_filters=top_time_filters,
                search_time_filter=search_time_filter,
                search_limit_per_group=search_limit_per_group,
                keyword_groups=keyword_groups,
                sleep_seconds=sleep_seconds,
                skip_search=skip_search,
            )
            all_rows.extend(chunk)
        except Exception as e:
            logger.error("r/%s: %s", name, e)
            errors.append(f"r/{name}: {e}")
        _sleep(sleep_seconds, f"between subreddits ({name})")

    by_id = {r["id"]: r for r in all_rows}
    deduped = sorted(by_id.values(), key=lambda x: x.get("created_utc", 0), reverse=True)

    summary = {
        "run_id": run_id,
        "subreddits_tried": len(subreddits),
        "raw_rows": len(all_rows),
        "unique_posts": len(deduped),
        "errors": errors,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    logger.info(
        "Collected %d unique posts from %d subreddit passes (raw %d)",
        len(deduped),
        len(subreddits),
        len(all_rows),
    )

    inserted = 0
    if store_db and deduped:
        res = ImportService().import_from_records(deduped)
        inserted = res["records_inserted"]
        logger.info(
            "DB import: loaded=%s inserted=%s",
            res["records_loaded"],
            inserted,
        )

    if write_files and deduped:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_json = output_dir / f"reddit_bulk_{run_id}.json"
        meta_json = output_dir / f"reddit_bulk_{run_id}_meta.json"
        out_json.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote %s and %s", out_json, meta_json)

    summary["records_inserted"] = inserted
    return summary


def _load_subreddits_file(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk Reddit → SQLite sentiment ingest")
    p.add_argument(
        "--subreddits",
        nargs="+",
        default=None,
        help="Subreddit names (without r/). Default: built-in finance list.",
    )
    p.add_argument(
        "--subreddits-file",
        type=Path,
        default=None,
        help="File with one subreddit per line (# comments ok)",
    )
    p.add_argument(
        "--limit-per-feed",
        type=int,
        default=400,
        help="Max posts per listing (new/hot/top); PRAW paginates (default 400)",
    )
    p.add_argument(
        "--top-filters",
        nargs="+",
        default=["day", "week", "month"],
        choices=["hour", "day", "week", "month", "year", "all"],
        help="Time filters for /top passes",
    )
    p.add_argument(
        "--search-time-filter",
        default="month",
        choices=["hour", "day", "week", "month", "year", "all"],
        help="time_filter for search()",
    )
    p.add_argument(
        "--search-limit",
        type=int,
        default=200,
        help="Max results per keyword-group search (default 200)",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Pause after each API block (reduce if 429; increase if throttled)",
    )
    p.add_argument(
        "--skip-search",
        action="store_true",
        help="Only use new/hot/top/rising (no search queries)",
    )
    p.add_argument(
        "--store-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ImportService (default True)",
    )
    p.add_argument(
        "--write-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write JSON dump under --output",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/artifacts/reddit_bulk"),
        help="Directory for optional JSON export",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Run id for filenames (default UTC date)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or datetime.utcnow().strftime("%Y-%m-%d-%H%M")

    if args.subreddits_file:
        subs = _load_subreddits_file(args.subreddits_file)
    elif args.subreddits:
        subs = args.subreddits
    else:
        subs = list(DEFAULT_BULK_SUBREDDITS)

    if not subs:
        logger.error("No subreddits to fetch")
        return 1

    logger.info(
        "Bulk ingest: %d subreddits, limit_per_feed=%d, sleep=%.2fs, search=%s",
        len(subs),
        args.limit_per_feed,
        args.sleep_seconds,
        "off" if args.skip_search else "on",
    )

    try:
        summary = run_bulk_ingest(
            subreddits=subs,
            keyword_groups=DEFAULT_KEYWORD_GROUPS,
            limit_per_feed=args.limit_per_feed,
            top_time_filters=list(args.top_filters),
            search_time_filter=args.search_time_filter,
            search_limit_per_group=args.search_limit,
            sleep_seconds=args.sleep_seconds,
            skip_search=args.skip_search,
            store_db=args.store_db,
            write_files=args.write_files,
            output_dir=args.output,
            run_id=run_id,
        )
        logger.info("Done: %s", json.dumps(summary, indent=2)[:2000])
        return 0 if not summary.get("errors") else 0
    except Exception as e:
        logger.exception("Bulk ingest failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
