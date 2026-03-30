#!/usr/bin/env python3
"""Fetch tweets (API, snscrape, CSV, or auto) and write a plain-text file."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv

# Windows / some Python builds lack a default CA bundle for urllib3.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

# Repo root (…/investor-sentiment-dashboard)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _append_csv_sample(
    lines: list[str],
    csv_path: Path,
    limit: int,
) -> int:
    """Append up to ``limit`` rows from stock_tweets-style CSV. Returns rows written."""
    if not csv_path.is_file():
        lines.append(f"(No fallback file: {csv_path})")
        return 0
    n = 0
    with csv_path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if n >= limit:
                break
            sym = (row.get("Stock Name") or "").strip()
            dt = (row.get("Date") or "").strip()
            body = (row.get("Tweet") or "").replace("\n", " ").strip()
            lines.append(f"--- {n + 1}. ticker={sym} | date={dt} ---")
            lines.append(body)
            lines.append("")
            n += 1
    return n


def _collect_tweets(backend: str, max_results: int) -> tuple[list, str]:
    """Return (tweets, label describing source)."""
    from app.pipelines import ingest_twitter as tw

    PRIORITY_TICKERS = tw.PRIORITY_TICKERS

    mode = tw.resolve_ingest_backend(backend)
    if mode == "api":
        client = tw.initialize_twitter_client()
        tweets = tw.fetch_tweets_api(
            client,
            max_results=max_results,
            lang="en",
            min_engagement=0,
            tickers=PRIORITY_TICKERS,
            keywords=None,
        )
        return tweets, "official X API v2 (recent search)"
    if mode == "csv":
        tweets = tw.fetch_tweets_csv(
            max_results=max_results,
            lang="en",
            min_engagement=0,
            tickers=PRIORITY_TICKERS,
            keywords=None,
        )
        return tweets, "local CSV (TWITTER_CSV_PATH or stock_tweets.csv)"
    tweets = tw.fetch_tweets_snscrape(
        max_results=max_results,
        lang="en",
        min_engagement=0,
        tickers=PRIORITY_TICKERS,
        keywords=None,
    )
    return tweets, "snscrape (unofficial; often blocked)"


def main() -> None:
    load_dotenv(_REPO_ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["auto", "api", "snscrape", "csv"],
        default="auto",
        help="auto = API if TWITTER_BEARER_TOKEN set, else snscrape (default: auto)",
    )
    parser.add_argument("--max", type=int, default=25, help="Max tweets to collect")
    parser.add_argument(
        "--fallback-csv",
        type=Path,
        default=_REPO_ROOT / "stock_tweets.csv",
        help="Plain-text fallback if collection returns nothing",
    )
    parser.add_argument(
        "--fallback-rows",
        type=int,
        default=25,
        help="Max fallback CSV rows when primary collection is empty",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_REPO_ROOT / "data" / "artifacts" / "twitter" / "collected_tweets_local.txt",
        help="Output .txt path",
    )
    args = parser.parse_args()

    tweets, source_label = _collect_tweets(args.backend, args.max)

    from app.pipelines.ingest_twitter import PRIORITY_TICKERS

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "Twitter/X collection — local test export",
        f"Backend: {args.backend} → {source_label}",
        f"Priority tickers (when applicable): {', '.join(PRIORITY_TICKERS)}",
        f"Collected count (after filters): {len(tweets)}",
        "",
    ]

    if tweets:
        lines.append(f"=== {source_label} ===")
        lines.append("=" * 72)
        lines.append("")
        for i, t in enumerate(tweets, 1):
            body = (t.get("raw_text") or t.get("text") or "").strip()
            hints = t.get("hint_tickers") or []
            lines.append(f"--- {i}. id={t.get('id')} | tickers={hints} ---")
            lines.append(body)
            lines.append("")
    else:
        lines.append(
            "Primary collection returned no rows (API/snscrape error, filters, or empty CSV). "
            "Try --backend csv with TWITTER_CSV_PATH, or set TWITTER_BEARER_TOKEN for --backend api."
        )
        lines.append("")
        lines.append(f"=== Fallback sample from CSV ({args.fallback_csv.name}) ===")
        lines.append("=" * 72)
        lines.append("")
        n_csv = _append_csv_sample(lines, args.fallback_csv, args.fallback_rows)
        lines[3] = (
            f"Collected count (after filters): {len(tweets)} | "
            f"CSV fallback rows: {n_csv}"
        )

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
