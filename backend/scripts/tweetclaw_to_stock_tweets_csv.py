#!/usr/bin/env python3
"""Convert TweetClaw exports into the repository's stock_tweets CSV shape."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TEXT_FIELDS = (
    "text",
    "tweet",
    "Tweet",
    "tweet_text",
    "tweetText",
    "full_text",
    "content",
    "body",
    "summary",
    "title",
)
DATE_FIELDS = ("created_at", "createdAt", "published_at", "timestamp", "date")
TICKER_FIELDS = (
    "ticker",
    "symbol",
    "cashtag",
    "stock",
    "stock_name",
    "Stock Name",
)
LIST_FIELDS = ("tweets", "data", "results", "items", "records")
OUTPUT_FIELDS = ("Date", "Tweet", "Stock Name", "Company Name")
CASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])\$([A-Za-z]{1,6})(?![A-Za-z0-9_])")


def _first_scalar(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = row.get(field)
        if value is None or isinstance(value, (dict, list)):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_ticker(row: dict[str, Any], text: str, fallback: str) -> str:
    raw = _first_scalar(row, TICKER_FIELDS)
    if raw:
        return raw.upper().lstrip("$#")
    match = CASHTAG_RE.search(text)
    if match:
        return match.group(1).upper()
    return fallback.upper().lstrip("$#")


def _normalize_date(raw: str) -> str:
    if not raw:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    cleaned = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return raw
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _records_from_json(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return
    for field in LIST_FIELDS:
        value = payload.get(field)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
            return
    yield payload


def read_tweetclaw_export(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            yield from csv.DictReader(handle)
        return
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield from _records_from_json(payload)
        return
    message = "TweetClaw export must be .json, .jsonl, .ndjson, or .csv"
    raise ValueError(message)


def convert_rows(
    input_path: Path,
    fallback_ticker: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in read_tweetclaw_export(input_path):
        text = _first_scalar(record, TEXT_FIELDS)
        if not text:
            continue
        rows.append(
            {
                "Date": _normalize_date(_first_scalar(record, DATE_FIELDS)),
                "Tweet": text.replace("\r", " ").replace("\n", " ").strip(),
                "Stock Name": _extract_ticker(record, text, fallback_ticker),
                "Company Name": "",
            }
        )
    return rows


def write_stock_tweets_csv(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TweetClaw JSON, JSONL, NDJSON, or CSV exports."
    )
    parser.add_argument("input", type=Path, help="TweetClaw export path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("stock_tweets.csv"),
        help="Output CSV path for the local Twitter CSV backend",
    )
    parser.add_argument(
        "--fallback-ticker",
        default="",
        help="Ticker to use when a row has no ticker, symbol, or cashtag",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = convert_rows(args.input, args.fallback_ticker)
    write_stock_tweets_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
