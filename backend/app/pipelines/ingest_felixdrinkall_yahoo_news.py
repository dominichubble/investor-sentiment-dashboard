#!/usr/bin/env python3
"""
Ingest **Felix Drinkall** Yahoo Finance news JSON (2017–2023) → FinBERT → Neon.

Source: https://github.com/FelixDrinkall/financial-news-dataset  
(~51k articles, ``.json.xz`` per year). **CC BY-NC-SA 4.0** — non-commercial;
ensure your use complies with the license.

Streams each yearly file with ``ijson`` + ``lzma`` (no full-file JSON load).

By default, **newer years are ingested first** (2023 → 2017) so when storage is
limited, recent news fills the database before older archives.

Usage (clone or download ``data/*.json.xz`` into a folder):

  pip install ijson   # listed in backend/requirements.txt
  python -m app.pipelines.ingest_felixdrinkall_yahoo_news \\
    --archive-dir /path/to/financial-news-dataset/data --store-db

Download archives into ``./data/external/felix_yahoo_news`` then ingest:

  python -m app.pipelines.ingest_felixdrinkall_yahoo_news --download --store-db

Environment: ``DATABASE_URL``, repo-root ``.env`` optional.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import lzma
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from app.pipelines.ingest_news import clean_text
from app.services.import_service import ImportService
from app.utils.ticker_detection import TickerDetector, normalize_ticker_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_BASE = (
    "https://raw.githubusercontent.com/FelixDrinkall/financial-news-dataset/main/data"
)
DEFAULT_ARCHIVE_SUBDIR = "data/external/felix_yahoo_news"


def _year_from_archive_name(path: Path) -> int:
    m = re.match(r"(\d{4})_processed\.json\.xz$", path.name)
    return int(m.group(1)) if m else 0


def _default_archive_dir() -> Path:
    repo = Path(__file__).resolve().parents[3]
    return repo / DEFAULT_ARCHIVE_SUBDIR


def _hint_tickers_from_article(
    art: Dict[str, Any],
    detector: TickerDetector,
    *,
    include_related: bool,
    max_related: int,
) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        if raw is None:
            return
        sym = normalize_ticker_symbol(str(raw).strip())
        if not sym or sym in seen or not detector.is_valid_ticker(sym):
            return
        seen.add(sym)
        ordered.append(sym)

    for x in art.get("mentioned_companies") or []:
        add(x)

    for ne in art.get("named_entities") or []:
        if not isinstance(ne, dict):
            continue
        add(ne.get("company_key"))
        add(ne.get("normalized"))

    if include_related:
        for x in (art.get("related_companies") or [])[:max_related]:
            add(x)

    return ordered


def _parse_published(raw: Any) -> str:
    from datetime import datetime, timezone

    if raw is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    s = str(raw).strip()
    if not s:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    s = s.replace("Z", "+00:00")
    if "+" not in s[-6:] and s.count(":") >= 2:
        try:
            dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            return dt.isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except ValueError:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_source_id(url: str, filename: str) -> str:
    base = (url or filename or "")[:800]
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()[:32]


def article_to_record(
    art: Dict[str, Any],
    detector: TickerDetector,
    *,
    english_only: bool,
    include_related: bool,
    max_related: int,
) -> Optional[Dict[str, Any]]:
    if english_only:
        lang = art.get("language")
        if lang is not None and str(lang).lower() not in ("en", "eng"):
            return None

    title = clean_text(str(art.get("title") or art.get("title_rss") or ""))
    body = clean_text(str(art.get("maintext") or art.get("description") or ""))
    if not title and not body:
        return None
    if len(title) < 6 and len(body) < 30:
        return None

    if not title:
        title = body[:240] + ("…" if len(body) > 240 else "")
    # Cap body length for FinBERT / DB pragmatism (~8k chars)
    if len(body) > 12000:
        body = body[:12000] + "…"

    url = str(art.get("url") or "")
    fn = str(art.get("filename") or "")
    source_id = _stable_source_id(url, fn)

    outlet = art.get("news_outlet") or art.get("source_domain") or "yahoo_finance"
    if not isinstance(outlet, str):
        outlet = str(outlet)
    outlet = re.sub(r"\s+", " ", outlet).strip()[:80]

    hints = _hint_tickers_from_article(
        art,
        detector,
        include_related=include_related,
        max_related=max_related,
    )

    published = _parse_published(
        art.get("date_publish") or art.get("date_download")
    )

    rec: Dict[str, Any] = {
        "data_source": "news",
        "source_name": outlet,
        "source_id": source_id,
        "title": title,
        "selftext": body,
        "published_at": published,
        "url": url[:500] if url else None,
    }
    if hints:
        rec["hint_tickers"] = hints
    return rec


def _iter_articles_json_array(path: Path) -> Iterator[Dict[str, Any]]:
    """Decompress each yearly file and parse the top-level JSON array.

    Upstream files sometimes contain ``NaN`` / ``Infinity`` (invalid in strict
    JSON). Those tokens are replaced with ``null`` before parsing.
    """
    try:
        import ijson
    except ImportError as e:
        raise RuntimeError(
            "Install ijson to stream large JSON arrays: pip install ijson"
        ) from e

    raw = lzma.open(path, "rb").read()
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"\bNaN\b", "null", text)
    text = re.sub(r"\b-?Infinity\b", "null", text)
    logger.info(
        "Decompressed %s (~%d MB text, sanitizing NaN→null)",
        path.name,
        max(1, len(text) // (1024 * 1024)),
    )
    buf = io.BytesIO(text.encode("utf-8"))
    for obj in ijson.items(buf, "item"):
        if isinstance(obj, dict):
            yield obj


def download_year_files(dest: Path, years: Optional[List[int]] = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    want = set(years) if years else set(range(2017, 2024))
    # Newest year first so partial downloads / interrupted runs favor recent data.
    for y in sorted(want, reverse=True):
        name = f"{y}_processed.json.xz"
        url = f"{RAW_BASE}/{name}"
        out = dest / name
        if out.is_file() and out.stat().st_size > 0:
            logger.info("Skip existing %s", out.name)
            continue
        logger.info("Downloading %s", url)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "investor-sentiment-dashboard/1.0 (research ingest)"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            out.write_bytes(resp.read())


def run_ingest(
    archive_dir: Path,
    *,
    max_articles: Optional[int],
    flush_batch: int,
    store_db: bool,
    english_only: bool,
    include_related: bool,
    max_related: int,
    year_filter: Optional[set[int]],
    newest_year_first: bool,
) -> Dict[str, Any]:
    detector = TickerDetector.get_instance()
    service = ImportService()

    paths: List[Path] = []
    for p in archive_dir.glob("*_processed.json.xz"):
        y = _year_from_archive_name(p)
        if y == 0:
            continue
        if year_filter is not None and y not in year_filter:
            continue
        paths.append(p)

    paths.sort(key=_year_from_archive_name, reverse=newest_year_first)
    logger.info(
        "Year ingest order: %s",
        " → ".join(str(_year_from_archive_name(p)) for p in paths),
    )

    if not paths:
        raise FileNotFoundError(
            f"No *_processed.json.xz files under {archive_dir}. "
            "Clone the repo or run with --download."
        )

    pending: List[Dict[str, Any]] = []
    total_seen = 0
    total_loaded = 0
    total_inserted = 0

    for path in paths:
        logger.info("Reading %s", path.name)
        for art in _iter_articles_json_array(path):
            if max_articles is not None and total_seen >= max_articles:
                break
            total_seen += 1
            rec = article_to_record(
                art,
                detector,
                english_only=english_only,
                include_related=include_related,
                max_related=max_related,
            )
            if rec is None:
                continue
            pending.append(rec)
            total_loaded += 1

            if store_db and len(pending) >= flush_batch:
                r = service.import_from_records(pending)
                total_inserted += int(r.get("records_inserted", 0))
                pending.clear()
                if total_loaded % (flush_batch * 8) == 0:
                    logger.info(
                        "Progress: seen=%d loaded=%d inserted_rows=%d",
                        total_seen,
                        total_loaded,
                        total_inserted,
                    )

            if max_articles is not None and total_seen >= max_articles:
                break

        if max_articles is not None and total_seen >= max_articles:
            break

    if store_db and pending:
        r = service.import_from_records(pending)
        total_inserted += int(r.get("records_inserted", 0))

    summary = {
        "archive_dir": str(archive_dir),
        "files": [p.name for p in paths],
        "year_order": "newest_first" if newest_year_first else "oldest_first",
        "articles_seen": total_seen,
        "articles_loaded": total_loaded,
        "records_inserted": total_inserted,
        "store_db": store_db,
    }
    logger.info("Done: %s", summary)
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Felix Drinkall Yahoo Finance news (JSON.xz) → FinBERT → Neon"
    )
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help=f"Folder with YYYY_processed.json.xz (default: repo/{DEFAULT_ARCHIVE_SUBDIR})",
    )
    p.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download .xz files from GitHub into --archive-dir (default: off)",
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Limit to these years (e.g. 2022 2023). Default: all present files.",
    )
    p.add_argument(
        "--newest-year-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process 2023 before 2017 so limited DB space keeps recent news (default: on).",
    )
    p.add_argument("--max-articles", type=int, default=None)
    p.add_argument("--flush-batch", type=int, default=150)
    p.add_argument(
        "--store-db",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--english-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--include-related-companies",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add related_companies as hint_tickers (more rows per article)",
    )
    p.add_argument(
        "--max-related",
        type=int,
        default=12,
        help="Cap related_companies when --include-related-companies",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[3]
    try:
        from dotenv import load_dotenv as ld
    except ImportError:
        ld = None  # type: ignore[misc, assignment]
    if ld:
        ld(repo / ".env")
        ld(repo / "backend" / ".env")

    if not os.environ.get("DATABASE_URL", "").strip():
        logger.error("DATABASE_URL is not set")
        return 1

    archive_dir = args.archive_dir or _default_archive_dir()
    year_filter = set(args.years) if args.years else None

    if args.download:
        download_year_files(archive_dir, args.years)

    try:
        run_ingest(
            archive_dir,
            max_articles=args.max_articles,
            flush_batch=args.flush_batch,
            store_db=args.store_db,
            english_only=args.english_only,
            include_related=args.include_related_companies,
            max_related=args.max_related,
            year_filter=year_filter,
            newest_year_first=args.newest_year_first,
        )
        return 0
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
