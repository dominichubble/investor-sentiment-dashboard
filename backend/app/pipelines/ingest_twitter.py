#!/usr/bin/env python3
"""
Twitter/X Finance Data Ingestion Pipeline

**Default:** collects posts via **snscrape** (no official X API key). Uses the same
search query shape as the web UI (cashtags / keywords). X may block or change
scraping at any time; respect their terms and rate limits.

**Optional official API:** set ``TWITTER_INGEST_BACKEND=api`` and ``TWITTER_BEARER_TOKEN``.

**Local CSV (no X):** set ``TWITTER_INGEST_BACKEND=csv`` and optionally ``TWITTER_CSV_PATH``
(path to a file with columns ``Date``, ``Tweet``, ``Stock Name`` — same shape as ``stock_tweets.csv``).

**Auto:** ``TWITTER_INGEST_BACKEND=auto`` uses the API when ``TWITTER_BEARER_TOKEN`` is set,
otherwise falls back to snscrape.

Env:
    TWITTER_INGEST_BACKEND   ``snscrape`` (default), ``api``, ``csv``, or ``auto``
    TWITTER_CSV_PATH       CSV file for ``csv`` backend (default: repo ``stock_tweets.csv``)

Usage:
    python -m app.pipelines.ingest_twitter --max-tweets 50
    python -m app.pipelines.ingest_twitter --tickers AAPL TSLA BTC --max-tweets 40
    python -m app.pipelines.ingest_twitter --backend api --max-tweets 50
    python -m app.pipelines.ingest_twitter --keywords earnings "fed rate"
"""

import argparse
import csv
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from app.services.import_service import ImportService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Regex for URL detection
_URL_RE = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


# Default: cashtag search — tweets must match at least one symbol (recent search).
PRIORITY_TICKERS = ["BTC", "GOOGL", "NVDA", "ETH", "META"]

# Optional keyword mode (--keywords); broader but noisier than cashtags.
DEFAULT_KEYWORDS = [
    "stock market",
    "stocks",
    "earnings",
    "NVDA",
    "TSLA",
    "AAPL",
]
DEFAULT_MAX_TWEETS = 30
DEFAULT_LANGUAGE = "en"
# Ticker chatter is often low-engagement; keep 0 default, raise with --min-engagement if needed.
DEFAULT_MIN_ENGAGEMENT = 0

# Repo root (…/investor-sentiment-dashboard) for default CSV path
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CSV_REL = _REPO_ROOT / "stock_tweets.csv"


def clean_text(txt: Optional[str]) -> str:
    """
    Remove URLs, mentions, normalize hashtags, and collapse whitespace.

    Args:
        txt: Input text string (can be None)

    Returns:
        Cleaned text with URLs removed and whitespace normalized
    """
    if not txt:
        return ""

    # Remove URLs
    txt = _URL_RE.sub("", txt)

    # Remove @mentions
    txt = re.sub(r"@\w+", "", txt)

    # Keep hashtags but remove # symbol (useful for sentiment)
    txt = re.sub(r"#(\w+)", r"\1", txt)

    # Normalize whitespace
    txt = re.sub(r"\s+", " ", txt)

    return txt.strip()


def normalize_tweet(tweet) -> Dict[str, Any]:
    """
    Extract and normalize fields from a Twitter API v2 tweet object (Tweepy).

    Args:
        tweet: Tweepy Tweet object

    Returns:
        Dictionary with normalized tweet data
    """
    # Extract user data if available
    author_id = tweet.author_id if hasattr(tweet, "author_id") else None

    # Get metrics
    metrics = tweet.public_metrics if hasattr(tweet, "public_metrics") else {}
    raw = tweet.text if hasattr(tweet, "text") and tweet.text else ""
    hints = extract_cashtags(raw)

    return {
        "id": tweet.id,
        "data_source": "twitter",
        "text": clean_text(tweet.text),
        "raw_text": raw,
        "hint_tickers": hints,
        "author_id": author_id,
        "created_at": (
            tweet.created_at.isoformat() if hasattr(tweet, "created_at") else None
        ),
        "retweet_count": metrics.get("retweet_count", 0),
        "reply_count": metrics.get("reply_count", 0),
        "like_count": metrics.get("like_count", 0),
        "quote_count": metrics.get("quote_count", 0),
        "lang": tweet.lang if hasattr(tweet, "lang") else None,
    }


def _normalize_ticker_symbol(sym: str) -> str:
    s = sym.strip().upper().lstrip("$")
    if not s.isalnum():
        raise ValueError(f"Invalid ticker symbol: {sym!r}")
    return s


def build_ticker_search_query(tickers: List[str], lang: str = "en") -> str:
    """
    Build a recent-search query that matches cashtags ($NVDA, $BTC, …).

    X/Twitter matches $SYMBOL for stock/crypto tickers in search.
    """
    normalized = [_normalize_ticker_symbol(t) for t in tickers]
    cashtags = [f"${s}" for s in normalized]
    query = "(" + " OR ".join(cashtags) + f") lang:{lang} -is:retweet"
    return query


def build_keyword_query(keywords: List[str], lang: str = "en") -> str:
    """
    Build a Twitter search query from free-text keywords / phrases.
    """
    terms = [f'"{k}"' if " " in k else k for k in keywords]
    query = "(" + " OR ".join(terms) + f") lang:{lang} -is:retweet"
    return query


def extract_cashtags(raw_text: str) -> List[str]:
    """Return unique uppercased tickers from $CASHTAG tokens."""
    found = re.findall(r"\$([A-Za-z]{1,6})\b", raw_text or "")
    out: List[str] = []
    seen = set()
    for s in found:
        u = s.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def normalize_snscrape_tweet(tweet: Any) -> Dict[str, Any]:
    """Map a snscrape ``Tweet`` dataclass into the same dict shape as the API path."""
    raw = getattr(tweet, "rawContent", None) or ""
    cashtags = getattr(tweet, "cashtags", None) or []
    hints: List[str] = []
    if cashtags:
        for h in cashtags:
            if not h:
                continue
            u = str(h).strip().lstrip("$").upper()
            if u and u not in hints:
                hints.append(u)
    if not hints:
        hints = extract_cashtags(raw)

    user = getattr(tweet, "user", None)
    author_id = str(user.id) if user is not None and getattr(user, "id", None) else None

    created = getattr(tweet, "date", None)
    created_at = created.isoformat() if created is not None else None

    tid = getattr(tweet, "id", None)
    try:
        id_int = int(tid) if tid is not None else 0
    except (TypeError, ValueError):
        id_int = abs(hash(str(tid))) % (10**12)

    return {
        "id": id_int,
        "data_source": "twitter",
        "text": clean_text(raw),
        "raw_text": raw,
        "hint_tickers": hints,
        "author_id": author_id,
        "created_at": created_at,
        "retweet_count": int(getattr(tweet, "retweetCount", 0) or 0),
        "reply_count": int(getattr(tweet, "replyCount", 0) or 0),
        "like_count": int(getattr(tweet, "likeCount", 0) or 0),
        "quote_count": int(getattr(tweet, "quoteCount", 0) or 0),
        "lang": getattr(tweet, "lang", None),
    }


def normalize_csv_finance_row(row: Dict[str, str], seq: int) -> Dict[str, Any]:
    """
    Map a ``stock_tweets.csv``-style row into the same dict shape as API/snscrape tweets.
    Engagement fields are unknown in this dataset and are set to 0.
    """
    raw = (row.get("Tweet") or "").strip()
    sym = (row.get("Stock Name") or "").strip().upper()
    hints = extract_cashtags(raw)
    if sym:
        hints = [sym] + [h for h in hints if h != sym]
    id_src = f"{row.get('Date', '')}:{raw[:240]}:{seq}"
    id_int = abs(hash(id_src)) % (10**12) or seq

    return {
        "id": id_int,
        "data_source": "twitter",
        "text": clean_text(raw),
        "raw_text": raw,
        "hint_tickers": hints,
        "author_id": None,
        "created_at": (row.get("Date") or "").strip() or None,
        "retweet_count": 0,
        "reply_count": 0,
        "like_count": 0,
        "quote_count": 0,
        "lang": None,
    }


def filter_low_quality_tweets(tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop obvious spam / empty noise. Ticker posts are often short — keep a low
    character floor when a cashtag is present.
    """
    filtered: List[Dict[str, Any]] = []

    spam_patterns = [
        r"check\s+(out|latest)",
        r"follow\s+(me|us|for|him)\b",
        r"click\s+(here|link)",
        r"\bdm\s+me\b",
        r"this\s+(blogger|investor|trader).+recommends?\s+stocks?",
        r"recommends?\s+stocks?\s+that\s+rise",
        r"buy\s+the\s+stocks?\s+(he|she|they)\s+recommends?",
        r"make\s+money\s+every\s+day",
        r"\bfree\s+money\b",
        r"guaranteed\s+profit",
        r"\bairdrop\b",
    ]

    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
    )

    for tweet in tweets:
        raw = tweet.get("raw_text") or ""
        text = tweet.get("text") or ""
        text_lower = text.lower()
        has_cashtag = "$" in raw

        if any(re.search(p, text_lower) for p in spam_patterns):
            continue

        # Cashtag posts can be very short ("$NVDA calls"); others need a bit more text.
        min_len = 8 if has_cashtag else 15
        if len(text.strip()) < min_len:
            continue

        if len(emoji_pattern.findall(raw)) > 8:
            continue

        if raw.count("#") > 12:
            continue

        if raw.count("$") > 15:
            continue

        if re.match(r"^[\U0001F000-\U0001FFFF\s]{12,}", raw):
            continue

        filtered.append(tweet)

    return filtered


def filter_by_engagement(
    tweets: List[Dict[str, Any]], min_engagement: int = 5
) -> List[Dict[str, Any]]:
    """
    Keep only tweets with minimum engagement threshold.

    Args:
        tweets: List of tweet dictionaries
        min_engagement: Minimum total engagement (likes + retweets + replies)

    Returns:
        Filtered list of tweets
    """
    return [
        t
        for t in tweets
        if (t["like_count"] + t["retweet_count"] + t["reply_count"]) >= min_engagement
    ]


def _resolve_search_query(
    lang: str,
    tickers: Optional[List[str]],
    keywords: Optional[List[str]],
) -> str:
    if keywords:
        q = build_keyword_query(keywords, lang)
        logger.info("Search mode: keywords (%d terms)", len(keywords))
    else:
        tlist = tickers if tickers else PRIORITY_TICKERS
        q = build_ticker_search_query(tlist, lang)
        logger.info("Search mode: cashtags (%s)", ", ".join(tlist))
    logger.info("Search query: %s", q)
    return q


def _apply_filters(records: List[Dict[str, Any]], min_engagement: int) -> List[Dict[str, Any]]:
    logger.info("Fetched %d raw tweets", len(records))
    logger.info("Applying light quality filters...")
    filtered_spam = filter_low_quality_tweets(records)
    logger.info("  After quality filter: %d tweets", len(filtered_spam))
    filtered_engagement = filter_by_engagement(filtered_spam, min_engagement)
    logger.info(
        "  After engagement filter (min=%s): %d tweets",
        min_engagement,
        len(filtered_engagement),
    )
    return filtered_engagement


def resolve_ingest_backend(explicit: Optional[str] = None) -> str:
    """Return ``snscrape``, ``api``, ``csv``, or ``auto`` (normalized)."""
    raw = (explicit or os.getenv("TWITTER_INGEST_BACKEND") or "snscrape").strip().lower()
    if raw in ("api", "official", "twitter_api", "tweepy"):
        return "api"
    if raw in ("csv", "file", "local", "dataset"):
        return "csv"
    if raw == "auto":
        token = (os.getenv("TWITTER_BEARER_TOKEN") or "").strip()
        return "api" if token else "snscrape"
    return "snscrape"


def resolve_csv_path() -> Path:
    """Path for the CSV ingest backend."""
    env = (os.getenv("TWITTER_CSV_PATH") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_CSV_REL.resolve()


def initialize_twitter_client() -> Any:
    """Lazy-import Tweepy; only used when ``TWITTER_INGEST_BACKEND=api``."""
    import tweepy
    from tweepy.errors import TweepyException

    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

    if not bearer_token:
        raise ValueError(
            "Missing Twitter credentials. Set TWITTER_BEARER_TOKEN "
            "environment variable or create a .env file."
        )

    try:
        client = tweepy.Client(
            bearer_token=bearer_token,
            wait_on_rate_limit=True,
        )
        logger.info("Twitter API client initialized (Tweepy)")
        return client

    except TweepyException as e:
        logger.error("Failed to initialize Twitter client: %s", e)
        raise


def fetch_tweets_snscrape(
    max_results: int,
    lang: str,
    min_engagement: int,
    tickers: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Collect posts using snscrape (no bearer token). May break if X changes their site.
    """
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass

    try:
        import snscrape.modules.twitter as sntwitter
        from snscrape.modules.twitter import Tweet as SnsTweet
    except ImportError as e:
        raise ImportError(
            "snscrape is required for the default Twitter ingest. "
            "Install with: pip install snscrape"
        ) from e

    query = _resolve_search_query(lang, tickers, keywords)
    logger.info("Fetching up to %s posts via snscrape...", max_results)

    records: List[Dict[str, Any]] = []
    seen: set[int] = set()

    try:
        scraper = sntwitter.TwitterSearchScraper(query)
        for item in scraper.get_items():
            if len(records) >= max_results:
                break
            if not isinstance(item, SnsTweet):
                continue
            if getattr(item, "retweetedTweet", None) is not None:
                continue
            rec = normalize_snscrape_tweet(item)
            tid = rec.get("id")
            if isinstance(tid, int) and tid not in seen:
                seen.add(tid)
                records.append(rec)
    except Exception as e:
        logger.error(
            "snscrape failed (X often blocks automated access): %s",
            e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return []

    return _apply_filters(records, min_engagement)


def fetch_tweets_api(
    client: Any,
    max_results: int = 30,
    lang: str = "en",
    min_engagement: int = 0,
    tickers: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch recent tweets via official X API v2 recent search (Tweepy)."""
    from tweepy.errors import Forbidden, TweepyException, Unauthorized

    query = _resolve_search_query(lang, tickers, keywords)
    logger.info("Fetching up to %s tweets via API...", max_results)

    tweets: List[Dict[str, Any]] = []
    seen: set[int] = set()
    next_token: Optional[str] = None

    try:
        while len(tweets) < max_results:
            need = max_results - len(tweets)
            # v2 recent search allows 10–100 per request
            page_size = min(100, max(10, need))
            response = client.search_recent_tweets(
                query=query,
                max_results=page_size,
                tweet_fields=["created_at", "author_id", "lang", "public_metrics"],
                next_token=next_token,
            )

            if not response.data:  # type: ignore
                if not tweets:
                    logger.warning("No tweets found")
                break

            for tweet in response.data:  # type: ignore
                if tweet.id not in seen:
                    seen.add(tweet.id)
                    tweets.append(normalize_tweet(tweet))
                    if len(tweets) >= max_results:
                        break

            meta = getattr(response, "meta", None)
            if meta is None:
                next_token = None
            elif isinstance(meta, dict):
                next_token = meta.get("next_token")
            else:
                next_token = getattr(meta, "next_token", None)
            if not next_token:
                break

    except (Forbidden, Unauthorized) as e:
        logger.warning(
            "Twitter recent search unavailable (API rejected this token). "
            "Use an app inside a Project with v2 access, or switch to "
            "TWITTER_INGEST_BACKEND=csv / snscrape. %s",
            e,
        )
        return []

    except TweepyException as e:
        logger.error("Error fetching tweets: %s", e)
        raise

    return _apply_filters(tweets, min_engagement)


def fetch_tweets_csv(
    max_results: int,
    lang: str,
    min_engagement: int,
    tickers: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    csv_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load tweets from a local CSV (``Date``, ``Tweet``, ``Stock Name``, …).
    No network calls; suitable when X scraping/API is unavailable.

    Rows are read in file order up to ``max_results``. ``tickers`` is ignored
    (each row already has a stock in ``Stock Name``). If ``keywords`` is set,
    only rows whose tweet text contains at least one keyword (case-insensitive)
    are kept.
    """
    path = csv_path if csv_path is not None else resolve_csv_path()
    if not path.is_file():
        logger.error("CSV ingest: file not found: %s", path)
        return []

    logger.info("Loading up to %s rows from CSV: %s", max_results, path)

    records: List[Dict[str, Any]] = []
    seq = 0

    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(records) >= max_results:
                break
            seq += 1
            raw = (row.get("Tweet") or "").strip()
            if keywords:
                lower = raw.lower()
                if not any(k.lower() in lower for k in keywords):
                    continue
            rec = normalize_csv_finance_row(row, seq)
            records.append(rec)

    return _apply_filters(records, min_engagement)


def export_to_csv(tweets: List[Dict[str, Any]], output_dir: str, run_id: str) -> str:
    """
    Export tweets to CSV file.

    Args:
        tweets: List of tweet dictionaries
        output_dir: Output directory path
        run_id: Run identifier

    Returns:
        Path to output CSV file

    Raises:
        Exception: If export fails
    """
    if not tweets:
        logger.warning("No tweets to export")
        return ""

    # Create output directory
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Output file path
    output_file = run_dir / f"twitter_finance_{run_id}.csv"

    # Write to CSV
    fieldnames = [
        "id",
        "text",
        "raw_text",
        "hint_tickers",
        "author_id",
        "created_at",
        "retweet_count",
        "reply_count",
        "like_count",
        "quote_count",
        "lang",
    ]

    def _row_for_csv(t: Dict[str, Any]) -> Dict[str, Any]:
        row = {k: t.get(k) for k in fieldnames}
        hints = t.get("hint_tickers")
        if isinstance(hints, list):
            row["hint_tickers"] = " ".join(hints)
        return row

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for t in tweets:
            writer.writerow(_row_for_csv(t))

    logger.info(f"Exported {len(tweets)} tweets to {output_file}")

    # Also save metadata
    meta_file = run_dir / f"twitter_finance_{run_id}_meta.txt"
    with open(meta_file, "w", encoding="utf-8") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
        f.write(f"Total tweets: {len(tweets)}\n")

    logger.info(f"Saved metadata to {meta_file}")

    return str(output_file)


def run_ingestion(
    max_tweets: int,
    lang: str,
    min_engagement: int,
    output_dir: str,
    tickers: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    backend: Optional[str] = None,
    run_id: Optional[str] = None,
    store_db: bool = True,
    write_files: bool = False,
) -> str:
    """
    Run the complete Twitter data ingestion pipeline.

    By default searches recent tweets matching PRIORITY_TICKERS cashtags.
    Pass ``keywords`` to use free-text search instead.

    Args:
        max_tweets: Maximum tweets to collect
        lang: Language code
        min_engagement: Minimum engagement threshold
        output_dir: Output directory path
        tickers: Cashtag symbols (without $); defaults to PRIORITY_TICKERS
        keywords: If set, keyword search instead of tickers
        backend: ``snscrape``, ``api``, ``csv``, or ``auto``; default from env
        run_id: Optional run identifier (defaults to current date)
        store_db: If True, persist records directly to the database
        write_files: If True, also export CSV files to local data directory

    Returns:
        Path to the output CSV file

    Raises:
        Exception: If ingestion fails
    """
    logger.info("Starting Twitter data ingestion pipeline")
    if keywords:
        logger.info("Keywords: %s", ", ".join(keywords))
    else:
        tlist = tickers if tickers else PRIORITY_TICKERS
        logger.info("Tickers: %s", ", ".join(tlist))
    logger.info("Max tweets: %s, Min engagement: %s", max_tweets, min_engagement)

    mode = resolve_ingest_backend(backend)
    logger.info("Ingest backend: %s", mode)

    if not run_id:
        run_id = datetime.utcnow().strftime("%Y-%m-%d")

    if mode == "api":
        client = initialize_twitter_client()
        tweets = fetch_tweets_api(
            client,
            max_results=max_tweets,
            lang=lang,
            min_engagement=min_engagement,
            tickers=tickers,
            keywords=keywords,
        )
    elif mode == "csv":
        tweets = fetch_tweets_csv(
            max_results=max_tweets,
            lang=lang,
            min_engagement=min_engagement,
            tickers=tickers,
            keywords=keywords,
        )
    else:
        tweets = fetch_tweets_snscrape(
            max_results=max_tweets,
            lang=lang,
            min_engagement=min_engagement,
            tickers=tickers,
            keywords=keywords,
        )

    if not tweets:
        logger.warning(
            "No tweets after fetch and filtering (snscrape/API errors above, "
            "or loosen --min-engagement / filters)."
        )
        return ""

    if store_db:
        import_result = ImportService().import_from_records(tweets)
        logger.info(
            "Imported into DB: loaded=%s inserted=%s",
            import_result["records_loaded"],
            import_result["records_inserted"],
        )

    output_file = ""
    if write_files:
        output_file = export_to_csv(tweets, output_dir, run_id)

    logger.info("✓ Pipeline completed successfully")
    if write_files:
        return output_file
    return "db://sentiment_records" if store_db else ""


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest finance-related tweets from Twitter/X",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: cashtags for BTC, GOOGL, NVDA, ETH, META
  python -m app.pipelines.ingest_twitter

  # Custom ticker set (symbols without $)
  python -m app.pipelines.ingest_twitter --tickers AAPL MSFT COIN

  # Keyword search instead of cashtags
  python -m app.pipelines.ingest_twitter --keywords earnings "stock market"

  # Stricter engagement + CSV export
  python -m app.pipelines.ingest_twitter --min-engagement 5 --write-files --run-id 2025-10-28
        """,
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="SYM",
        help=(
            "Ticker symbols for cashtag search (no $). "
            f"Default: {' '.join(PRIORITY_TICKERS)}"
        ),
    )

    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="If set, search these phrases instead of --tickers (keyword mode)",
    )

    parser.add_argument(
        "--backend",
        choices=["snscrape", "api", "csv", "auto"],
        default=None,
        help="Override TWITTER_INGEST_BACKEND (snscrape | api | csv | auto)",
    )

    parser.add_argument(
        "--max-tweets",
        type=int,
        default=DEFAULT_MAX_TWEETS,
        help=f"Maximum tweets to collect (default: {DEFAULT_MAX_TWEETS})",
    )

    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})",
    )

    parser.add_argument(
        "--min-engagement",
        type=int,
        default=DEFAULT_MIN_ENGAGEMENT,
        help=f"Minimum engagement (likes+retweets+replies) (default: {DEFAULT_MIN_ENGAGEMENT})",
    )

    parser.add_argument(
        "--output",
        default="data/artifacts/twitter",
        help="Output directory for optional file artifacts (default: data/artifacts/twitter)",
    )
    parser.add_argument(
        "--store-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store ingested records in the database (default: True)",
    )
    parser.add_argument(
        "--write-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write CSV artifacts under --output (default: False)",
    )

    parser.add_argument(
        "--run-id", help="Run identifier (default: current date YYYY-MM-DD)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))

    try:
        output_file = run_ingestion(
            max_tweets=args.max_tweets,
            lang=args.language,
            min_engagement=args.min_engagement,
            output_dir=args.output,
            tickers=args.tickers,
            keywords=args.keywords,
            backend=args.backend,
            run_id=args.run_id,
            store_db=args.store_db,
            write_files=args.write_files,
        )

        if output_file:
            logger.info("✓ Pipeline completed successfully")
            return 0
        if os.environ.get("CI", "").lower() == "true":
            logger.warning(
                "No Twitter rows ingested; exiting 0 under CI so Reddit/News DB updates are not blocked."
            )
            return 0
        logger.error("✗ Pipeline failed: No data collected")
        return 1

    except Exception as e:
        logger.error(f"✗ Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
