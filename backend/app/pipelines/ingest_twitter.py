#!/usr/bin/env python3
"""
Twitter/X Finance Data Ingestion Pipeline

Collects finance-related tweets using the Twitter API (Tweepy),
filters spam/bots, and exports to CSV format.

Usage:
    python ingest_twitter.py --max-tweets 30
    python ingest_twitter.py --keywords "earnings" "fed rate" --min-engagement 10
    python ingest_twitter.py --query "stock market OR bitcoin" --max-results 200
    python ingest_twitter.py --output ../data/raw/twitter --run-id 2025-10-28
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

import tweepy
from tweepy.errors import TweepyException

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


# Default configuration
DEFAULT_KEYWORDS = [
    "stock market",
    "stocks",
    "earnings",
    "fed rate",
    "inflation",
    "NVDA",
    "TSLA",
    "AAPL",
    "wall street",
    "bull market",
    "bear market",
]
DEFAULT_MAX_TWEETS = 30
DEFAULT_LANGUAGE = "en"
DEFAULT_MIN_ENGAGEMENT = 5


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
    Extract and normalize fields from a Twitter API v2 tweet object.

    Args:
        tweet: Tweepy Tweet object

    Returns:
        Dictionary with normalized tweet data
    """
    # Extract user data if available
    author_id = tweet.author_id if hasattr(tweet, "author_id") else None

    # Get metrics
    metrics = tweet.public_metrics if hasattr(tweet, "public_metrics") else {}

    return {
        "id": tweet.id,
        "text": clean_text(tweet.text),
        "raw_text": tweet.text,  # Keep original for reference
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


def build_query(keywords: List[str], lang: str = "en") -> str:
    """
    Build a Twitter search query from keywords.

    Args:
        keywords: List of keywords to search for
        lang: Language code (default: 'en')

    Returns:
        Query string with filters
    """
    # Quote multi-word phrases
    terms = [f'"{k}"' if " " in k else k for k in keywords]

    # Join with OR and add language filter
    query = "(" + " OR ".join(terms) + f") lang:{lang}"

    # Exclude retweets for cleaner data
    query += " -is:retweet"

    return query


def filter_spam_and_bots(tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove spam, bots, and low-quality tweets.

    Args:
        tweets: List of tweet dictionaries

    Returns:
        Filtered list of quality tweets
    """
    filtered = []

    # Enhanced spam indicators
    spam_patterns = [
        # Promotional spam
        r"check\s+(out|latest)",
        r"follow\s+(me|us|for|him)",
        r"click\s+(here|link)",
        r"dm\s+me",
        # Stock promotion spam
        r"this\s+(blogger|investor|trader).+recommends?\s+stocks?",
        r"recommends?\s+stocks?\s+that\s+rise",
        r"his\s+judgment\s+is.+(accurate|incredible)",
        r"buy\s+the\s+stocks?\s+(he|she|they)\s+recommends?",
        r"make\s+money\s+every\s+day",
        r"you\s+can\s+also\s+follow",
        # Other spam
        r"recommend\s+a\s+blogger",
        r"just\s+earned",
        r"simulation\s+market",
        r"airdrop",
        r"free\s+money",
        r"guaranteed\s+profit",
        r"\d+%\s+movement\s+in",
    ]

    for tweet in tweets:
        text_lower = tweet["text"].lower()

        # Skip if spam pattern detected
        is_spam = any(re.search(pattern, text_lower) for pattern in spam_patterns)
        if is_spam:
            continue

        # Skip if too short (likely not meaningful)
        if len(tweet["text"]) < 20:
            continue

        # Detect suspicious uniform engagement (bot networks)
        if tweet["like_count"] == tweet["retweet_count"] == tweet["reply_count"] > 0:
            continue  # Likely bot network with fake engagement

        # Skip if too many emojis (often spam)
        emoji_pattern = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
        emoji_count = len(re.findall(emoji_pattern, tweet["raw_text"]))
        if emoji_count > 5:
            continue

        # Skip if excessive hashtags
        hashtag_count = tweet["raw_text"].count("#")
        if hashtag_count > 8:
            continue

        # Skip if contains too many cashtags
        cashtag_count = tweet["raw_text"].count("$")
        if cashtag_count > 5:
            continue

        # Skip if starts with multiple random emojis
        if re.match(r"^[\U0001F000-\U0001FFFF\s]{10,}", tweet["raw_text"]):
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


def initialize_twitter_client() -> tweepy.Client:
    """
    Initialize and authenticate Twitter API client.

    Returns:
        Authenticated Tweepy Client

    Raises:
        ValueError: If required credentials are missing
        TweepyException: If authentication fails
    """
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

    if not bearer_token:
        raise ValueError(
            "Missing Twitter credentials. Set TWITTER_BEARER_TOKEN "
            "environment variable or create a .env file."
        )

    try:
        client = tweepy.Client(
            bearer_token=bearer_token,
            wait_on_rate_limit=True,  # Automatically handle rate limits
        )
        logger.info("Twitter client initialized")
        return client

    except TweepyException as e:
        logger.error(f"Failed to initialize Twitter client: {e}")
        raise


def fetch_tweets(
    client: tweepy.Client,
    keywords: List[str],
    max_results: int = 30,
    lang: str = "en",
    min_engagement: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch tweets matching keywords with quality filtering.

    Args:
        client: Authenticated Tweepy client
        keywords: List of keywords to search for
        max_results: Maximum number of tweets to fetch (10-100 per request)
        lang: Language code
        min_engagement: Minimum engagement threshold

    Returns:
        List of normalized, filtered tweet dictionaries

    Raises:
        TweepyException: If there's an error fetching tweets
    """
    query = build_query(keywords, lang)
    logger.info(f"Search query: {query}")
    logger.info(f"Fetching up to {max_results} tweets...")

    tweets = []
    seen = set()

    try:
        # Twitter API v2 recent search
        response = client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),  # API limit is 100 per request
            tweet_fields=["created_at", "author_id", "lang", "public_metrics"],
        )

        if not response.data:  # type: ignore
            logger.warning("No tweets found")
            return []

        for tweet in response.data:  # type: ignore
            if tweet.id not in seen:
                seen.add(tweet.id)
                tweets.append(normalize_tweet(tweet))

        logger.info(f"Fetched {len(tweets)} raw tweets")

        # Apply quality filters
        logger.info("Applying quality filters...")
        filtered_spam = filter_spam_and_bots(tweets)
        logger.info(f"  After spam filter: {len(filtered_spam)} tweets")

        filtered_engagement = filter_by_engagement(filtered_spam, min_engagement)
        logger.info(
            f"  After engagement filter (min={min_engagement}): {len(filtered_engagement)} tweets"
        )

        return filtered_engagement

    except TweepyException as e:
        logger.error(f"Error fetching tweets: {e}")
        raise


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
        "author_id",
        "created_at",
        "retweet_count",
        "reply_count",
        "like_count",
        "quote_count",
        "lang",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tweets)

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
    keywords: List[str],
    max_tweets: int,
    lang: str,
    min_engagement: int,
    output_dir: str,
    run_id: Optional[str] = None,
) -> str:
    """
    Run the complete Twitter data ingestion pipeline.

    Args:
        keywords: List of keywords to search for
        max_tweets: Maximum tweets to collect
        lang: Language code
        min_engagement: Minimum engagement threshold
        output_dir: Output directory path
        run_id: Optional run identifier (defaults to current date)

    Returns:
        Path to the output CSV file

    Raises:
        Exception: If ingestion fails
    """
    logger.info("Starting Twitter data ingestion pipeline")
    logger.info(f"Keywords: {', '.join(keywords)}")
    logger.info(f"Max tweets: {max_tweets}, Min engagement: {min_engagement}")

    # Initialize Twitter client
    client = initialize_twitter_client()

    # Generate run ID if not provided
    if not run_id:
        run_id = datetime.utcnow().strftime("%Y-%m-%d")

    # Fetch tweets with quality filtering
    tweets = fetch_tweets(client, keywords, max_tweets, lang, min_engagement)

    if not tweets:
        logger.warning("No quality tweets collected!")
        return ""

    # Export to CSV
    output_file = export_to_csv(tweets, output_dir, run_id)

    logger.info("✓ Pipeline completed successfully")

    return output_file


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest finance-related tweets from Twitter/X",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (30 tweets with default keywords)
  python ingest_twitter.py

  # Custom keywords
  python ingest_twitter.py --keywords "NVDA earnings" "fed meeting" "market crash"

  # Higher engagement threshold
  python ingest_twitter.py --min-engagement 10 --max-tweets 50

  # Custom output directory
  python ingest_twitter.py --output /path/to/data --run-id 2025-10-28-evening
        """,
    )

    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help=f"Keywords to search for (default: {', '.join(DEFAULT_KEYWORDS[:3])}...)",
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
        default="data/raw/twitter",
        help="Output directory (default: data/raw/twitter)",
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
            keywords=args.keywords,
            max_tweets=args.max_tweets,
            lang=args.language,
            min_engagement=args.min_engagement,
            output_dir=args.output,
            run_id=args.run_id,
        )

        if output_file:
            logger.info("✓ Pipeline completed successfully")
            return 0
        else:
            logger.error("✗ Pipeline failed: No data collected")
            return 1

    except Exception as e:
        logger.error(f"✗ Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
