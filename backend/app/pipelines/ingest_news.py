#!/usr/bin/env python3
"""
News API Finance Data Ingestion Pipeline

Collects finance-related news articles using NewsAPI.org,
filters low-quality content, and exports to JSON format.

Usage:
    python ingest_news.py --max-articles 100
    python ingest_news.py --keywords "earnings" "fed rate" --sources bloomberg reuters
    python ingest_news.py --output ../data/processed/news --run-id 2025-11-02
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

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
    "federal reserve",
    "interest rates",
    "inflation",
    "NVDA",
    "TSLA",
    "AAPL",
    "market crash",
    "bull market",
    "bear market",
]

DEFAULT_SOURCES = [
    "bloomberg",
    "reuters",
    "the-wall-street-journal",
    "financial-times",
    "business-insider",
    "fortune",
    "cnbc",
]

DEFAULT_MAX_ARTICLES = 100
DEFAULT_LANGUAGE = "en"
DEFAULT_DAYS_BACK = 7


def clean_text(txt: Optional[str]) -> str:
    """
    Remove HTML tags, URLs, and normalize whitespace from text.

    Args:
        txt: Input text string (can be None)

    Returns:
        Cleaned text with HTML, URLs removed and whitespace normalized
    """
    if not txt:
        return ""

    # Remove HTML tags
    txt = re.sub(r"<[^>]+>", "", txt)

    # Remove URLs
    txt = _URL_RE.sub("", txt)

    # Remove [+XXX chars] artifacts from NewsAPI
    txt = re.sub(r"\[\+\d+ chars\]", "", txt)

    # Remove [Removed] markers
    txt = re.sub(r"\[Removed\]", "", txt, flags=re.IGNORECASE)

    # Normalize whitespace
    txt = re.sub(r"\s+", " ", txt)

    return txt.strip()


def normalize_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalize fields from a NewsAPI article object.

    Args:
        article: NewsAPI article dictionary

    Returns:
        Dictionary with normalized article data
    """
    source = article.get("source", {})

    return {
        "source_id": source.get("id"),
        "source_name": source.get("name"),
        "author": article.get("author"),
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "url": article.get("url"),
        "url_to_image": article.get("urlToImage"),
        "published_at": article.get("publishedAt"),
        "content": article.get("content", ""),
        "clean_title": clean_text(article.get("title", "")),
        "clean_description": clean_text(article.get("description", "")),
        "clean_content": clean_text(article.get("content", "")),
    }


def build_query(keywords: List[str]) -> str:
    """
    Build a NewsAPI search query from keywords.

    Args:
        keywords: List of keywords to search for

    Returns:
        Query string with OR-separated terms
    """
    # Quote multi-word phrases
    terms = [f'"{k}"' if " " in k else k for k in keywords]

    # Join with OR for broad coverage
    query = " OR ".join(terms)

    return query


def filter_quality_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove low-quality, paywalled, or removed articles.

    Args:
        articles: List of article dictionaries

    Returns:
        Filtered list of quality articles
    """
    filtered = []

    for article in articles:
        # Skip if title is missing or too short
        if not article["clean_title"] or len(article["clean_title"]) < 10:
            continue

        # Skip if marked as [Removed]
        if "[removed]" in article.get("content", "").lower():
            continue

        # Skip if content is too short (likely paywalled)
        if article["clean_content"] and len(article["clean_content"]) < 100:
            continue

        # Skip if no description or content
        if not article["clean_description"] and not article["clean_content"]:
            continue

        filtered.append(article)

    return filtered


def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate articles by URL and similar titles.

    Args:
        articles: List of article dictionaries

    Returns:
        Deduplicated list of articles
    """
    seen_urls = set()
    seen_titles = set()
    deduped = []

    for article in articles:
        url = article["url"]
        title = article["clean_title"].lower()

        # Skip if we've seen this URL
        if url and url in seen_urls:
            continue

        # Skip if we've seen very similar title
        if title in seen_titles:
            continue

        seen_urls.add(url)
        seen_titles.add(title)
        deduped.append(article)

    return deduped


def initialize_news_client() -> NewsApiClient:
    """
    Initialize and authenticate NewsAPI client.

    Returns:
        Authenticated NewsApiClient

    Raises:
        ValueError: If required credentials are missing
        NewsAPIException: If authentication fails
    """
    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing NewsAPI credentials. Set NEWS_API_KEY "
            "environment variable or create a .env file."
        )

    try:
        client = NewsApiClient(api_key=api_key)
        logger.info("NewsAPI client initialized")
        return client

    except NewsAPIException as e:
        logger.error(f"Failed to initialize NewsAPI client: {e}")
        raise


def fetch_articles(
    newsapi: NewsApiClient,
    keywords: List[str],
    sources: Optional[List[str]] = None,
    days_back: int = 7,
    max_results: int = 100,
    language: str = "en",
) -> List[Dict[str, Any]]:
    """
    Fetch articles matching keywords with quality filtering.

    Args:
        newsapi: Authenticated NewsApiClient
        keywords: List of keywords to search for
        sources: Optional list of source IDs to filter by
        days_back: Number of days to search back
        max_results: Maximum number of articles to fetch
        language: Language code

    Returns:
        List of normalized, filtered article dictionaries

    Raises:
        NewsAPIException: If there's an error fetching articles
    """
    query = build_query(keywords)
    logger.info(f"Search query: {query[:100]}..." if len(query) > 100 else query)
    logger.info(f"Fetching up to {max_results} articles...")

    # Calculate date range (NewsAPI needs YYYY-MM-DD format)
    to_date = datetime.utcnow()
    from_date = to_date - timedelta(days=days_back)

    logger.info(f"Date range: {from_date.date()} to {to_date.date()}")

    try:
        # Call NewsAPI everything endpoint
        response = newsapi.get_everything(
            q=query,
            sources=",".join(sources) if sources else None,
            from_param=from_date.strftime("%Y-%m-%d"),
            to=to_date.strftime("%Y-%m-%d"),
            language=language,
            sort_by="publishedAt",
            page_size=min(max_results, 100),  # API limit is 100 per page
        )

        if response["status"] != "ok":
            logger.warning(f"API returned status: {response['status']}")
            return []

        articles = response.get("articles", [])
        logger.info(f"Fetched {len(articles)} raw articles")

        # Normalize articles
        normalized = [normalize_article(a) for a in articles]

        # Apply quality filters
        logger.info("Applying quality filters...")
        filtered = filter_quality_articles(normalized)
        logger.info(f"  After quality filter: {len(filtered)} articles")

        deduped = deduplicate_articles(filtered)
        logger.info(f"  After deduplication: {len(deduped)} articles")

        return deduped

    except NewsAPIException as e:
        logger.error(f"Error fetching articles: {e}")
        raise


def export_to_json(articles: List[Dict[str, Any]], output_dir: str, run_id: str) -> str:
    """
    Export articles to JSON file.

    Args:
        articles: List of article dictionaries
        output_dir: Output directory path
        run_id: Run identifier

    Returns:
        Path to output JSON file

    Raises:
        Exception: If export fails
    """
    if not articles:
        logger.warning("No articles to export")
        return ""

    # Create output directory
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Output file path
    output_file = run_dir / f"news_finance_{run_id}.json"

    # Write to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(f"Exported {len(articles)} articles to {output_file}")

    # Also save metadata
    meta_file = run_dir / f"news_finance_{run_id}_meta.json"
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "total_articles": len(articles),
        "sources": list(set(a["source_name"] for a in articles if a["source_name"])),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved metadata to {meta_file}")

    return str(output_file)


def run_ingestion(
    keywords: List[str],
    sources: Optional[List[str]],
    days_back: int,
    max_articles: int,
    language: str,
    output_dir: str,
    run_id: Optional[str] = None,
) -> str:
    """
    Run the complete News API data ingestion pipeline.

    Args:
        keywords: List of keywords to search for
        sources: Optional list of source IDs
        days_back: Number of days to search back
        max_articles: Maximum articles to collect
        language: Language code
        output_dir: Output directory path
        run_id: Optional run identifier (defaults to current date)

    Returns:
        Path to the output JSON file

    Raises:
        Exception: If ingestion fails
    """
    logger.info("Starting News API data ingestion pipeline")
    logger.info(f"Keywords: {', '.join(keywords[:5])}... ({len(keywords)} total)")
    if sources:
        logger.info(f"Sources: {', '.join(sources[:3])}... ({len(sources)} total)")
    logger.info(f"Max articles: {max_articles}, Days back: {days_back}")

    # Initialize NewsAPI client
    newsapi = initialize_news_client()

    # Generate run ID if not provided
    if not run_id:
        run_id = datetime.utcnow().strftime("%Y-%m-%d")

    # Fetch articles with quality filtering
    articles = fetch_articles(
        newsapi, keywords, sources, days_back, max_articles, language
    )

    if not articles:
        logger.warning("No quality articles collected!")
        return ""

    # Export to JSON
    output_file = export_to_json(articles, output_dir, run_id)

    logger.info("✓ Pipeline completed successfully")

    return output_file


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest finance-related news articles from NewsAPI.org",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (100 articles with default keywords)
  python ingest_news.py

  # Custom keywords
  python ingest_news.py --keywords "NVDA earnings" "fed meeting" "market crash"

  # Specific sources only
  python ingest_news.py --sources bloomberg reuters wsj

  # Longer date range
  python ingest_news.py --days-back 30 --max-articles 100

  # Custom output directory
  python ingest_news.py --output /path/to/data --run-id 2025-11-02-evening
        """,
    )

    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help=f"Keywords to search for (default: {', '.join(DEFAULT_KEYWORDS[:3])}...)",
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help=f"Source IDs to filter by (default: {', '.join(DEFAULT_SOURCES[:3])}...)",
    )

    parser.add_argument(
        "--days-back",
        type=int,
        default=DEFAULT_DAYS_BACK,
        help=f"Number of days to search back (default: {DEFAULT_DAYS_BACK})",
    )

    parser.add_argument(
        "--max-articles",
        type=int,
        default=DEFAULT_MAX_ARTICLES,
        help=f"Maximum articles to collect (default: {DEFAULT_MAX_ARTICLES})",
    )

    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})",
    )

    parser.add_argument(
        "--output",
        default="data/processed/news",
        help="Output directory (default: data/processed/news)",
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
            sources=args.sources,
            days_back=args.days_back,
            max_articles=args.max_articles,
            language=args.language,
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
