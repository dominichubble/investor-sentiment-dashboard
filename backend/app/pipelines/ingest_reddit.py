#!/usr/bin/env python3
"""
Reddit Finance Data Ingestion Pipeline

Collects finance-related posts from specified subreddits using the PRAW API,
cleans and normalizes the data, and exports to JSON format.

Usage:
    python ingest_reddit.py --time-filter week --limit 400
    python ingest_reddit.py --subreddits wallstreetbets stocks --keywords "stock" "market"
    python ingest_reddit.py --output ../data/raw/reddit --run-id 2025-10-28
"""

import os
import json
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import praw
from praw.exceptions import PRAWException


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# Regex for URL detection
_URL_RE = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


# Default configuration
DEFAULT_SUBREDDITS = ['wallstreetbets', 'stocks', 'investing', 'finance']
DEFAULT_KEYWORDS = [
    'stock', 'stocks', 'market', 'earnings', 'fed', 'inflation', 
    'rate hike', 'nvda', 'tsla', 'aapl'
]
DEFAULT_TIME_FILTER = 'week'
DEFAULT_LIMIT_PER_SUBREDDIT = 400


def clean_text(txt: str) -> str:
    """
    Remove URLs and collapse whitespace from text.
    
    Args:
        txt: Input text string
        
    Returns:
        Cleaned text with URLs removed and whitespace normalized
    """
    if not txt:
        return ''
    txt = _URL_RE.sub('', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()


def normalize_post(submission) -> Dict[str, Any]:
    """
    Extract and normalize fields from a PRAW submission.
    
    Args:
        submission: PRAW submission object
        
    Returns:
        Dictionary with normalized post data
    """
    return {
        'id': submission.id,
        'title': clean_text(submission.title),
        'selftext': clean_text(submission.selftext),
        'author': str(submission.author) if submission.author else '[deleted]',
        'subreddit': str(submission.subreddit),
        'created_utc': int(submission.created_utc),
        'score': submission.score,
        'num_comments': submission.num_comments,
        'upvote_ratio': getattr(submission, 'upvote_ratio', None),
        'url': submission.url,
        'permalink': f"https://www.reddit.com{submission.permalink}",
    }


def build_query(keywords: List[str]) -> str:
    """
    Build a Reddit search query from keywords using OR logic.
    
    Args:
        keywords: List of keywords to search for
        
    Returns:
        Query string with OR-separated terms
    """
    terms = [f'"{k}"' if ' ' in k else k for k in keywords]
    return ' OR '.join(terms)


def fetch_posts_for_subreddit(
    reddit: praw.Reddit,
    name: str,
    limit: int,
    time_filter: str,
    keywords: List[str]
) -> List[Dict[str, Any]]:
    """
    Fetch posts from a single subreddit matching the keyword query.
    
    Args:
        reddit: Authenticated PRAW Reddit instance
        name: Subreddit name (without r/)
        limit: Maximum number of posts to fetch
        time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
        keywords: List of keywords to search for
        
    Returns:
        List of normalized post dictionaries
        
    Raises:
        PRAWException: If there's an error accessing the subreddit
    """
    logger.info(f"Fetching r/{name}...")
    
    try:
        sr = reddit.subreddit(name)
        q = build_query(keywords)
        seen = set()
        rows: List[Dict[str, Any]] = []
        
        for s in sr.search(q, sort='new', time_filter=time_filter, limit=limit):
            if s.id in seen:
                continue
            seen.add(s.id)
            rows.append(normalize_post(s))
        
        logger.info(f"  Found {len(rows)} posts from r/{name}")
        return rows
        
    except PRAWException as e:
        logger.error(f"Error fetching from r/{name}: {e}")
        raise


def initialize_reddit_client() -> praw.Reddit:
    """
    Initialize and authenticate Reddit API client.
    
    Returns:
        Authenticated PRAW Reddit instance
        
    Raises:
        ValueError: If required credentials are missing
        PRAWException: If authentication fails
    """
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT', 'investor-sentiment-dashboard/0.1 by script')
    
    if not client_id or not client_secret:
        raise ValueError(
            'Missing Reddit credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET '
            'environment variables or create a .env file.'
        )
    
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False,
        )
        logger.info(f"Reddit client ready (read-only: {reddit.read_only})")
        return reddit
        
    except PRAWException as e:
        logger.error(f"Failed to initialize Reddit client: {e}")
        raise


def run_ingestion(
    subreddits: List[str],
    keywords: List[str],
    time_filter: str,
    limit_per_subreddit: int,
    output_dir: str,
    run_id: Optional[str] = None
) -> str:
    """
    Run the complete Reddit data ingestion pipeline.
    
    Args:
        subreddits: List of subreddit names to fetch from
        keywords: List of keywords to search for
        time_filter: Time filter for posts
        limit_per_subreddit: Maximum posts per subreddit
        output_dir: Output directory path
        run_id: Optional run identifier (defaults to current date)
        
    Returns:
        Path to the output JSON file
        
    Raises:
        Exception: If ingestion fails
    """
    logger.info("Starting Reddit data ingestion pipeline")
    logger.info(f"Subreddits: {', '.join(subreddits)}")
    logger.info(f"Time filter: {time_filter}, Limit: {limit_per_subreddit}")
    
    # Initialize Reddit client
    reddit = initialize_reddit_client()
    
    # Generate run ID if not provided
    if not run_id:
        run_id = datetime.utcnow().strftime('%Y-%m-%d')
    
    # Create output directory
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")
    
    # Fetch posts from all subreddits
    all_rows: List[Dict[str, Any]] = []
    errors = []
    
    for sub in subreddits:
        try:
            rows = fetch_posts_for_subreddit(
                reddit, sub, limit_per_subreddit, time_filter, keywords
            )
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"Failed to fetch from r/{sub}: {e}")
            errors.append(f"r/{sub}: {str(e)}")
    
    if errors:
        logger.warning(f"Encountered {len(errors)} errors during fetching")
    
    if not all_rows:
        logger.warning("No posts collected!")
        return ""
    
    # Deduplicate posts
    logger.info("Deduplicating posts...")
    by_id = {r['id']: r for r in all_rows}
    deduped = list(by_id.values())
    deduped.sort(key=lambda x: x.get('created_utc', 0), reverse=True)
    logger.info(f"Total unique posts: {len(deduped)}")
    
    # Write to JSON
    out_path = run_dir / f'reddit_finance_{run_id}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Successfully wrote {len(deduped)} posts to {out_path}")
    
    # Write metadata
    meta_path = run_dir / f'reddit_finance_{run_id}_meta.json'
    metadata = {
        'run_id': run_id,
        'timestamp': datetime.utcnow().isoformat(),
        'subreddits': subreddits,
        'keywords': keywords,
        'time_filter': time_filter,
        'limit_per_subreddit': limit_per_subreddit,
        'total_posts': len(deduped),
        'errors': errors
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Wrote metadata to {meta_path}")
    
    return str(out_path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Ingest finance-related posts from Reddit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (week, 400 posts per subreddit)
  python ingest_reddit.py
  
  # Custom time filter and limit
  python ingest_reddit.py --time-filter day --limit 100
  
  # Specific subreddits and keywords
  python ingest_reddit.py --subreddits stocks investing --keywords "market" "earnings"
  
  # Custom output directory
  python ingest_reddit.py --output /path/to/data --run-id 2025-10-28-evening
        """
    )
    
    parser.add_argument(
        '--subreddits',
        nargs='+',
        default=DEFAULT_SUBREDDITS,
        help=f'Subreddits to fetch from (default: {DEFAULT_SUBREDDITS})'
    )
    
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=DEFAULT_KEYWORDS,
        help='Keywords to search for (default: stock, stocks, market, etc.)'
    )
    
    parser.add_argument(
        '--time-filter',
        choices=['hour', 'day', 'week', 'month', 'year', 'all'],
        default=DEFAULT_TIME_FILTER,
        help=f'Time filter for posts (default: {DEFAULT_TIME_FILTER})'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=DEFAULT_LIMIT_PER_SUBREDDIT,
        help=f'Maximum posts per subreddit (default: {DEFAULT_LIMIT_PER_SUBREDDIT})'
    )
    
    parser.add_argument(
        '--output',
        default='data/processed/reddit',
        help='Output directory (default: data/processed/reddit)'
    )
    
    parser.add_argument(
        '--run-id',
        help='Run identifier (default: current date YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    try:
        output_file = run_ingestion(
            subreddits=args.subreddits,
            keywords=args.keywords,
            time_filter=args.time_filter,
            limit_per_subreddit=args.limit,
            output_dir=args.output,
            run_id=args.run_id
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


if __name__ == '__main__':
    exit(main())
