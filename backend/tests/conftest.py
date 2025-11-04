"""Test configuration and shared fixtures for pytest."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def sample_reddit_post():
    """Sample Reddit post data for testing."""
    return {
        "id": "test123",
        "title": "Market rally continues",
        "selftext": "The stock market is showing strong bullish momentum.",
        "author": "test_user",
        "subreddit": "stocks",
        "score": 100,
        "num_comments": 50,
        "created_utc": 1699123456,
        "url": "https://reddit.com/r/stocks/test123",
    }


@pytest.fixture
def sample_tweet():
    """Sample tweet data for testing."""
    return {
        "id": "1234567890",
        "text": "Breaking: Stock XYZ up 15% after earnings beat!",
        "author_id": "user123",
        "created_at": "2025-11-04T10:00:00.000Z",
        "public_metrics": {
            "retweet_count": 10,
            "reply_count": 5,
            "like_count": 25,
        },
    }


@pytest.fixture
def sample_news_article():
    """Sample news article data for testing."""
    return {
        "title": "Federal Reserve Announces Rate Decision",
        "description": "The Fed maintains current interest rates amid economic uncertainty.",
        "content": "Full article content here...",
        "url": "https://example.com/article",
        "source": {"name": "Financial Times"},
        "publishedAt": "2025-11-04T09:00:00Z",
        "author": "John Doe",
    }
