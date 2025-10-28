"""
Unit tests for Reddit data ingestion pipeline.

Tests cover:
- Text cleaning functionality
- Post normalization
- Query building
- Data structures
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipelines.ingest_reddit import (
    clean_text,
    normalize_post,
    build_query,
)


class TestCleanText:
    """Test the clean_text function."""
    
    def test_clean_text_removes_urls(self):
        """URLs should be removed from text."""
        text = "Check out NVDA! https://example.com is bullish"
        result = clean_text(text)
        assert "https://example.com" not in result
        assert "Check out NVDA" in result
        assert "is bullish" in result
    
    def test_clean_text_removes_multiple_urls(self):
        """Multiple URLs should be removed."""
        text = "Visit http://site1.com and https://site2.com for info"
        result = clean_text(text)
        assert "http://site1.com" not in result
        assert "https://site2.com" not in result
        assert "Visit" in result
        assert "and" in result
        assert "for info" in result
    
    def test_clean_text_normalizes_whitespace(self):
        """Multiple spaces should be collapsed to single space."""
        text = "TSLA    stock    going    up"
        result = clean_text(text)
        assert result == "TSLA stock going up"
        assert "    " not in result
    
    def test_clean_text_handles_newlines(self):
        """Newlines should be converted to spaces."""
        text = "Line 1\nLine 2\nLine 3"
        result = clean_text(text)
        assert result == "Line 1 Line 2 Line 3"
    
    def test_clean_text_strips_leading_trailing_whitespace(self):
        """Leading and trailing whitespace should be removed."""
        text = "   AAPL earnings   "
        result = clean_text(text)
        assert result == "AAPL earnings"
        assert not result.startswith(" ")
        assert not result.endswith(" ")
    
    def test_clean_text_handles_empty_string(self):
        """Empty strings should return empty string."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""
    
    def test_clean_text_handles_none(self):
        """None should return empty string."""
        assert clean_text(None) == ""
    
    def test_clean_text_preserves_normal_text(self):
        """Normal text without URLs or extra spaces should be preserved."""
        text = "The market is volatile today"
        result = clean_text(text)
        assert result == text


class TestBuildQuery:
    """Test the build_query function."""
    
    def test_build_query_single_word(self):
        """Single word keywords should be joined with OR."""
        keywords = ["stock"]
        result = build_query(keywords)
        assert result == "stock"
    
    def test_build_query_multiple_words(self):
        """Multiple keywords should be OR-separated."""
        keywords = ["stock", "market", "earnings"]
        result = build_query(keywords)
        assert result == "stock OR market OR earnings"
    
    def test_build_query_phrases_get_quoted(self):
        """Multi-word phrases should be quoted."""
        keywords = ["rate hike", "stock market"]
        result = build_query(keywords)
        assert '"rate hike"' in result
        assert '"stock market"' in result
    
    def test_build_query_mixed_single_and_phrases(self):
        """Mix of single words and phrases should be handled correctly."""
        keywords = ["stock", "rate hike", "earnings"]
        result = build_query(keywords)
        assert "stock" in result
        assert '"rate hike"' in result
        assert "earnings" in result
        assert " OR " in result
    
    def test_build_query_empty_list(self):
        """Empty keyword list should return empty string."""
        keywords = []
        result = build_query(keywords)
        assert result == ""


class TestNormalizePost:
    """Test the normalize_post function."""
    
    def test_normalize_post_basic_fields(self):
        """Basic post fields should be extracted correctly."""
        # Create mock submission
        mock_submission = Mock()
        mock_submission.id = "abc123"
        mock_submission.title = "TSLA to the moon"
        mock_submission.selftext = "Stock is going up!"
        mock_submission.author = "trader123"
        mock_submission.subreddit = "wallstreetbets"
        mock_submission.created_utc = 1730073600.0
        mock_submission.score = 42
        mock_submission.num_comments = 10
        mock_submission.upvote_ratio = 0.89
        mock_submission.url = "https://reddit.com/..."
        mock_submission.permalink = "/r/wallstreetbets/comments/abc123/..."
        
        result = normalize_post(mock_submission)
        
        assert result['id'] == "abc123"
        assert result['title'] == "TSLA to the moon"
        assert result['selftext'] == "Stock is going up!"
        assert result['author'] == "trader123"
        assert result['subreddit'] == "wallstreetbets"
        assert result['created_utc'] == 1730073600
        assert result['score'] == 42
        assert result['num_comments'] == 10
        assert result['upvote_ratio'] == 0.89
    
    def test_normalize_post_cleans_text(self):
        """Post title and selftext should be cleaned."""
        mock_submission = Mock()
        mock_submission.id = "test123"
        mock_submission.title = "Check this https://example.com    out"
        mock_submission.selftext = "Visit    http://site.com   for more"
        mock_submission.author = "user"
        mock_submission.subreddit = "stocks"
        mock_submission.created_utc = 1234567890.0
        mock_submission.score = 1
        mock_submission.num_comments = 0
        mock_submission.upvote_ratio = 1.0
        mock_submission.url = "https://reddit.com/..."
        mock_submission.permalink = "/r/stocks/comments/test123/..."
        
        result = normalize_post(mock_submission)
        
        assert "https://example.com" not in result['title']
        assert "http://site.com" not in result['selftext']
        assert "    " not in result['title']
        assert "    " not in result['selftext']
    
    def test_normalize_post_deleted_author(self):
        """Deleted authors should be handled as '[deleted]'."""
        mock_submission = Mock()
        mock_submission.id = "deleted123"
        mock_submission.title = "Title"
        mock_submission.selftext = "Text"
        mock_submission.author = None  # Deleted author
        mock_submission.subreddit = "stocks"
        mock_submission.created_utc = 1234567890.0
        mock_submission.score = 5
        mock_submission.num_comments = 2
        mock_submission.upvote_ratio = 0.75
        mock_submission.url = "https://reddit.com/..."
        mock_submission.permalink = "/r/stocks/comments/deleted123/..."
        
        result = normalize_post(mock_submission)
        
        assert result['author'] == "[deleted]"
    
    def test_normalize_post_missing_upvote_ratio(self):
        """Missing upvote_ratio should be handled gracefully."""
        mock_submission = Mock(spec=['id', 'title', 'selftext', 'author', 
                                      'subreddit', 'created_utc', 'score', 
                                      'num_comments', 'url', 'permalink'])
        mock_submission.id = "test123"
        mock_submission.title = "Title"
        mock_submission.selftext = "Text"
        mock_submission.author = "user"
        mock_submission.subreddit = "stocks"
        mock_submission.created_utc = 1234567890.0
        mock_submission.score = 5
        mock_submission.num_comments = 2
        mock_submission.url = "https://reddit.com/..."
        mock_submission.permalink = "/r/stocks/comments/test123/..."
        
        result = normalize_post(mock_submission)
        
        assert result['upvote_ratio'] is None
    
    def test_normalize_post_permalink_formatted(self):
        """Permalink should be formatted as full URL."""
        mock_submission = Mock()
        mock_submission.id = "test123"
        mock_submission.title = "Title"
        mock_submission.selftext = "Text"
        mock_submission.author = "user"
        mock_submission.subreddit = "stocks"
        mock_submission.created_utc = 1234567890.0
        mock_submission.score = 5
        mock_submission.num_comments = 2
        mock_submission.upvote_ratio = 0.75
        mock_submission.url = "https://reddit.com/..."
        mock_submission.permalink = "/r/stocks/comments/test123/title/"
        
        result = normalize_post(mock_submission)
        
        assert result['permalink'].startswith("https://www.reddit.com")
        assert "/r/stocks/comments/test123/title/" in result['permalink']


class TestDataStructures:
    """Test data structure validity."""
    
    def test_normalized_post_has_required_fields(self):
        """Normalized post should have all required fields."""
        mock_submission = Mock()
        mock_submission.id = "test"
        mock_submission.title = "Title"
        mock_submission.selftext = "Text"
        mock_submission.author = "user"
        mock_submission.subreddit = "stocks"
        mock_submission.created_utc = 1234567890.0
        mock_submission.score = 5
        mock_submission.num_comments = 2
        mock_submission.upvote_ratio = 0.75
        mock_submission.url = "https://reddit.com/..."
        mock_submission.permalink = "/r/stocks/comments/test/..."
        
        result = normalize_post(mock_submission)
        
        required_fields = [
            'id', 'title', 'selftext', 'author', 'subreddit',
            'created_utc', 'score', 'num_comments', 'upvote_ratio',
            'url', 'permalink'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
    
    def test_normalized_post_types(self):
        """Normalized post fields should have correct types."""
        mock_submission = Mock()
        mock_submission.id = "test"
        mock_submission.title = "Title"
        mock_submission.selftext = "Text"
        mock_submission.author = "user"
        mock_submission.subreddit = "stocks"
        mock_submission.created_utc = 1234567890.0
        mock_submission.score = 5
        mock_submission.num_comments = 2
        mock_submission.upvote_ratio = 0.75
        mock_submission.url = "https://reddit.com/..."
        mock_submission.permalink = "/r/stocks/comments/test/..."
        
        result = normalize_post(mock_submission)
        
        assert isinstance(result['id'], str)
        assert isinstance(result['title'], str)
        assert isinstance(result['selftext'], str)
        assert isinstance(result['author'], str)
        assert isinstance(result['subreddit'], str)
        assert isinstance(result['created_utc'], int)
        assert isinstance(result['score'], int)
        assert isinstance(result['num_comments'], int)
        assert isinstance(result['url'], str)
        assert isinstance(result['permalink'], str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
