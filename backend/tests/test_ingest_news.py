"""
Unit tests for News API data ingestion pipeline.

Tests cover:
- Text cleaning functionality
- Article normalization
- Query building
- Quality filtering
- Deduplication
- Data structures
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup (flake8: noqa: E402)
from app.pipelines.ingest_news import (  # noqa: E402
    build_query,
    clean_text,
    deduplicate_articles,
    filter_quality_articles,
    normalize_article,
)


class TestCleanText:
    """Test the clean_text function."""

    def test_clean_text_removes_html_tags(self):
        """HTML tags should be removed from text."""
        text = "<p>NVDA earnings <b>beat</b> expectations!</p>"
        result = clean_text(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "NVDA earnings beat expectations" in result

    def test_clean_text_removes_urls(self):
        """URLs should be removed."""
        text = "Check out https://example.com for more info"
        result = clean_text(text)
        assert "https://example.com" not in result
        assert "Check out" in result
        assert "for more info" in result

    def test_clean_text_removes_newsapi_artifacts(self):
        """NewsAPI [+XXX chars] artifacts should be removed."""
        text = "This is the article content [+1234 chars]"
        result = clean_text(text)
        assert "[+1234 chars]" not in result
        assert "This is the article content" in result

    def test_clean_text_removes_removed_markers(self):
        """[Removed] markers should be removed."""
        text = "Some content [Removed] more text"
        result = clean_text(text)
        assert "[Removed]" not in result
        assert "Some content" in result
        assert "more text" in result

    def test_clean_text_normalizes_whitespace(self):
        """Multiple spaces should be collapsed to single space."""
        text = "NVDA    stock    going    up"
        result = clean_text(text)
        assert result == "NVDA stock going up"
        assert "    " not in result

    def test_clean_text_handles_newlines(self):
        """Newlines should be converted to spaces."""
        text = "Line 1\nLine 2\nLine 3"
        result = clean_text(text)
        assert result == "Line 1 Line 2 Line 3"

    def test_clean_text_handles_empty_string(self):
        """Empty strings should return empty string."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_clean_text_handles_none(self):
        """None should return empty string."""
        assert clean_text(None) == ""

    def test_clean_text_complex_article(self):
        """Complex article with multiple elements."""
        text = "<div><p>NVDA earnings! <a href='http://example.com'>Read</a> [+500 chars]</p></div>"
        result = clean_text(text)
        assert "<div>" not in result
        assert "http://example.com" not in result
        assert "[+500 chars]" not in result
        assert "NVDA earnings" in result


class TestBuildQuery:
    """Test the build_query function."""

    def test_build_query_single_word(self):
        """Single word keyword."""
        keywords = ["stocks"]
        result = build_query(keywords)
        assert result == "stocks"

    def test_build_query_multiple_words(self):
        """Multiple keywords should be OR-separated."""
        keywords = ["stocks", "market", "earnings"]
        result = build_query(keywords)
        assert result == "stocks OR market OR earnings"

    def test_build_query_phrases_get_quoted(self):
        """Multi-word phrases should be quoted."""
        keywords = ["stock market", "federal reserve"]
        result = build_query(keywords)
        assert '"stock market"' in result
        assert '"federal reserve"' in result

    def test_build_query_mixed_single_and_phrases(self):
        """Mix of single words and phrases."""
        keywords = ["stocks", "stock market", "earnings"]
        result = build_query(keywords)
        assert "stocks" in result
        assert '"stock market"' in result
        assert "earnings" in result
        assert " OR " in result

    def test_build_query_empty_list(self):
        """Empty keyword list should return empty string."""
        keywords = []
        result = build_query(keywords)
        assert result == ""


class TestNormalizeArticle:
    """Test the normalize_article function."""

    def test_normalize_article_basic_fields(self):
        """Basic article fields should be extracted correctly."""
        mock_article = {
            "source": {"id": "bloomberg", "name": "Bloomberg"},
            "author": "John Doe",
            "title": "NVDA Earnings Beat",
            "description": "Nvidia reported strong Q3 earnings",
            "url": "https://bloomberg.com/article123",
            "urlToImage": "https://bloomberg.com/image.jpg",
            "publishedAt": "2025-11-02T10:00:00Z",
            "content": "Nvidia reported strong Q3 earnings that beat...",
        }

        result = normalize_article(mock_article)

        assert result["source_id"] == "bloomberg"
        assert result["source_name"] == "Bloomberg"
        assert result["author"] == "John Doe"
        assert result["title"] == "NVDA Earnings Beat"
        assert result["url"] == "https://bloomberg.com/article123"
        assert result["published_at"] == "2025-11-02T10:00:00Z"

    def test_normalize_article_cleans_text(self):
        """Article fields should be cleaned."""
        mock_article = {
            "source": {"id": "reuters", "name": "Reuters"},
            "title": "<b>Market Update</b> https://example.com",
            "description": "Stock market    today    rising",
            "content": "Content [+500 chars]",
        }

        result = normalize_article(mock_article)

        assert "<b>" not in result["clean_title"]
        assert "https://example.com" not in result["clean_title"]
        assert "Market Update" in result["clean_title"]
        assert "    " not in result["clean_description"]
        assert "[+500 chars]" not in result["clean_content"]

    def test_normalize_article_missing_fields(self):
        """Missing fields should be handled gracefully."""
        mock_article = {
            "source": {},
            "title": "Test Article",
        }

        result = normalize_article(mock_article)

        assert result["source_id"] is None
        assert result["source_name"] is None
        assert result["author"] is None
        assert result["title"] == "Test Article"
        assert result["description"] == ""
        assert result["content"] == ""


class TestFilterQualityArticles:
    """Test the filter_quality_articles function."""

    def create_article(
        self, title="", description="", content="", title_len=None
    ) -> dict:
        """Helper to create mock article dict."""
        if title_len:
            title = "A" * title_len
        return {
            "source_id": "test",
            "source_name": "Test",
            "title": title,
            "clean_title": clean_text(title),
            "description": description,
            "clean_description": clean_text(description),
            "content": content,
            "clean_content": clean_text(content),
            "url": "https://example.com",
        }

    def test_filter_removes_short_titles(self):
        """Should remove articles with titles < 10 chars."""
        articles = [
            self.create_article(title="Good article title", description="Good desc"),
            self.create_article(title="Short", description="Description"),
        ]
        result = filter_quality_articles(articles)
        assert len(result) == 1
        assert "Good article" in result[0]["clean_title"]

    def test_filter_removes_removed_content(self):
        """Should remove articles marked as [Removed]."""
        articles = [
            self.create_article(
                title="Good article", description="Good", content="A" * 150
            ),
            self.create_article(
                title="Bad article", description="Bad", content="[Removed]"
            ),
        ]
        result = filter_quality_articles(articles)
        assert len(result) == 1
        assert "Good article" in result[0]["clean_title"]
        assert "Good article" in result[0]["clean_title"]

    def test_filter_removes_short_content(self):
        """Should remove articles with content < 100 chars (paywalled)."""
        articles = [
            self.create_article(
                title="Good article", description="Good", content="A" * 150
            ),
            self.create_article(
                title="Paywalled article", description="Pay", content="Short"
            ),
        ]
        result = filter_quality_articles(articles)
        assert len(result) == 1
        assert "Good article" in result[0]["clean_title"]

    def test_filter_removes_empty_content(self):
        """Should remove articles with no description or content."""
        articles = [
            self.create_article(title="Good article", description="Has description"),
            self.create_article(title="Empty article", description="", content=""),
        ]
        result = filter_quality_articles(articles)
        assert len(result) == 1
        assert "Good article" in result[0]["clean_title"]

    def test_filter_keeps_quality_articles(self):
        """Should keep quality articles with good content."""
        articles = [
            self.create_article(
                title="NVDA earnings report",
                description="Nvidia reported strong Q3 earnings",
                content="A" * 200,
            ),
            self.create_article(
                title="Fed rate decision",
                description="Federal Reserve maintains rates",
                content="B" * 200,
            ),
        ]
        result = filter_quality_articles(articles)
        assert len(result) == 2


class TestDeduplicateArticles:
    """Test the deduplicate_articles function."""

    def create_article(self, url="", title="") -> dict:
        """Helper to create mock article dict."""
        return {
            "url": url,
            "title": title,
            "clean_title": clean_text(title),
        }

    def test_deduplicate_by_url(self):
        """Should remove articles with duplicate URLs."""
        articles = [
            self.create_article(
                url="https://example.com/article1", title="First Article"
            ),
            self.create_article(
                url="https://example.com/article1", title="Duplicate Article"
            ),
            self.create_article(
                url="https://example.com/article2", title="Second Article"
            ),
        ]
        result = deduplicate_articles(articles)
        assert len(result) == 2
        assert result[0]["clean_title"] == "First Article"
        assert result[1]["clean_title"] == "Second Article"

    def test_deduplicate_by_title(self):
        """Should remove articles with duplicate titles."""
        articles = [
            self.create_article(
                url="https://site1.com/article", title="Market Update Today"
            ),
            self.create_article(
                url="https://site2.com/article", title="Market Update Today"
            ),
        ]
        result = deduplicate_articles(articles)
        assert len(result) == 1

    def test_deduplicate_keeps_different_articles(self):
        """Should keep articles with different URLs and titles."""
        articles = [
            self.create_article(url="https://example.com/1", title="Article One"),
            self.create_article(url="https://example.com/2", title="Article Two"),
            self.create_article(url="https://example.com/3", title="Article Three"),
        ]
        result = deduplicate_articles(articles)
        assert len(result) == 3

    def test_deduplicate_empty_list(self):
        """Should handle empty list."""
        articles = []
        result = deduplicate_articles(articles)
        assert len(result) == 0


class TestDataStructures:
    """Test data structure validity."""

    def test_normalized_article_has_required_fields(self):
        """Normalized article should have all required fields."""
        mock_article = {
            "source": {"id": "test", "name": "Test"},
            "author": "Author",
            "title": "Title",
            "description": "Description",
            "url": "https://example.com",
            "urlToImage": "https://example.com/image.jpg",
            "publishedAt": "2025-11-02T10:00:00Z",
            "content": "Content",
        }

        result = normalize_article(mock_article)

        required_fields = [
            "source_id",
            "source_name",
            "author",
            "title",
            "description",
            "url",
            "url_to_image",
            "published_at",
            "content",
            "clean_title",
            "clean_description",
            "clean_content",
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_normalized_article_types(self):
        """Normalized article fields should have correct types."""
        mock_article = {
            "source": {"id": "test", "name": "Test"},
            "author": "Author",
            "title": "Title",
            "description": "Description",
            "url": "https://example.com",
            "urlToImage": "https://example.com/image.jpg",
            "publishedAt": "2025-11-02T10:00:00Z",
            "content": "Content",
        }

        result = normalize_article(mock_article)

        assert isinstance(result["source_id"], str)
        assert isinstance(result["source_name"], str)
        assert isinstance(result["author"], str)
        assert isinstance(result["title"], str)
        assert isinstance(result["clean_title"], str)
        assert isinstance(result["url"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
