"""
Unit tests for Twitter data ingestion pipeline.

Tests cover:
- Text cleaning functionality
- Tweet normalization
- Query building
- Spam/bot detection
- Engagement filtering
- Data structures
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup (flake8: noqa: E402)
from app.pipelines.ingest_twitter import (  # noqa: E402
    build_keyword_query,
    build_ticker_search_query,
    clean_text,
    extract_cashtags,
    fetch_tweets_csv,
    filter_by_engagement,
    filter_low_quality_tweets,
    normalize_csv_finance_row,
    normalize_snscrape_tweet,
    normalize_tweet,
    resolve_ingest_backend,
)


class TestCleanText:
    """Test the clean_text function."""

    def test_clean_text_removes_urls(self):
        """URLs should be removed from text."""
        text = "Check out TSLA! https://example.com is bullish"
        result = clean_text(text)
        assert "https://example.com" not in result
        assert "Check out TSLA" in result
        assert "is bullish" in result

    def test_clean_text_removes_mentions(self):
        """@mentions should be removed."""
        text = "Hey @elonmusk what about $TSLA earnings?"
        result = clean_text(text)
        assert "@elonmusk" not in result
        assert "Hey" in result
        assert "what about" in result

    def test_clean_text_removes_hashtag_symbol_but_keeps_word(self):
        """Hashtags should be converted to plain text (keep word, remove #)."""
        text = "Bullish on #stocks and #trading today"
        result = clean_text(text)
        assert "#" not in result
        assert "stocks" in result
        assert "trading" in result

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

    def test_clean_text_handles_empty_string(self):
        """Empty strings should return empty string."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_clean_text_handles_none(self):
        """None should return empty string."""
        assert clean_text(None) == ""

    def test_clean_text_complex_tweet(self):
        """Complex tweet with multiple elements."""
        text = "🚀 @trader Check #NVDA  https://t.co/abc  earnings beat! #bullish"
        result = clean_text(text)
        assert "@trader" not in result
        assert "https://t.co/" not in result
        assert "#" not in result
        assert "NVDA" in result
        assert "earnings beat" in result


class TestBuildKeywordQuery:
    """Test build_keyword_query (free-text search mode)."""

    def test_build_keyword_query_single_word(self):
        keywords = ["stocks"]
        result = build_keyword_query(keywords)
        assert "stocks" in result
        assert "lang:en" in result
        assert "-is:retweet" in result

    def test_build_keyword_query_multiple_words(self):
        keywords = ["stocks", "market", "earnings"]
        result = build_keyword_query(keywords)
        assert "stocks OR market OR earnings" in result

    def test_build_keyword_query_phrases_get_quoted(self):
        keywords = ["stock market", "fed rate"]
        result = build_keyword_query(keywords)
        assert '"stock market"' in result
        assert '"fed rate"' in result

    def test_build_keyword_query_excludes_retweets(self):
        keywords = ["stocks"]
        result = build_keyword_query(keywords)
        assert "-is:retweet" in result

    def test_build_keyword_query_includes_language(self):
        keywords = ["stocks"]
        result = build_keyword_query(keywords, lang="en")
        assert "lang:en" in result


class TestBuildTickerSearchQuery:
    """Cashtag recent-search query."""

    def test_build_ticker_query_or_cashtags(self):
        q = build_ticker_search_query(["NVDA", "BTC"], lang="en")
        assert "($NVDA OR $BTC)" in q
        assert "lang:en" in q
        assert "-is:retweet" in q

    def test_build_ticker_query_strips_dollar(self):
        q = build_ticker_search_query(["$GOOGL", "META"], lang="en")
        assert "$GOOGL" in q or "GOOGL" in q
        assert "$META" in q


class TestExtractCashtags:
    def test_extract_cashtags_unique_order(self):
        assert extract_cashtags("Long $NVDA and $BTC, short $NVDA") == ["NVDA", "BTC"]


class TestResolveIngestBackend:
    def test_default_snscrape(self, monkeypatch):
        monkeypatch.delenv("TWITTER_INGEST_BACKEND", raising=False)
        assert resolve_ingest_backend() == "snscrape"

    def test_api_aliases(self, monkeypatch):
        for v in ("api", "official", "tweepy"):
            monkeypatch.setenv("TWITTER_INGEST_BACKEND", v)
            assert resolve_ingest_backend() == "api"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TWITTER_INGEST_BACKEND", "api")
        assert resolve_ingest_backend("snscrape") == "snscrape"

    def test_csv_aliases(self, monkeypatch):
        for v in ("csv", "file", "local", "dataset"):
            monkeypatch.setenv("TWITTER_INGEST_BACKEND", v)
            assert resolve_ingest_backend() == "csv"

    def test_auto_uses_api_when_token(self, monkeypatch):
        monkeypatch.setenv("TWITTER_INGEST_BACKEND", "auto")
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "x")
        assert resolve_ingest_backend() == "api"

    def test_auto_falls_back_snscrape_without_token(self, monkeypatch):
        monkeypatch.setenv("TWITTER_INGEST_BACKEND", "auto")
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
        assert resolve_ingest_backend() == "snscrape"


class TestNormalizeCsvFinanceRow:
    def test_maps_stock_and_cashtags(self):
        row = {
            "Date": "2022-01-01T00:00:00+00:00",
            "Tweet": "Long $NVDA and $AAPL",
            "Stock Name": "NVDA",
        }
        out = normalize_csv_finance_row(row, 1)
        assert out["hint_tickers"][0] == "NVDA"
        assert "NVDA" in out["raw_text"]
        assert out["like_count"] == 0
        assert out["data_source"] == "twitter"


class TestFetchTweetsCsv:
    def test_reads_rows(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text(
            "Date,Tweet,Stock Name\n"
            "2022-01-01T00:00:00+00:00,\"Hello $TSLA world with enough chars\",TSLA\n",
            encoding="utf-8",
        )
        out = fetch_tweets_csv(
            max_results=5,
            lang="en",
            min_engagement=0,
            csv_path=p,
        )
        assert len(out) == 1
        assert "TSLA" in (out[0].get("hint_tickers") or [])


class TestNormalizeSnscrapeTweet:
    def test_maps_fields(self):
        dt = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        item = SimpleNamespace(
            rawContent="Long $NVDA",
            cashtags=["NVDA"],
            user=SimpleNamespace(id=999),
            date=dt,
            id=1234567890123456789,
            retweetCount=1,
            replyCount=2,
            likeCount=10,
            quoteCount=0,
            lang="en",
        )
        out = normalize_snscrape_tweet(item)
        assert out["hint_tickers"] == ["NVDA"]
        assert out["like_count"] == 10
        assert out["author_id"] == "999"
        assert "2025-06-01" in (out["created_at"] or "")
        assert out["data_source"] == "twitter"


class TestNormalizeTweet:
    """Test the normalize_tweet function."""

    def test_normalize_tweet_basic_fields(self):
        """Basic tweet fields should be extracted correctly."""
        mock_tweet = Mock()
        mock_tweet.id = 123456789
        mock_tweet.text = "TSLA earnings beat estimates!"
        mock_tweet.author_id = "user123"
        mock_tweet.created_at = Mock()
        mock_tweet.created_at.isoformat.return_value = "2025-10-28T10:00:00"
        mock_tweet.public_metrics = {
            "retweet_count": 10,
            "reply_count": 5,
            "like_count": 50,
            "quote_count": 2,
        }
        mock_tweet.lang = "en"

        result = normalize_tweet(mock_tweet)

        assert result["id"] == 123456789
        assert result["text"] == "TSLA earnings beat estimates!"
        assert result["raw_text"] == "TSLA earnings beat estimates!"
        assert result["author_id"] == "user123"
        assert result["created_at"] == "2025-10-28T10:00:00"
        assert result["retweet_count"] == 10
        assert result["reply_count"] == 5
        assert result["like_count"] == 50
        assert result["quote_count"] == 2
        assert result["lang"] == "en"

    def test_normalize_tweet_cleans_text(self):
        """Tweet text should be cleaned."""
        mock_tweet = Mock()
        mock_tweet.id = 123
        mock_tweet.text = "Check @user https://example.com #stocks"
        mock_tweet.author_id = "user1"
        mock_tweet.public_metrics = {}
        mock_tweet.lang = "en"

        result = normalize_tweet(mock_tweet)

        assert "@user" not in result["text"]
        assert "https://example.com" not in result["text"]
        assert "#" not in result["text"]
        assert "Check" in result["text"]
        assert "stocks" in result["text"]
        assert result["hint_tickers"] == []


class TestFilterLowQualityTweets:
    """Test filter_low_quality_tweets (light spam / noise removal)."""

    def create_tweet(self, text, raw_text=None, likes=0, retweets=0, replies=0) -> dict:
        """Helper to create mock tweet dict."""
        return {
            "id": 123,
            "text": text,
            "raw_text": raw_text or text,
            "author_id": "user1",
            "created_at": "2025-10-28T10:00:00",
            "retweet_count": retweets,
            "reply_count": replies,
            "like_count": likes,
            "quote_count": 0,
            "lang": "en",
        }

    def test_filter_removes_follow_spam(self):
        """Should remove 'follow me' spam."""
        tweets = [
            self.create_tweet("Great market analysis today"),
            self.create_tweet("Follow me for stock tips!"),
        ]
        result = filter_low_quality_tweets(tweets)
        assert len(result) == 1
        assert "analysis" in result[0]["text"]

    def test_filter_removes_blogger_promotion_spam(self):
        """Should remove blogger recommendation spam (real example)."""
        tweets = [
            self.create_tweet("NVDA earnings beat expectations"),
            self.create_tweet("This blogger recommends stocks that rise every day"),
        ]
        result = filter_low_quality_tweets(tweets)
        assert len(result) == 1
        assert "NVDA" in result[0]["text"]

    def test_filter_removes_make_money_spam(self):
        """Should remove 'make money every day' spam."""
        tweets = [
            self.create_tweet("Market volatility increasing"),
            self.create_tweet("Buy the stocks he recommends and make money every day"),
        ]
        result = filter_low_quality_tweets(tweets)
        assert len(result) == 1
        assert "volatility" in result[0]["text"]

    def test_filter_keeps_uniform_engagement(self):
        """Uniform engagement is no longer treated as a bot signal."""
        tweets = [
            self.create_tweet(
                "Real market discussion", likes=10, retweets=5, replies=3
            ),
            self.create_tweet("Also valid take", likes=9, retweets=9, replies=9),
        ]
        result = filter_low_quality_tweets(tweets)
        assert len(result) == 2

    def test_filter_removes_excessive_emojis(self):
        """Should remove tweets with too many emojis (regex-counted glyphs)."""
        spam_raw = ("🚀" * 10) + " buy now"  # 10 × U+1F680 range matches filter
        tweets = [
            self.create_tweet("TSLA stock analysis is positive today"),
            self.create_tweet("rocket spam", raw_text=spam_raw),
        ]
        result = filter_low_quality_tweets(tweets)
        assert len(result) == 1
        assert "TSLA" in result[0]["text"]

    def test_filter_removes_excessive_hashtags(self):
        """Should remove tweets with too many hashtags."""
        tags = " ".join(f"#{i}" for i in range(13))
        tweets = [
            self.create_tweet("Market update today shows strong momentum"),
            self.create_tweet("x", raw_text=f"{tags} spam here"),
        ]
        result = filter_low_quality_tweets(tweets)
        assert len(result) == 1
        assert "update" in result[0]["text"]

    def test_filter_removes_excessive_cashtags(self):
        """Should remove tweets with excessive $ symbols."""
        cash = " ".join(f"${c}" for c in "ABCDEFGHIJKLMNOP")
        tweets = [
            self.create_tweet("NVDA earnings strong"),
            self.create_tweet("x", raw_text=f"Pump {cash} now"),
        ]
        result = filter_low_quality_tweets(tweets)
        assert len(result) == 1
        assert "NVDA" in result[0]["text"]

    def test_filter_removes_short_tweets(self):
        """Should remove tweets that are too short."""
        tweets = [
            self.create_tweet("Fed raises rates by 25 basis points today"),
            self.create_tweet("Buy now!"),
        ]
        result = filter_low_quality_tweets(tweets)
        assert len(result) == 1
        assert "Fed" in result[0]["text"]

    def test_filter_keeps_short_cashtag_tweet(self):
        """Cashtag posts can be shorter than generic tweets."""
        tweets = [
            self.create_tweet(
                "$NVDA ripping",
                raw_text="$NVDA ripping",
                likes=0,
                retweets=0,
                replies=0,
            ),
        ]
        result = filter_low_quality_tweets(tweets)
        assert len(result) == 1

    def test_filter_keeps_quality_tweets(self):
        """Should keep quality tweets."""
        tweets = [
            self.create_tweet(
                "Nvidia reports Q3 earnings: revenue up 40% YoY, beating estimates"
            ),
            self.create_tweet(
                "Fed minutes show concern over persistent inflation pressures"
            ),
        ]
        result = filter_low_quality_tweets(tweets)
        assert len(result) == 2


class TestFilterByEngagement:
    """Test the filter_by_engagement function."""

    def create_tweet(self, likes=0, retweets=0, replies=0) -> dict:
        """Helper to create mock tweet dict."""
        return {
            "id": 123,
            "text": "Test tweet",
            "raw_text": "Test tweet",
            "like_count": likes,
            "retweet_count": retweets,
            "reply_count": replies,
        }

    def test_filter_keeps_high_engagement(self):
        """Should keep tweets meeting engagement threshold."""
        tweets = [
            self.create_tweet(likes=10, retweets=5, replies=3),  # Total: 18
            self.create_tweet(likes=1, retweets=0, replies=0),  # Total: 1
        ]
        result = filter_by_engagement(tweets, min_engagement=5)
        assert len(result) == 1
        assert result[0]["like_count"] == 10

    def test_filter_at_exact_threshold(self):
        """Should keep tweets at exact threshold."""
        tweets = [
            self.create_tweet(likes=3, retweets=1, replies=1),  # Total: 5
        ]
        result = filter_by_engagement(tweets, min_engagement=5)
        assert len(result) == 1

    def test_filter_below_threshold(self):
        """Should remove tweets below threshold."""
        tweets = [
            self.create_tweet(likes=2, retweets=1, replies=1),  # Total: 4
        ]
        result = filter_by_engagement(tweets, min_engagement=5)
        assert len(result) == 0


class TestDataStructures:
    """Test data structure validity."""

    def test_normalized_tweet_has_required_fields(self):
        """Normalized tweet should have all required fields."""
        mock_tweet = Mock()
        mock_tweet.id = 123
        mock_tweet.text = "Test"
        mock_tweet.author_id = "user1"
        mock_tweet.created_at = Mock()
        mock_tweet.created_at.isoformat.return_value = "2025-10-28T10:00:00"
        mock_tweet.public_metrics = {
            "retweet_count": 1,
            "reply_count": 2,
            "like_count": 3,
            "quote_count": 0,
        }
        mock_tweet.lang = "en"

        result = normalize_tweet(mock_tweet)

        required_fields = [
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

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_normalized_tweet_types(self):
        """Normalized tweet fields should have correct types."""
        mock_tweet = Mock()
        mock_tweet.id = 123456789
        mock_tweet.text = "Test tweet"
        mock_tweet.author_id = "user123"
        mock_tweet.created_at = Mock()
        mock_tweet.created_at.isoformat.return_value = "2025-10-28T10:00:00"
        mock_tweet.public_metrics = {
            "retweet_count": 5,
            "reply_count": 3,
            "like_count": 10,
            "quote_count": 1,
        }
        mock_tweet.lang = "en"

        result = normalize_tweet(mock_tweet)

        assert isinstance(result["id"], int)
        assert isinstance(result["text"], str)
        assert isinstance(result["raw_text"], str)
        assert isinstance(result["author_id"], str)
        assert isinstance(result["created_at"], str)
        assert isinstance(result["retweet_count"], int)
        assert isinstance(result["reply_count"], int)
        assert isinstance(result["like_count"], int)
        assert isinstance(result["quote_count"], int)
        assert isinstance(result["lang"], str)
        assert isinstance(result["hint_tickers"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
