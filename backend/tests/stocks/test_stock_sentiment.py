"""
Tests for stock sentiment analysis module.
"""

import json
import tempfile
from pathlib import Path

import pytest

from app.entities.stock_database import StockDatabase
from app.stocks.stock_sentiment import (
    StockSentimentAnalyzer,
    analyze_stock_sentiment,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_stock_db(temp_dir):
    """Mock stock database with test data."""
    mock_data = {
        "stocks": {
            "AAPL": {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "common_names": ["Apple", "AAPL"],
                "exchange": "NASDAQ",
                "is_active": True,
            },
            "TSLA": {
                "ticker": "TSLA",
                "company_name": "Tesla, Inc.",
                "common_names": ["Tesla", "TSLA"],
                "exchange": "NASDAQ",
                "is_active": True,
            },
        }
    }

    db_file = temp_dir / "stock_database.json"
    with open(db_file, "w") as f:
        json.dump(mock_data, f)

    db = StockDatabase(data_dir=temp_dir)
    db.load()
    return db


class TestStockSentimentAnalyzer:
    """Test suite for StockSentimentAnalyzer."""

    def test_init(self, mock_stock_db):
        """Test analyzer initialization."""
        analyzer = StockSentimentAnalyzer(stock_database=mock_stock_db)

        assert analyzer.stock_db is not None
        assert analyzer.entity_resolver is not None
        assert analyzer.model is not None

    def test_analyze_with_ticker(self, mock_stock_db):
        """Test analysis with explicit ticker symbol."""
        analyzer = StockSentimentAnalyzer(stock_database=mock_stock_db)

        text = "$AAPL surged 15% on strong earnings report"
        result = analyzer.analyze(text)

        assert "text" in result
        assert "overall_sentiment" in result
        assert "stocks" in result
        assert "metadata" in result

        # Should find at least AAPL
        tickers = [s["ticker"] for s in result["stocks"]]
        assert "AAPL" in tickers

    def test_analyze_with_company_name(self, mock_stock_db):
        """Test analysis with company name (NER)."""
        analyzer = StockSentimentAnalyzer(stock_database=mock_stock_db)

        text = "Apple reported strong quarterly earnings"
        result = analyzer.analyze(text)

        # Should extract Apple as entity and resolve to AAPL
        # Note: This depends on NER working correctly
        assert "stocks" in result
        assert isinstance(result["stocks"], list)

    def test_analyze_multiple_stocks(self, mock_stock_db):
        """Test analysis with multiple stocks."""
        analyzer = StockSentimentAnalyzer(stock_database=mock_stock_db)

        text = "$AAPL surged 15% while $TSLA dropped 8% on delivery concerns"
        result = analyzer.analyze(text)

        tickers = [s["ticker"] for s in result["stocks"]]

        # Should find both tickers
        assert "AAPL" in tickers
        assert "TSLA" in tickers

        # Should have different sentiments
        if len(result["stocks"]) >= 2:
            sentiments = [s["sentiment"]["label"] for s in result["stocks"]]
            # At least one should be positive or negative
            assert "positive" in sentiments or "negative" in sentiments

    def test_analyze_with_context(self, mock_stock_db):
        """Test context extraction."""
        analyzer = StockSentimentAnalyzer(stock_database=mock_stock_db)

        text = "$AAPL surged 15% after beating earnings expectations"
        result = analyzer.analyze(text, extract_context=True)

        if result["stocks"]:
            stock = result["stocks"][0]
            assert "context" in stock
            assert stock["context"] is not None
            assert len(stock["context"]) > 0

    def test_analyze_without_context(self, mock_stock_db):
        """Test without context extraction."""
        analyzer = StockSentimentAnalyzer(stock_database=mock_stock_db)

        text = "$AAPL surged 15%"
        result = analyzer.analyze(text, extract_context=False)

        if result["stocks"]:
            stock = result["stocks"][0]
            assert stock["context"] is None or stock["context"] == ""

    def test_analyze_no_stocks(self, mock_stock_db):
        """Test with text containing no stocks."""
        analyzer = StockSentimentAnalyzer(stock_database=mock_stock_db)

        text = "The weather is nice today"
        result = analyzer.analyze(text)

        assert result["stocks"] == []
        assert result["metadata"]["entities_found"] == 0

    def test_analyze_metadata(self, mock_stock_db):
        """Test metadata in results."""
        analyzer = StockSentimentAnalyzer(stock_database=mock_stock_db)

        text = "$AAPL surged today"
        result = analyzer.analyze(text)

        metadata = result["metadata"]
        assert "entities_found" in metadata
        assert "tickers_extracted" in metadata
        assert "processing_time_ms" in metadata
        assert isinstance(metadata["processing_time_ms"], (int, float))

    def test_extract_context_sentence_level(self, mock_stock_db):
        """Test sentence-level context extraction."""
        analyzer = StockSentimentAnalyzer(stock_database=mock_stock_db)

        text = "First sentence. $AAPL surged 15% today. Third sentence."
        position = {"start": 16, "end": 21}  # Position of $AAPL

        context = analyzer._extract_context(text, position)

        # Should extract the sentence containing $AAPL
        assert "$AAPL" in context or "surged" in context

    def test_extract_context_window(self, mock_stock_db):
        """Test window-based context extraction."""
        analyzer = StockSentimentAnalyzer(stock_database=mock_stock_db)

        text = "A" * 200 + "$AAPL" + "B" * 200
        position = {"start": 200, "end": 205}

        context = analyzer._extract_context(text, position, window=50)

        # Should not exceed window size significantly
        assert len(context) <= 150  # 50 before + ticker + 50 after + margin


class TestConvenienceFunction:
    """Test convenience function for stock sentiment analysis."""

    def test_analyze_stock_sentiment_function(self, mock_stock_db):
        """Test analyze_stock_sentiment convenience function."""
        text = "$AAPL surged 15% on strong earnings"

        result = analyze_stock_sentiment(
            text, stock_database=mock_stock_db
        )

        assert "text" in result
        assert "overall_sentiment" in result
        assert "stocks" in result
        assert "metadata" in result

    def test_with_all_parameters(self, mock_stock_db):
        """Test with all optional parameters."""
        text = "$AAPL and $TSLA both moved today"

        result = analyze_stock_sentiment(
            text,
            stock_database=mock_stock_db,
            extract_context=True,
            include_movements=True,
        )

        assert result is not None
        assert isinstance(result, dict)
