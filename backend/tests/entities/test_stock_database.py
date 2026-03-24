"""
Tests for stock database module.
"""

import json
import tempfile
from pathlib import Path

import pytest

from app.entities.stock_database import StockDatabase


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_stock_data():
    """Mock stock database data."""
    return {
        "stocks": {
            "AAPL": {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "common_names": ["Apple", "AAPL"],
                "exchange": "NASDAQ",
                "cik": "0000320193",
                "is_active": True,
            },
            "TSLA": {
                "ticker": "TSLA",
                "company_name": "Tesla, Inc.",
                "common_names": ["Tesla", "TSLA"],
                "exchange": "NASDAQ",
                "cik": "0001318605",
                "is_active": True,
            },
            "MSFT": {
                "ticker": "MSFT",
                "company_name": "Microsoft Corporation",
                "common_names": ["Microsoft", "MSFT"],
                "exchange": "NASDAQ",
                "cik": "0000789019",
                "is_active": True,
            },
        },
        "metadata": {"total_stocks": 3, "source": "SEC EDGAR"},
    }


@pytest.fixture
def stock_db_with_data(temp_dir, mock_stock_data):
    """Stock database with pre-loaded data."""
    db_file = temp_dir / "stock_database.json"
    with open(db_file, "w") as f:
        json.dump(mock_stock_data, f)

    db = StockDatabase(data_dir=temp_dir)
    db.load()
    return db


class TestStockDatabase:
    """Test suite for StockDatabase class."""

    def test_init(self, temp_dir):
        """Test database initialization."""
        db = StockDatabase(data_dir=temp_dir)
        assert db.data_dir == temp_dir
        assert not db._loaded

    def test_load_existing_database(self, stock_db_with_data):
        """Test loading existing database file."""
        db = stock_db_with_data
        assert db._loaded
        assert len(db._stocks) == 3
        assert "AAPL" in db._stocks
        assert "TSLA" in db._stocks

    def test_get_by_ticker(self, stock_db_with_data):
        """Test getting stock by ticker symbol."""
        db = stock_db_with_data

        # Test existing ticker
        apple = db.get_by_ticker("AAPL")
        assert apple is not None
        assert apple["ticker"] == "AAPL"
        assert apple["company_name"] == "Apple Inc."

        # Test case insensitivity
        apple2 = db.get_by_ticker("aapl")
        assert apple2 is not None
        assert apple2["ticker"] == "AAPL"

        # Test non-existent ticker
        assert db.get_by_ticker("INVALID") is None

    def test_get_by_name(self, stock_db_with_data):
        """Test getting stock by company name."""
        db = stock_db_with_data

        # Test by full company name
        apple = db.get_by_name("Apple Inc.")
        assert apple is not None
        assert apple["ticker"] == "AAPL"

        # Test by common name
        apple2 = db.get_by_name("Apple")
        assert apple2 is not None
        assert apple2["ticker"] == "AAPL"

        # Test case insensitivity
        apple3 = db.get_by_name("apple")
        assert apple3 is not None

        # Test non-existent name
        assert db.get_by_name("Invalid Company") is None

    def test_search(self, stock_db_with_data):
        """Test searching for stocks."""
        db = stock_db_with_data

        # Search by partial name
        results = db.search("Apple")
        assert len(results) >= 1
        assert any(s["ticker"] == "AAPL" for s in results)

        # Search by ticker
        results = db.search("AAPL")
        assert len(results) >= 1
        assert results[0]["ticker"] == "AAPL"

        # Search with limit
        results = db.search("", limit=2)
        assert len(results) <= 2

    def test_get_all_tickers(self, stock_db_with_data):
        """Test getting all tickers."""
        db = stock_db_with_data
        tickers = db.get_all_tickers()

        assert len(tickers) == 3
        assert "AAPL" in tickers
        assert "TSLA" in tickers
        assert "MSFT" in tickers

    def test_get_total_stocks(self, stock_db_with_data):
        """Test getting total stock count."""
        db = stock_db_with_data
        total = db.get_total_stocks()
        assert total == 3

    def test_extract_short_name(self, temp_dir):
        """Test extracting short company names."""
        db = StockDatabase(data_dir=temp_dir)

        assert db._extract_short_name("Apple Inc.") == "Apple"
        assert db._extract_short_name("Microsoft Corporation") == "Microsoft"
        assert db._extract_short_name("Tesla, Inc.") == "Tesla,"
        assert (
            db._extract_short_name("Goldman Sachs Group, Inc.")
            == "Goldman Sachs Group,"
        )

    def test_save_and_load(self, temp_dir):
        """Test saving and loading database."""
        db = StockDatabase(data_dir=temp_dir)
        db._stocks = {
            "TEST": {
                "ticker": "TEST",
                "company_name": "Test Company",
                "common_names": ["Test"],
                "exchange": "NASDAQ",
                "is_active": True,
            }
        }
        db._build_name_index()
        db.save()

        # Load in new instance
        db2 = StockDatabase(data_dir=temp_dir)
        db2.load()

        assert db2.get_by_ticker("TEST") is not None
        assert db2.get_by_name("Test Company") is not None
