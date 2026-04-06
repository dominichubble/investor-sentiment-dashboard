"""Tests for lightweight ticker detection."""

from app.utils.ticker_detection import TickerDetector


def _detector():
    return TickerDetector()


def test_cashtags_detected():
    d = _detector()
    result = dict(d.detect("$AAPL is up 3% and $TSLA is down"))
    assert "AAPL" in result
    assert "TSLA" in result


def test_bare_tickers_detected():
    d = _detector()
    result = dict(d.detect("NVDA and AMD are leading the chip race"))
    assert "NVDA" in result
    assert "AMD" in result


def test_company_names_detected():
    d = _detector()
    result = dict(d.detect("Apple reported strong earnings"))
    assert "AAPL" in result
    assert result["AAPL"] == "Apple"


def test_no_false_positives_on_common_words():
    d = _detector()
    result = d.detect("Markets are uncertain today, no clear direction")
    assert result == []


def test_multiple_tickers_per_text():
    d = _detector()
    result = dict(d.detect("$AAPL up, $MSFT down, $GOOGL flat"))
    assert "AAPL" in result
    assert "MSFT" in result
    assert "GOOGL" in result


def test_empty_text_returns_empty():
    d = _detector()
    assert d.detect("") == []
    assert d.detect(None) == []


def test_deduplication():
    d = _detector()
    result = d.detect("AAPL AAPL $AAPL Apple")
    tickers = [t for t, _ in result]
    assert tickers.count("AAPL") == 1


def test_cashtag_dot_normalizes_to_exchange_symbol():
    """$BRK.B style cashtags map to BRK-B in stock_database.json."""
    d = _detector()
    result = dict(d.detect("Berkshire $BRK.B adds to position"))
    assert "BRK-B" in result


def test_normalize_ticker_symbol():
    from app.utils.ticker_detection import normalize_ticker_symbol

    assert normalize_ticker_symbol("$brk.b") == "BRK-B"
    assert normalize_ticker_symbol("  AI  ") == "AI"
