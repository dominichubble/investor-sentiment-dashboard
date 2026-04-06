"""Tests for shared Reddit keyword helpers."""

from app.pipelines.reddit_common import hint_tickers_from_keyword_group


def test_hint_tickers_extracts_symbols():
    g = ["nvda", "inflation", "TSLA", "rate hike"]
    assert hint_tickers_from_keyword_group(g) == ["NVDA", "TSLA"]


def test_hint_tickers_dedupes():
    g = ["AAPL", "aapl", "$MSFT", "msft"]
    assert hint_tickers_from_keyword_group(g) == ["AAPL", "MSFT"]


def test_hint_tickers_skips_long_and_invalid():
    g = ["TOOLONG", "not_a_sym!", "BRK.B"]
    assert hint_tickers_from_keyword_group(g) == ["BRK.B"]
