"""Felix Drinkall Yahoo Finance news → record mapping."""

from pathlib import Path

from app.pipelines.ingest_felixdrinkall_yahoo_news import (
    _year_from_archive_name,
    article_to_record,
)
from app.utils.ticker_detection import TickerDetector


def test_article_to_record_maps_hints_and_text():
    detector = TickerDetector.get_instance()
    art = {
        "title": "Google is making it easier to plan your night",
        "maintext": "Google (GOOG, GOOGL) wants to make searching easier.",
        "description": "Short summary",
        "url": "http://finance.yahoo.com/news/google-search-update-134232625.html",
        "language": "en",
        "date_publish": "2017-03-21 13:42:32",
        "news_outlet": "finance.yahoo.com",
        "mentioned_companies": ["GOOGL"],
        "named_entities": [],
    }
    rec = article_to_record(
        art,
        detector,
        english_only=True,
        include_related=False,
        max_related=5,
    )
    assert rec is not None
    assert rec["data_source"] == "news"
    assert rec["source_name"] == "finance.yahoo.com"
    assert "GOOGL" in rec.get("hint_tickers", [])
    assert "Google" in rec["title"]
    assert "GOOGL" in rec["selftext"]


def test_archive_paths_sort_newest_year_first():
    paths = [
        Path("2017_processed.json.xz"),
        Path("2023_processed.json.xz"),
        Path("2020_processed.json.xz"),
    ]
    paths.sort(key=_year_from_archive_name, reverse=True)
    assert [p.name[:4] for p in paths] == ["2023", "2020", "2017"]


def test_article_to_record_skips_non_english_when_filtered():
    detector = TickerDetector.get_instance()
    art = {
        "title": "Hola",
        "maintext": "Mucho texto en español para probar el filtro de idioma.",
        "language": "es",
        "url": "http://example.com/a",
        "date_publish": "2020-01-01 12:00:00",
    }
    assert (
        article_to_record(
            art,
            detector,
            english_only=True,
            include_related=False,
            max_related=5,
        )
        is None
    )
