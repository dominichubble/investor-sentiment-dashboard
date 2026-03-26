"""Tests for FinBERT-derived rationale and aspect extraction."""

from app.analysis.financial_sentiment_enrichment import (
    build_rationale,
    extract_aspect_snippets,
    normalized_label_entropy,
)


def test_entropy_uniform_is_high():
    scores = {"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3}
    assert normalized_label_entropy(scores) > 0.99


def test_entropy_one_hot_is_low():
    scores = {"positive": 0.97, "negative": 0.02, "neutral": 0.01}
    assert normalized_label_entropy(scores) < 0.25


def test_rationale_mentions_distribution():
    scores = {"positive": 0.5, "negative": 0.35, "neutral": 0.15}
    ent = normalized_label_entropy(scores)
    text = build_rationale("positive", 0.5, scores, ent)
    assert "FinBERT" in text
    assert "50%" in text or "0.5" in text


def test_aspect_snippets_find_revenue():
    text = "Q3 revenue missed expectations but user growth remains strong."
    aspects = extract_aspect_snippets(text)
    aspects_set = {a["aspect"] for a in aspects}
    assert "revenue_earnings" in aspects_set
    assert "growth_demand" in aspects_set
