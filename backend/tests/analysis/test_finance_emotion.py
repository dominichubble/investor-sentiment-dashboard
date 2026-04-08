"""Tests for finance-specific emotion scoring."""

from app.analysis.finance_emotion import infer_finance_emotion


def test_finance_emotion_detects_fear_from_negative_risk_language():
    result = infer_finance_emotion(
        text="Shares plunged after a profit warning and lawsuit concerns.",
        sentiment_label="negative",
        sentiment_score=0.9,
        scores={"positive": 0.03, "negative": 0.9, "neutral": 0.07},
        uncertainty=0.25,
        aspects=[{"aspect": "risk_liquidity", "label": "negative"}],
    )

    assert result["label"] == "fear"
    assert result["scores"]["fear"] > result["scores"]["optimism"]


def test_finance_emotion_uses_mixed_for_high_entropy_conflict():
    result = infer_finance_emotion(
        text="Strong demand, but valuation looks stretched and the outlook is uncertain.",
        sentiment_label="positive",
        sentiment_score=0.44,
        scores={"positive": 0.44, "negative": 0.28, "neutral": 0.28},
        uncertainty=0.9,
        aspects=[
            {"aspect": "growth_demand", "label": "positive"},
            {"aspect": "valuation", "label": "negative"},
        ],
    )

    assert result["label"] == "mixed"
    assert "split" in result["rationale"].lower() or "mixed" in result["rationale"].lower()
