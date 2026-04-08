"""Finance-specific emotion scoring layered on top of FinBERT outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

EMOTION_LABELS = (
    "fear",
    "optimism",
    "uncertainty",
    "confidence",
    "skepticism",
    "mixed",
)

_LEXICONS: dict[str, tuple[str, ...]] = {
    "fear": (
        "selloff",
        "panic",
        "crash",
        "plunge",
        "slump",
        "wipeout",
        "bankruptcy",
        "default",
        "downgrade",
        "warning",
        "miss",
        "loss",
        "investigation",
        "lawsuit",
        "recession",
        "tariff",
        "bearish",
    ),
    "optimism": (
        "beat",
        "surge",
        "rally",
        "upside",
        "bullish",
        "momentum",
        "growth",
        "breakout",
        "strong demand",
        "tailwind",
        "rebound",
        "opportunity",
    ),
    "uncertainty": (
        "uncertain",
        "uncertainty",
        "volatility",
        "mixed",
        "unclear",
        "wait and see",
        "cautious",
        "guidance cut",
        "guidance withdrawn",
        "headwind",
        "risk",
        "question mark",
        "depends",
        "watching",
    ),
    "confidence": (
        "confident",
        "conviction",
        "resilient",
        "execute",
        "outperform",
        "strong balance sheet",
        "priced in",
        "durable",
        "leadership",
        "dominant",
        "cash flow",
        "moat",
        "raised guidance",
    ),
    "skepticism": (
        "skeptical",
        "overvalued",
        "hype",
        "bubble",
        "doubt",
        "questionable",
        "stretched",
        "priced for perfection",
        "too expensive",
        "fade",
        "disconnected",
        "not convinced",
    ),
}

_ASPECT_PRIORS: dict[str, dict[str, float]] = {
    "revenue_earnings": {"optimism": 0.12, "confidence": 0.06, "skepticism": 0.04},
    "growth_demand": {"optimism": 0.12, "confidence": 0.08},
    "risk_liquidity": {"fear": 0.2, "uncertainty": 0.1, "skepticism": 0.08},
    "valuation": {"skepticism": 0.16, "confidence": 0.08, "optimism": 0.05},
    "product_ai": {"optimism": 0.1, "confidence": 0.08, "skepticism": 0.04},
    "macro_policy": {"uncertainty": 0.16, "fear": 0.08, "skepticism": 0.04},
}


def _count_phrase_hits(text: str, phrases: Iterable[str]) -> int:
    low = text.lower()
    hits = 0
    for phrase in phrases:
        if " " in phrase:
            if phrase in low:
                hits += 1
        elif re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", low):
            hits += 1
    return hits


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 0:
        return {k: round(1.0 / len(scores), 4) for k in scores}
    return {k: round(max(v, 0.0) / total, 4) for k, v in scores.items()}


def _fallback_distribution(sentiment_label: str, sentiment_score: float) -> dict[str, float]:
    rest = max(0.0, (1.0 - sentiment_score) / 2.0)
    out = {"positive": rest, "negative": rest, "neutral": rest}
    if sentiment_label in out:
        out[sentiment_label] = sentiment_score
    return out


def infer_finance_emotion(
    *,
    text: str,
    sentiment_label: str,
    sentiment_score: float,
    scores: dict[str, float] | None = None,
    uncertainty: float | None = None,
    aspects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Infer a dominant finance emotion plus score distribution."""
    probs = scores or _fallback_distribution(sentiment_label, sentiment_score)
    entropy = float(uncertainty or 0.0)
    aspects = aspects or []

    raw_scores = {
        "fear": 0.05,
        "optimism": 0.05,
        "uncertainty": 0.05,
        "confidence": 0.05,
        "skepticism": 0.05,
        "mixed": 0.03,
    }

    pos = float(probs.get("positive", 0.0))
    neg = float(probs.get("negative", 0.0))
    neu = float(probs.get("neutral", 0.0))

    raw_scores["optimism"] += pos * 0.65
    raw_scores["confidence"] += pos * 0.35
    raw_scores["fear"] += neg * 0.55
    raw_scores["skepticism"] += neg * 0.35
    raw_scores["uncertainty"] += neu * 0.2 + entropy * 0.65
    raw_scores["mixed"] += max(0.0, entropy - 0.72) * 0.9

    if pos >= 0.58 and entropy <= 0.55:
        raw_scores["confidence"] += 0.12
    if neg >= 0.58 and entropy <= 0.55:
        raw_scores["fear"] += 0.12
    if entropy >= 0.78:
        raw_scores["uncertainty"] += 0.2
        raw_scores["mixed"] += 0.18

    text_low = text.lower()
    lexicon_hits: dict[str, int] = {}
    for emotion, phrases in _LEXICONS.items():
        hits = _count_phrase_hits(text_low, phrases)
        lexicon_hits[emotion] = hits
        raw_scores[emotion] += hits * 0.16

    for aspect in aspects:
        aspect_name = str(aspect.get("aspect", "")).strip().lower()
        priors = _ASPECT_PRIORS.get(aspect_name)
        if not priors:
            continue
        for emotion, bonus in priors.items():
            raw_scores[emotion] += bonus

        snippet_label = str(aspect.get("label") or "").lower()
        if snippet_label == "negative":
            raw_scores["fear"] += 0.05
            raw_scores["skepticism"] += 0.04
        elif snippet_label == "positive":
            raw_scores["optimism"] += 0.05
            raw_scores["confidence"] += 0.04
        elif snippet_label == "neutral":
            raw_scores["uncertainty"] += 0.03

    if lexicon_hits["fear"] and lexicon_hits["optimism"]:
        raw_scores["mixed"] += 0.12
    if lexicon_hits["uncertainty"] >= 2:
        raw_scores["uncertainty"] += 0.08

    normalized = _normalize(raw_scores)
    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1]

    if top_score < 0.24 or (top_score - second_score) < 0.06 or entropy >= 0.88:
        dominant = "mixed"
    else:
        dominant = top_label

    reasons: List[str] = []
    if dominant == "mixed":
        reasons.append("Signals were split across multiple finance emotions.")
    else:
        reasons.append(f"Dominant finance emotion is {dominant}.")
    if entropy >= 0.78:
        reasons.append("FinBERT probability entropy was elevated, indicating a mixed or unclear stance.")
    if lexicon_hits.get(dominant, 0) > 0:
        reasons.append(f"Detected {lexicon_hits[dominant]} matching finance-emotion phrase(s) for {dominant}.")
    matched_aspects = [a.get("aspect") for a in aspects if a.get("aspect") in _ASPECT_PRIORS]
    if matched_aspects:
        reasons.append(
            "Relevant aspect evidence was found in: " + ", ".join(dict.fromkeys(str(a) for a in matched_aspects)) + "."
        )

    return {
        "label": dominant,
        "score": normalized.get(dominant, top_score),
        "scores": normalized,
        "rationale": " ".join(reasons),
    }


def serialize_emotion_scores(emotion_scores: dict[str, float] | None) -> str | None:
    if not emotion_scores:
        return None
    return json.dumps(emotion_scores, ensure_ascii=False)
