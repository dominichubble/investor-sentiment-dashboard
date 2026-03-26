"""Document-level financial sentiment metadata beyond a single label.

FinBERT still produces the headline label/score; this module adds:
- Normalised entropy (mixed-signal / uncertainty)
- A short rationale derived from the softmax (no LLM)
- Aspect-tagged evidence sentences + optional per-snippet FinBERT scores

Limitations (intentional honesty for FYP / dashboards):
- FinBERT is trained on financial phrase sentiment, not full discourse reasoning;
  ambiguous headlines remain hard. Uncertainty scores flag low-confidence cases.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

LABEL_ORDER = ("positive", "negative", "neutral")

# Aspect buckets: keyword hit → attach sentence as evidence (ABSA-lite).
_ASPECT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "revenue_earnings": (
        "revenue",
        "earnings",
        "eps",
        "guidance",
        "margin",
        "profit",
        "loss",
        "ebitda",
        "quarter",
        "q1",
        "q2",
        "q3",
        "q4",
        "beat estimates",
        "miss estimates",
    ),
    "growth_demand": (
        "growth",
        "demand",
        "capacity",
        "subscriber",
        "users",
        "adoption",
        "expansion",
        "scaling",
    ),
    "risk_liquidity": (
        "bankruptcy",
        "debt",
        "downgrade",
        "lawsuit",
        "sec",
        "investigation",
        "liquidity",
        "default",
        "writedown",
        "impairment",
    ),
    "valuation": (
        "valuation",
        "priced",
        "multiple",
        "pe ratio",
        "p/e",
        "upside",
        "downside",
        "overvalued",
        "undervalued",
        "target price",
    ),
    "product_ai": (
        "ai",
        "gpu",
        "chip",
        "product",
        "launch",
        "innovation",
        "patent",
    ),
    "macro_policy": (
        "fed",
        "rates",
        "inflation",
        "tariff",
        "recession",
        "stimulus",
        "policy",
    ),
}


def _split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    chunks = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    out: List[str] = []
    for c in chunks:
        c = c.strip()
        if len(c) < 12:
            continue
        out.append(c[:2000])
    if not out and text.strip():
        return [text.strip()[:2000]]
    return out


def extract_aspect_snippets(
    text: str,
    max_snippets_total: int = 8,
    max_per_aspect: int = 2,
    max_snippet_len: int = 320,
) -> List[Dict[str, str]]:
    """Return evidence snippets tagged by coarse aspect (keyword buckets)."""
    sentences = _split_sentences(text)
    seen_aspect: Dict[str, int] = {}
    results: List[Dict[str, str]] = []

    for sent in sentences:
        if len(results) >= max_snippets_total:
            break
        low = sent.lower()
        for aspect, keywords in _ASPECT_KEYWORDS.items():
            if seen_aspect.get(aspect, 0) >= max_per_aspect:
                continue
            hit = False
            for kw in keywords:
                if " " in kw:
                    if kw in low:
                        hit = True
                        break
                elif re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", low):
                    hit = True
                    break
            if not hit:
                continue
            snippet = sent if len(sent) <= max_snippet_len else sent[: max_snippet_len - 1] + "…"
            results.append({"aspect": aspect, "snippet": snippet})
            seen_aspect[aspect] = seen_aspect.get(aspect, 0) + 1
            if len(results) >= max_snippets_total:
                break

    return results


def normalized_label_entropy(scores: Dict[str, float]) -> float:
    """0 ≈ one-hot (decisive), 1 ≈ uniform (maximally mixed) for 3 classes."""
    probs = [max(float(scores.get(l, 0.0)), 1e-12) for l in LABEL_ORDER]
    s = sum(probs)
    probs = [p / s for p in probs]
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(3)


def build_rationale(
    label: str,
    top_score: float,
    scores: Dict[str, float],
    entropy: float,
) -> str:
    """Human-readable summary from FinBERT softmax (debugging / UI), not chain-of-thought."""
    p = float(scores.get("positive", 0.0))
    n = float(scores.get("negative", 0.0))
    neu = float(scores.get("neutral", 0.0))
    parts: List[str] = []

    parts.append(
        f"FinBERT: dominant {label} ({top_score:.0%}); "
        f"distribution pos {p:.0%} / neg {n:.0%} / neutral {neu:.0%}."
    )

    if entropy >= 0.88:
        parts.append("High entropy — competing labels; treat headline sentiment as uncertain.")
    elif entropy >= 0.75:
        parts.append("Moderate mixed signals between labels.")

    ordered = sorted(
        [("positive", p), ("negative", n), ("neutral", neu)],
        key=lambda x: x[1],
        reverse=True,
    )
    runner_label, runner_p = ordered[1]
    if runner_p >= 0.22 and top_score < 0.65:
        parts.append(
            f"Substantial mass on {runner_label} ({runner_p:.0%}) alongside top label — "
            "text may mix bullish and bearish cues (e.g. demand vs capacity)."
        )

    if neu >= 0.35 and label != "neutral":
        parts.append(
            "Large neutral probability often reflects factual/reporting tone rather than clear bull/bear stance."
        )

    return " ".join(parts)


def enrich_aspects_with_scores(
    aspects: List[Dict[str, str]],
    sentiment_by_snippet: Dict[str, Dict[str, Any]],
) -> str:
    """JSON string for DB: list of {aspect, snippet, label, score, scores?}."""
    if not aspects:
        return "[]"
    enriched: List[Dict[str, Any]] = []
    for item in aspects:
        snip = item["snippet"]
        s = sentiment_by_snippet.get(snip)
        if not s:
            enriched.append({**item, "label": None, "score": None})
            continue
        row: Dict[str, Any] = {
            "aspect": item["aspect"],
            "snippet": snip,
            "label": s.get("label"),
            "score": float(s.get("score", 0.0)),
        }
        if isinstance(s.get("scores"), dict):
            row["scores"] = s["scores"]
        enriched.append(row)
    return json.dumps(enriched, ensure_ascii=False)


def collect_unique_snippets(aspect_rows: List[List[Dict[str, str]]]) -> List[str]:
    """Preserve order, dedupe by snippet text."""
    seen = set()
    out: List[str] = []
    for row_aspects in aspect_rows:
        for a in row_aspects:
            sn = a["snippet"]
            if sn not in seen:
                seen.add(sn)
                out.append(sn)
    return out
