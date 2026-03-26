"""Shared prediction metadata for API responses (FinBERT distribution, aspects, etc.)."""

from __future__ import annotations

import json
from typing import Any, Dict


def build_prediction_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    record_type = record.get("record_type", "document")
    metadata: Dict[str, Any] = {"record_type": record_type}
    if record.get("ticker"):
        metadata["ticker"] = record["ticker"]
        metadata["mentioned_as"] = record.get("mentioned_as")
    if record.get("data_source"):
        metadata["data_source"] = record["data_source"]
    sm = record.get("source_meta_json")
    if sm:
        try:
            metadata["source_meta"] = json.loads(sm)
        except (TypeError, json.JSONDecodeError):
            metadata["source_meta"] = None
    if record.get("score_positive") is not None:
        metadata["finbert_distribution"] = {
            "positive": float(record["score_positive"]),
            "negative": float(record["score_negative"] or 0.0),
            "neutral": float(record["score_neutral"] or 0.0),
        }
    if record.get("sentiment_uncertainty") is not None:
        metadata["uncertainty"] = float(record["sentiment_uncertainty"])
    rationale = record.get("rationale")
    if rationale:
        metadata["rationale"] = rationale
    aj = record.get("aspects_json")
    if aj:
        try:
            metadata["aspects"] = json.loads(aj)
        except (TypeError, json.JSONDecodeError):
            metadata["aspects"] = None
    return metadata
