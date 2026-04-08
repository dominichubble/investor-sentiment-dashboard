#!/usr/bin/env python3
"""Backfill finance emotion labels for existing stock sentiment rows."""

from __future__ import annotations

import json
from typing import Any

from app.analysis.finance_emotion import infer_finance_emotion, serialize_emotion_scores
from app.storage.database import SentimentRecordRow, get_session


def main(limit: int | None = None) -> None:
    session = get_session()
    try:
        query = (
            session.query(SentimentRecordRow)
            .filter(SentimentRecordRow.ticker.isnot(None))
            .filter(SentimentRecordRow.emotion_label.is_(None))
            .order_by(SentimentRecordRow.published_at.asc())
        )
        if limit is not None:
            query = query.limit(limit)

        rows = query.all()
        for row in rows:
            aspect_payload: list[dict[str, Any]] = []
            if row.aspects_json:
                try:
                    aspect_payload = json.loads(row.aspects_json)
                except (TypeError, json.JSONDecodeError):
                    aspect_payload = []

            scores = None
            if (
                row.score_positive is not None
                or row.score_negative is not None
                or row.score_neutral is not None
            ):
                scores = {
                    "positive": float(row.score_positive or 0.0),
                    "negative": float(row.score_negative or 0.0),
                    "neutral": float(row.score_neutral or 0.0),
                }

            emotion = infer_finance_emotion(
                text=row.text or "",
                sentiment_label=row.sentiment_label or "neutral",
                sentiment_score=float(row.sentiment_score or 0.5),
                scores=scores,
                uncertainty=float(row.sentiment_uncertainty or 0.0),
                aspects=aspect_payload,
            )
            row.emotion_label = emotion["label"]
            row.emotion_scores_json = serialize_emotion_scores(emotion.get("scores"))
            row.emotion_rationale = emotion.get("rationale")

        session.commit()
        print(f"Backfilled finance emotions for {len(rows)} row(s).")
    finally:
        session.close()


if __name__ == "__main__":
    main()
