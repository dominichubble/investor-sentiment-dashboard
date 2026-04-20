"""Synthetic sentiment rows for local / examiner runs without PostgreSQL."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.storage.database import SentimentRecordRow

logger = logging.getLogger(__name__)


def _demo_rows(anchor: datetime) -> list[dict[str, Any]]:
    """Return a small, hand-authored panel (not real market data)."""
    rows: list[dict[str, Any]] = []
    templates = [
        # ticker, label, pos, neg, neu, emotion, text snippet, source, data_source, day_offset
        (
            "AAPL",
            "positive",
            0.88,
            0.06,
            0.06,
            "optimism",
            "Apple reported record quarterly revenue, beating analyst expectations.",
            "r/stocks",
            "reddit",
            2,
        ),
        (
            "AAPL",
            "neutral",
            0.35,
            0.30,
            0.35,
            "uncertainty",
            "Apple scheduled its annual shareholder meeting with no major surprises announced.",
            "r/investing",
            "reddit",
            5,
        ),
        (
            "AAPL",
            "negative",
            0.12,
            0.78,
            0.10,
            "fear",
            "Supply chain warnings weighed on Apple sentiment after supplier guidance cut.",
            "Bloomberg",
            "news",
            8,
        ),
        (
            "MSFT",
            "positive",
            0.82,
            0.09,
            0.09,
            "confidence",
            "Microsoft cloud growth continues to outpace legacy software declines.",
            "r/wallstreetbets",
            "reddit",
            3,
        ),
        (
            "MSFT",
            "neutral",
            0.40,
            0.28,
            0.32,
            "mixed",
            "Microsoft closed flat as traders balanced AI upside against valuation concerns.",
            "Reuters",
            "news",
            6,
        ),
        (
            "NVDA",
            "positive",
            0.91,
            0.04,
            0.05,
            "optimism",
            "NVIDIA data centre demand remains exceptionally strong this quarter.",
            "r/stocks",
            "reddit",
            1,
        ),
        (
            "NVDA",
            "negative",
            0.15,
            0.80,
            0.05,
            "scepticism",
            "Some desks flagged overheated positioning in NVIDIA ahead of options expiry.",
            "Financial Times",
            "news",
            9,
        ),
        (
            "SPY",
            "neutral",
            0.33,
            0.34,
            0.33,
            "mixed",
            "The S&P 500 index finished the session little changed on light macro news.",
            "CNBC",
            "news",
            4,
        ),
        (
            "TSLA",
            "negative",
            0.20,
            0.72,
            0.08,
            "fear",
            "Tesla delivery numbers missed whisper expectations; shares traded lower in sympathy.",
            "Twitter",
            "twitter",
            7,
        ),
        (
            "TSLA",
            "positive",
            0.79,
            0.12,
            0.09,
            "confidence",
            "Tesla energy storage deployments beat internal targets for the region.",
            "Twitter",
            "twitter",
            11,
        ),
        (
            "AMZN",
            "positive",
            0.84,
            0.08,
            0.08,
            "optimism",
            "Amazon Web Services margin expansion supported a constructive read on cash flow.",
            "r/stocks",
            "reddit",
            10,
        ),
        (
            "AMZN",
            "neutral",
            0.38,
            0.31,
            0.31,
            "uncertainty",
            "Amazon reiterated prior capital expenditure guidance without raising the range.",
            "WSJ",
            "news",
            12,
        ),
    ]
    idx = 0
    for cycle in range(0, 4):
        for (
            ticker,
            label,
            sp,
            sn,
            su,
            emotion,
            text,
            src,
            ds,
            day_off,
        ) in templates:
            idx += 1
            published = anchor - timedelta(days=day_off + cycle * 13)
            score = sp if label == "positive" else sn if label == "negative" else su
            rows.append(
                {
                    "id": f"demo-seed-{idx:03d}",
                    "text": text,
                    "ticker": ticker,
                    "mentioned_as": ticker,
                    "sentiment_label": label,
                    "sentiment_score": float(score),
                    "score_positive": sp,
                    "score_negative": sn,
                    "score_neutral": su,
                    "sentiment_uncertainty": 0.25 + (idx % 5) * 0.03,
                    "rationale": "Synthetic demo record for offline evaluation.",
                    "aspects_json": json.dumps(
                        [{"aspect": "revenue", "polarity": label, "snippet": text[:60]}]
                    ),
                    "emotion_label": emotion,
                    "emotion_scores_json": json.dumps({emotion: 0.62, "mixed": 0.12}),
                    "emotion_rationale": "Demo emotion layer derived from softmax and lexicon priors.",
                    "source": src,
                    "data_source": ds,
                    "source_id": f"demo-src-{idx}",
                    "source_meta_json": None,
                    "published_at": published,
                }
            )
    # A few document-level rows (no ticker) so the predictions feed is not empty-only stocks.
    for j in range(1, 7):
        rows.append(
            {
                "id": f"demo-doc-{j:03d}",
                "text": f"Synthetic macro headline {j}: central bank communication dominated the tape.",
                "ticker": None,
                "mentioned_as": "",
                "sentiment_label": "neutral",
                "sentiment_score": 0.5,
                "score_positive": 0.34,
                "score_negative": 0.33,
                "score_neutral": 0.33,
                "sentiment_uncertainty": 0.4,
                "rationale": None,
                "aspects_json": None,
                "emotion_label": "uncertainty",
                "emotion_scores_json": None,
                "emotion_rationale": None,
                "source": "DemoWire",
                "data_source": "news",
                "source_id": f"demo-doc-src-{j}",
                "source_meta_json": None,
                "published_at": anchor - timedelta(days=j + 20),
            }
        )
    return rows


def ensure_demo_dataset(engine: Engine) -> None:
    """Insert synthetic rows once if the stock-linked table is empty."""
    if engine.dialect.name != "sqlite":
        return

    SessionLocal = sessionmaker(bind=engine)
    session: Session = SessionLocal()
    try:
        existing = (
            session.query(func.count(SentimentRecordRow.id))
            .filter(SentimentRecordRow.ticker.isnot(None))
            .scalar()
            or 0
        )
        if existing > 0:
            return

        anchor = datetime.utcnow()
        payload = _demo_rows(anchor)
        for spec in payload:
            session.add(SentimentRecordRow(**spec))
        session.commit()
        logger.info(
            "Loaded %s synthetic sentiment records (local demo database).",
            len(payload),
        )
    except Exception:
        session.rollback()
        logger.exception("Failed to seed local demo database")
        raise
    finally:
        session.close()
