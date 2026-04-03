"""
Grounded ticker-level sentiment narratives via Groq (free tier, OpenAI-compatible API).

Uses only ingested DB rows in the same date window as correlation analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import date, datetime
from typing import Any, Optional

import httpx
from sqlalchemy import func

from app.services.sentiment_window import resolve_sentiment_window
from app.storage.database import (
    SentimentNarrativeCacheRow,
    SentimentRecordRow,
    get_session,
)

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = os.environ.get(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
)
DEFAULT_MODEL = os.environ.get(
    "SENTIMENT_NARRATIVE_MODEL", "llama-3.1-8b-instant"
)
MAX_EXCERPTS = 36
EXCERPT_MAX_CHARS = 260


def _period_key(
    period: str,
    start_date: Optional[date],
    end_date: Optional[date],
) -> str:
    if start_date is not None and end_date is not None:
        return f"c:{start_date.isoformat()}:{end_date.isoformat()}"
    return f"p:{period.strip()}"


def _data_signature(session, ticker: str, start_dt: datetime, end_dt: datetime) -> str:
    """Fingerprint rows in window so new ingests invalidate cache."""
    q = session.query(
        func.count(SentimentRecordRow.id),
        func.min(SentimentRecordRow.published_at),
        func.max(SentimentRecordRow.published_at),
        func.max(SentimentRecordRow.ingested_at),
    ).filter(
        SentimentRecordRow.ticker == ticker.upper(),
        SentimentRecordRow.published_at >= start_dt,
        SentimentRecordRow.published_at <= end_dt,
    )
    cnt, mn, mx, mx_ing = q.one()
    cnt = int(cnt or 0)
    mn_s = mn.isoformat() if mn else ""
    mx_s = mx.isoformat() if mx else ""
    mx_i = mx_ing.isoformat() if mx_ing else ""
    raw = f"{cnt}|{mn_s}|{mx_s}|{mx_i}"
    return hashlib.sha256(raw.encode()).hexdigest()[:48]


def _aggregate_by_label_and_source(
    session, ticker: str, start_dt: datetime, end_dt: datetime
) -> dict[str, Any]:
    rows = (
        session.query(
            SentimentRecordRow.sentiment_label,
            SentimentRecordRow.data_source,
            func.count(SentimentRecordRow.id),
        )
        .filter(
            SentimentRecordRow.ticker == ticker.upper(),
            SentimentRecordRow.published_at >= start_dt,
            SentimentRecordRow.published_at <= end_dt,
        )
        .group_by(SentimentRecordRow.sentiment_label, SentimentRecordRow.data_source)
        .all()
    )
    by_label: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
    by_source: dict[str, int] = {}
    total = 0
    for label, ds, c in rows:
        c = int(c)
        total += c
        lb = (label or "neutral").lower()
        if lb in by_label:
            by_label[lb] += c
        key = (ds or "unknown").lower()
        by_source[key] = by_source.get(key, 0) + c
    return {
        "total": total,
        "by_label": by_label,
        "by_source": by_source,
    }


def _sample_excerpts(
    session, ticker: str, start_dt: datetime, end_dt: datetime
) -> list[dict[str, str]]:
    """Random sample (Postgres) then stratify roughly by label."""
    pool = (
        session.query(SentimentRecordRow)
        .filter(
            SentimentRecordRow.ticker == ticker.upper(),
            SentimentRecordRow.published_at >= start_dt,
            SentimentRecordRow.published_at <= end_dt,
        )
        .order_by(func.random())
        .limit(120)
        .all()
    )
    if not pool:
        return []

    by_lbl: dict[str, list] = {"positive": [], "negative": [], "neutral": []}
    for r in pool:
        lbl = (r.sentiment_label or "neutral").lower()
        if lbl not in by_lbl:
            lbl = "neutral"
        by_lbl[lbl].append(r)

    out: list[dict[str, str]] = []
    per = max(1, MAX_EXCERPTS // 3)

    def add_row(r: SentimentRecordRow) -> None:
        if len(out) >= MAX_EXCERPTS:
            return
        t = (r.text or "").strip().replace("\n", " ")
        if len(t) > EXCERPT_MAX_CHARS:
            t = t[: EXCERPT_MAX_CHARS - 1] + "…"
        ds = (r.data_source or r.source or "?").lower()
        out.append(
            {
                "label": (r.sentiment_label or "neutral").lower(),
                "channel": ds,
                "excerpt": t,
            }
        )

    for lbl in ("positive", "negative", "neutral"):
        for r in by_lbl[lbl][:per]:
            add_row(r)
    # fill remainder from any
    for r in pool:
        if len(out) >= MAX_EXCERPTS:
            break
        add_row(r)
    # dedupe by excerpt text
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for item in out:
        key = item["excerpt"][:80]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:MAX_EXCERPTS]


def _build_user_prompt(
    ticker: str,
    window_start: datetime,
    window_end: datetime,
    agg: dict[str, Any],
    excerpts: list[dict[str, str]],
) -> str:
    total_n = int(agg["total"])
    bl = agg["by_label"]
    if total_n > 0:
        pos_pct = round(100.0 * bl["positive"] / total_n, 1)
        neu_pct = round(100.0 * bl["neutral"] / total_n, 1)
        neg_pct = round(100.0 * bl["negative"] / total_n, 1)
    else:
        pos_pct = neu_pct = neg_pct = 0.0
    lines = [
        f"Ticker: {ticker.upper()}",
        f"Date window (inclusive): {window_start.date().isoformat()} → {window_end.date().isoformat()}",
        "",
        "Aggregates from ingested posts only (each row is one mention in our database):",
        f"- Total mentions in window: {agg['total']}",
        f"- Raw counts — positive: {bl['positive']}, neutral: {bl['neutral']}, negative: {bl['negative']}",
        f"- Approximate shares — positive ~{pos_pct}%, neutral ~{neu_pct}%, negative ~{neg_pct}%",
        f"- By ingest channel (raw counts): {json.dumps(agg['by_source'], sort_keys=True)}",
        "",
        "Numbered excerpts below are a random sample from this window (text may be truncated). "
        "Infer themes and tone from them; paraphrase—do not paste long quotes.",
        "",
        "Excerpts:",
    ]
    for i, ex in enumerate(excerpts, 1):
        lines.append(
            f"{i}. [{ex['channel']}|{ex['label']}] {ex['excerpt']}"
        )
    return "\n".join(lines)


def _call_groq_chat(system: str, user: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/"
        )
    model = os.environ.get("SENTIMENT_NARRATIVE_MODEL", DEFAULT_MODEL).strip()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.28,
        "max_tokens": 1100,
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Groq returned no choices")
    content = (choices[0].get("message") or {}).get("content") or ""
    if not content.strip():
        raise RuntimeError("Groq returned empty content")
    return _strip_markdown_fences(content.strip())


def _strip_markdown_fences(text: str) -> str:
    """Remove accidental ``` / ```markdown wrappers from model output."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if not lines:
        return t
    first = lines[0].strip()
    if first.startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _upsert_cache(
    session,
    *,
    ticker: str,
    period_key: str,
    data_signature: str,
    narrative_text: str,
    model_id: str,
    record_count: int,
    window_start: datetime,
    window_end: datetime,
) -> None:
    existing = (
        session.query(SentimentNarrativeCacheRow)
        .filter_by(
            ticker=ticker,
            period_key=period_key,
            data_signature=data_signature,
        )
        .first()
    )
    now = datetime.utcnow()
    if existing:
        existing.narrative_text = narrative_text
        existing.model_id = model_id
        existing.record_count = record_count
        existing.window_start = window_start
        existing.window_end = window_end
        existing.created_at = now
    else:
        session.add(
            SentimentNarrativeCacheRow(
                id=str(uuid.uuid4()),
                ticker=ticker,
                period_key=period_key,
                data_signature=data_signature,
                narrative_text=narrative_text,
                model_id=model_id,
                record_count=record_count,
                window_start=window_start,
                window_end=window_end,
                created_at=now,
            )
        )


SYSTEM_PROMPT = """You are a careful data analyst explaining ingested social and news sentiment for ONE stock.

Non-negotiable rules:
- Use ONLY the aggregates and numbered excerpts in the user message. Do not invent prices, dates, earnings, or events. Do not use general knowledge about the company beyond what the excerpts support.
- If one channel (reddit / news / twitter) dominates the counts, say so and caution that the view may be skewed.
- If totals are small (<15 mentions) or labels are evenly split, say the signal is weak or mixed.
- UK English. Short, precise sentences. No buy/sell recommendations.

Output MUST be Markdown with exactly these sections and headings (## level 2 only for section titles). No code fences. No preamble like "Here is your analysis".

## Overview
2–4 sentences: overall sentiment tilt (which label leads, if any), rough mention volume, and whether tone looks mixed.

## What the numbers show
One short paragraph: interpret the positive/neutral/negative split and the channel breakdown using only the counts given.

## Themes in the excerpts
3–6 bullet points (- item). Each bullet names a recurring topic, tone, or concern visible in the excerpts. Paraphrase; do not copy sentences wholesale.

## Limitations
1–3 sentences: mention sampling (not every post is shown), truncation, possible bot/spam noise, and that this is not a complete market picture.

---
**Disclaimer:** This summary is machine-generated from sampled ingested text in our database only. It is not investment advice."""


def generate_ticker_narrative(
    ticker: str,
    period: str = "90d",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Return narrative dict with keys: narrative, cached, model, record_count,
    window_start, window_end, period_key, error (optional).
    """
    sym = ticker.strip().upper()
    if not sym:
        return {"error": "ticker is required"}

    pkey = _period_key(period, start_date, end_date)
    start_dt, end_dt = resolve_sentiment_window(sym, period, start_date, end_date)
    if start_dt is None or end_dt is None:
        return {
            "error": "Could not resolve a price history window for this ticker and period.",
            "narrative": "",
            "cached": False,
            "model": "",
            "record_count": 0,
            "period_key": pkey,
        }

    session = get_session()
    try:
        sig = _data_signature(session, sym, start_dt, end_dt)
        if not force_refresh:
            cached = (
                session.query(SentimentNarrativeCacheRow)
                .filter_by(ticker=sym, period_key=pkey, data_signature=sig)
                .first()
            )
            if cached:
                return {
                    "narrative": cached.narrative_text,
                    "cached": True,
                    "model": cached.model_id,
                    "record_count": cached.record_count,
                    "window_start": cached.window_start.isoformat() + "Z"
                    if cached.window_start
                    else None,
                    "window_end": cached.window_end.isoformat() + "Z"
                    if cached.window_end
                    else None,
                    "period_key": pkey,
                    "data_signature": sig,
                }

        agg = _aggregate_by_label_and_source(session, sym, start_dt, end_dt)
        if agg["total"] == 0:
            msg = (
                f"No ingested mentions of **{sym}** were found in this date window "
                f"({start_dt.date().isoformat()} → {end_dt.date().isoformat()}). "
                "Ingest Reddit, news, or X data that includes this ticker to enable a narrative."
            )
            _upsert_cache(
                session,
                ticker=sym,
                period_key=pkey,
                data_signature=sig,
                narrative_text=msg,
                model_id="none",
                record_count=0,
                window_start=start_dt,
                window_end=end_dt,
            )
            session.commit()
            return {
                "narrative": msg,
                "cached": False,
                "model": "none",
                "record_count": 0,
                "window_start": start_dt.isoformat() + "Z",
                "window_end": end_dt.isoformat() + "Z",
                "period_key": pkey,
                "data_signature": sig,
            }

        excerpts = _sample_excerpts(session, sym, start_dt, end_dt)
        user_prompt = _build_user_prompt(sym, start_dt, end_dt, agg, excerpts)
        try:
            narrative = _call_groq_chat(SYSTEM_PROMPT, user_prompt)
            model_used = os.environ.get(
                "SENTIMENT_NARRATIVE_MODEL", DEFAULT_MODEL
            ).strip()
        except Exception as e:
            logger.exception("Groq narrative failed: %s", e)
            session.rollback()
            return {
                "error": str(e),
                "narrative": "",
                "cached": False,
                "model": "",
                "record_count": agg["total"],
                "window_start": start_dt.isoformat() + "Z",
                "window_end": end_dt.isoformat() + "Z",
                "period_key": pkey,
                "data_signature": sig,
            }

        _upsert_cache(
            session,
            ticker=sym,
            period_key=pkey,
            data_signature=sig,
            narrative_text=narrative,
            model_id=model_used,
            record_count=agg["total"],
            window_start=start_dt,
            window_end=end_dt,
        )
        session.commit()
        return {
            "narrative": narrative,
            "cached": False,
            "model": model_used,
            "record_count": agg["total"],
            "window_start": start_dt.isoformat() + "Z",
            "window_end": end_dt.isoformat() + "Z",
            "period_key": pkey,
            "data_signature": sig,
        }
    finally:
        session.close()
