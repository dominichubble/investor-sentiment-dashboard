"""
Per-ticker sentiment data quality and confidence signals for the analysis window.

Uses the same date window as correlation and AI narrative (see sentiment_window).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional

from sqlalchemy import Date, cast, func

from app.services.sentiment_window import resolve_sentiment_window
from app.storage.database import SentimentRecordRow, get_session

_NORMALIZE_CHANNELS = {
    "reddit": "reddit",
    "news": "news",
    "twitter": "twitter",
    "x": "twitter",
}


def _norm_channel(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    k = raw.strip().lower()
    return _NORMALIZE_CHANNELS.get(k, "unknown")


def _calendar_days(start_d: date, end_d: date) -> int:
    return (end_d - start_d).days + 1


def _longest_gap_days(
    window_start: date, window_end: date, mention_dates: set[date]
) -> int:
    longest = 0
    streak = 0
    d = window_start
    while d <= window_end:
        if d in mention_dates:
            streak = 0
        else:
            streak += 1
            longest = max(longest, streak)
        d += timedelta(days=1)
    return longest


def _build_flags(
    *,
    total: int,
    neutral_pct: float,
    by_channel: dict[str, int],
    calendar_days: int,
    days_with_mentions: int,
    longest_gap: int,
) -> tuple[list[dict[str, str]], float, str]:
    """Return (flags, confidence_score 0..1, confidence_label)."""
    flags: list[dict[str, str]] = []
    score = 1.0

    if total == 0:
        flags.append(
            {
                "id": "no_mentions",
                "severity": "warning",
                "title": "No mentions in window",
                "detail": "There are no ingested sentiment rows for this ticker in the selected range.",
            }
        )
        return flags, 0.0, "none"

    if total < 5:
        flags.append(
            {
                "id": "very_thin_sample",
                "severity": "warning",
                "title": "Very small sample",
                "detail": f"Only {total} mention(s). Correlations and themes are unreliable.",
            }
        )
        score = min(score, 0.28)
    elif total < 15:
        flags.append(
            {
                "id": "thin_sample",
                "severity": "caution",
                "title": "Thin sample",
                "detail": f"{total} mentions is below a comfortable minimum for stable statistics.",
            }
        )
        score = min(score, 0.48)
    elif total < 40:
        flags.append(
            {
                "id": "moderate_sample",
                "severity": "info",
                "title": "Moderate sample size",
                "detail": f"{total} mentions — interpret edge cases cautiously.",
            }
        )
        score = min(score, 0.72)

    if neutral_pct >= 0.58:
        flags.append(
            {
                "id": "neutral_heavy",
                "severity": "info",
                "title": "Neutral-heavy labels",
                "detail": f"~{neutral_pct * 100:.0f}% of labels are neutral; FinBERT may be cautious or text is factual.",
            }
        )
        score *= 0.92

    # Channel skew (ignore unknown in denominator if others exist)
    ch_total = sum(by_channel.values())
    if ch_total > 0:
        for name, c in sorted(by_channel.items(), key=lambda x: -x[1]):
            share = c / ch_total
            if name != "unknown" and share >= 0.82 and ch_total >= 8:
                flags.append(
                    {
                        "id": "channel_skew",
                        "severity": "caution",
                        "title": f"Mostly {name}",
                        "detail": f"~{share * 100:.0f}% of mentions come from {name}; view may not represent all channels.",
                    }
                )
                score *= 0.86
                break

    if calendar_days >= 14:
        cov = days_with_mentions / calendar_days
        if cov < 0.25 and total >= 5:
            flags.append(
                {
                    "id": "sparse_calendar_coverage",
                    "severity": "caution",
                    "title": "Sparse day coverage",
                    "detail": f"Mentions appear on only {days_with_mentions} of {calendar_days} calendar days (~{cov * 100:.0f}%).",
                }
            )
            score *= 0.82

    if longest_gap >= 14 and calendar_days >= 21:
        flags.append(
            {
                "id": "long_quiet_period",
                "severity": "info",
                "title": "Long gap without mentions",
                "detail": f"Up to {longest_gap} consecutive calendar days had no mentions in this window.",
            }
        )
        score *= 0.9

    score = max(0.0, min(1.0, round(score, 2)))
    if score >= 0.72:
        label = "high"
    elif score >= 0.42:
        label = "medium"
    else:
        label = "low"

    return flags, score, label


def get_stock_data_quality(
    ticker: str,
    period: str = "90d",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict[str, Any]:
    sym = ticker.strip().upper()
    if not sym:
        return {"error": "ticker is required", "ticker": ""}

    start_dt, end_dt = resolve_sentiment_window(sym, period, start_date, end_date)
    if start_dt is None or end_dt is None:
        return {
            "error": "Could not resolve analysis window (check ticker and period).",
            "ticker": sym,
        }

    session = get_session()
    try:
        base_filter = (
            session.query(SentimentRecordRow)
            .filter(
                SentimentRecordRow.ticker == sym,
                SentimentRecordRow.published_at >= start_dt,
                SentimentRecordRow.published_at <= end_dt,
            )
        )

        total = base_filter.count()

        label_rows = (
            session.query(
                SentimentRecordRow.sentiment_label,
                func.count(SentimentRecordRow.id),
            )
            .filter(
                SentimentRecordRow.ticker == sym,
                SentimentRecordRow.published_at >= start_dt,
                SentimentRecordRow.published_at <= end_dt,
            )
            .group_by(SentimentRecordRow.sentiment_label)
            .all()
        )
        by_label = {"positive": 0, "negative": 0, "neutral": 0}
        for lbl, c in label_rows:
            key = (lbl or "neutral").lower()
            if key in by_label:
                by_label[key] = int(c)

        src_rows = (
            session.query(
                SentimentRecordRow.data_source,
                func.count(SentimentRecordRow.id),
            )
            .filter(
                SentimentRecordRow.ticker == sym,
                SentimentRecordRow.published_at >= start_dt,
                SentimentRecordRow.published_at <= end_dt,
            )
            .group_by(SentimentRecordRow.data_source)
            .all()
        )
        by_channel: dict[str, int] = {"reddit": 0, "news": 0, "twitter": 0, "unknown": 0}
        for ds, c in src_rows:
            bucket = _norm_channel(ds)
            if bucket not in by_channel:
                by_channel[bucket] = 0
            by_channel[bucket] += int(c)

        day_col = cast(SentimentRecordRow.published_at, Date)
        daily = (
            session.query(day_col, func.count(SentimentRecordRow.id))
            .filter(
                SentimentRecordRow.ticker == sym,
                SentimentRecordRow.published_at >= start_dt,
                SentimentRecordRow.published_at <= end_dt,
            )
            .group_by(day_col)
            .all()
        )
        mention_dates: set[date] = set()
        for d, _cnt in daily:
            if d is None:
                continue
            if isinstance(d, datetime):
                mention_dates.add(d.date())
            else:
                mention_dates.add(d)

        w_start = start_dt.date()
        w_end = end_dt.date()
        calendar_days = _calendar_days(w_start, w_end)
        days_with_mentions = len(mention_dates)
        longest_gap = _longest_gap_days(w_start, w_end, mention_dates)

        t = max(total, 1)
        neutral_pct = by_label["neutral"] / t
        pos_pct = by_label["positive"] / t
        neg_pct = by_label["negative"] / t

        flags, confidence_score, confidence_label = _build_flags(
            total=total,
            neutral_pct=neutral_pct,
            by_channel=by_channel,
            calendar_days=calendar_days,
            days_with_mentions=days_with_mentions,
            longest_gap=longest_gap,
        )

        return {
            "ticker": sym,
            "window_start": start_dt.isoformat() + "Z",
            "window_end": end_dt.isoformat() + "Z",
            "calendar_days": calendar_days,
            "days_with_mentions": days_with_mentions,
            "calendar_coverage": round(days_with_mentions / calendar_days, 3)
            if calendar_days
            else 0.0,
            "longest_gap_days": longest_gap,
            "total_mentions": total,
            "by_label": by_label,
            "label_shares": {
                "positive": round(pos_pct, 4),
                "neutral": round(neutral_pct, 4),
                "negative": round(neg_pct, 4),
            },
            "by_channel": {k: v for k, v in by_channel.items() if v > 0},
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
            "flags": flags,
        }
    finally:
        session.close()
