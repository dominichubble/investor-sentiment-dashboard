"""
Resolve the sentiment query window to match correlation / narrative analysis.

Preset periods use yfinance price history bounds; custom ranges use inclusive calendar days.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Optional, Tuple

import pandas as pd

from app.analysis.price_service import PriceService


def resolve_sentiment_window(
    ticker: str,
    period: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Return (start_dt, end_dt) UTC-naive inclusive bounds for sentiment DB queries.

    When start_date and end_date are both set, uses calendar-day bounds.
    Otherwise uses first/last trading day from yfinance for the given period.
    """
    sym = ticker.strip().upper()
    if not sym:
        return None, None

    if start_date is not None and end_date is not None:
        start_dt = datetime.combine(start_date, time.min)
        end_dt = datetime.combine(end_date, time.max.replace(microsecond=0))
        return start_dt, end_dt

    df = PriceService.get_price_history(sym, period=period.strip() or "90d")
    if df is None or df.empty:
        return None, None
    d_min = pd.Timestamp(df["date"].min()).to_pydatetime()
    d_max = pd.Timestamp(df["date"].max()).to_pydatetime()
    start_dt = d_min.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = d_max.replace(hour=23, minute=59, second=59, microsecond=0)
    return start_dt, end_dt
