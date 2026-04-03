"""
Sentiment routes for LEAN_API deployments (requirements-lean.txt).

Exposes ticker narrative (Groq + DB) and clear 503 stubs for FinBERT/LIME endpoints
so clients do not get opaque 404s.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.sentiment_narrative_service import generate_ticker_narrative

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

_ML_DISABLED_DETAIL = (
    "FinBERT and LIME are not installed in this LEAN_API build. "
    "Use the full backend image (requirements.txt, LEAN_API unset) for "
    "/sentiment/analyze, /batch, and /explain."
)


class TickerNarrativeResponse(BaseModel):
    """Grounded LLM summary of ticker sentiment over a date window (cached)."""

    narrative: str = ""
    cached: bool = False
    model: str = ""
    record_count: int = 0
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    period_key: str = ""
    data_signature: Optional[str] = None
    error: Optional[str] = None


@router.get("/_ping")
async def ping_sentiment() -> dict[str, str]:
    return {"status": "ok", "mode": "lean"}


@router.post("/analyze")
async def analyze_unavailable() -> None:
    raise HTTPException(status_code=503, detail=_ML_DISABLED_DETAIL)


@router.post("/batch")
async def batch_unavailable() -> None:
    raise HTTPException(status_code=503, detail=_ML_DISABLED_DETAIL)


@router.post("/explain")
async def explain_unavailable() -> None:
    raise HTTPException(status_code=503, detail=_ML_DISABLED_DETAIL)


@router.get(
    "/ticker-narrative/{ticker}",
    response_model=TickerNarrativeResponse,
    summary="AI narrative for ticker sentiment (grounded, cached)",
)
async def get_ticker_sentiment_narrative(
    ticker: str,
    period: str = Query(
        "90d",
        description="Same as correlation: 7d, 30d, 90d, 6mo, 1y — ignored if start_date+end_date set",
    ),
    start_date: Optional[date] = Query(
        None,
        description="Custom range start (inclusive); requires end_date",
    ),
    end_date: Optional[date] = Query(
        None,
        description="Custom range end (inclusive); requires start_date",
    ),
    force_refresh: bool = Query(
        False,
        description="Bypass cache and regenerate (still keyed on current data fingerprint)",
    ),
) -> TickerNarrativeResponse:
    if start_date is not None or end_date is not None:
        if start_date is None or end_date is None:
            raise HTTPException(
                status_code=400,
                detail="Both start_date and end_date are required for a custom range.",
            )
        if start_date > end_date:
            raise HTTPException(
                status_code=400,
                detail="start_date must be on or before end_date.",
            )

    result = await asyncio.to_thread(
        generate_ticker_narrative,
        ticker,
        period,
        start_date,
        end_date,
        force_refresh,
    )
    return TickerNarrativeResponse(
        narrative=result.get("narrative") or "",
        cached=bool(result.get("cached")),
        model=result.get("model") or "",
        record_count=int(result.get("record_count") or 0),
        window_start=result.get("window_start"),
        window_end=result.get("window_end"),
        period_key=result.get("period_key") or "",
        data_signature=result.get("data_signature"),
        error=result.get("error"),
    )
