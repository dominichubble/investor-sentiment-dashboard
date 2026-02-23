"""Sentiment router stubs for API v1."""

from fastapi import APIRouter

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.get("/_ping")
async def ping_sentiment() -> dict[str, str]:
    """Temporary v1 sentiment route proving router mount."""
    return {"status": "ok"}

