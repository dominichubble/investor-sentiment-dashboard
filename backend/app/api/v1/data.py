"""Data router stubs for API v1."""

from fastapi import APIRouter

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/_ping")
async def ping_data() -> dict[str, str]:
    """Temporary v1 data route proving router mount."""
    return {"status": "ok"}

