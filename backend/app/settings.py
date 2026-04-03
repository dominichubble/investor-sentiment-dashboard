"""Runtime configuration from environment (production-friendly defaults)."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def load_dotenv_from_repo() -> None:
    """Load the nearest .env walking up from this package (same strategy as database module)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = here / ".env"
        if candidate.is_file():
            load_dotenv(candidate)
            return
        if here.parent == here:
            break
        here = here.parent


def cors_allow_origins() -> list[str]:
    raw = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
    if not raw:
        return list(_DEFAULT_CORS_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def api_key_set() -> frozenset[str]:
    """Bearer tokens accepted when API_KEYS is set. Empty set = auth disabled."""
    raw = (os.getenv("API_KEYS") or "").strip()
    if not raw:
        return frozenset()
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


def allowed_hosts() -> list[str] | None:
    """If set, enables TrustedHostMiddleware. Use comma-separated hostnames (no scheme)."""
    raw = (os.getenv("ALLOWED_HOSTS") or "").strip()
    if not raw:
        return None
    return [h.strip() for h in raw.split(",") if h.strip()]


def security_headers_enabled() -> bool:
    return (os.getenv("ENABLE_SECURITY_HEADERS") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
