"""
Shared helpers for generating stable sentiment record IDs.
"""

from __future__ import annotations

import hashlib
from typing import Optional


def make_record_id(prefix: str, *parts: Optional[str]) -> str:
    """
    Build a stable, short ID based on deterministic input parts.

    Args:
        prefix: ID prefix (e.g., "doc", "stock").
        *parts: Components that should uniquely identify the record.

    Returns:
        Stable ID string like "doc_ab12cd34ef56gh78".
    """
    normalized = ["" if p is None else str(p) for p in parts]
    base = "|".join(normalized)
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
