"""LEAN_API toggle."""

import pytest

from app.settings import lean_api_enabled


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("", False),
        ("false", False),
    ],
)
def test_lean_api_enabled(monkeypatch, raw, expected):
    monkeypatch.setenv("LEAN_API", raw)
    assert lean_api_enabled() is expected
