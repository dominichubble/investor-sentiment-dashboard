"""Production-oriented middleware: API keys, readiness, security headers."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def clear_api_keys(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)


@pytest.fixture
def client(clear_api_keys):
    # Import after env is clean so other modules are unaffected; app reads API_KEYS per request.
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_ready_returns_200_when_db_configured(client):
    """Readiness succeeds when DATABASE_URL is valid (same as rest of test suite)."""
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ready"
    assert body.get("database") == "ok"


def test_api_key_required_when_api_keys_env_set(monkeypatch):
    monkeypatch.setenv("API_KEYS", "unit-test-secret")
    from app.main import app

    with TestClient(app) as c:
        r = c.get("/api/v1/data/_ping")
        assert r.status_code == 401

        r2 = c.get(
            "/api/v1/data/_ping",
            headers={"Authorization": "Bearer wrong"},
        )
        assert r2.status_code == 403

        r3 = c.get(
            "/api/v1/data/_ping",
            headers={"Authorization": "Bearer unit-test-secret"},
        )
        assert r3.status_code == 200

        h = c.get("/health")
        assert h.status_code == 200

        opt = c.options("/api/v1/data/_ping")
        assert opt.status_code in (200, 405)


def test_security_headers_present_by_default(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
