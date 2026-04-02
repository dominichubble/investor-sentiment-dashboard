"""Tests for FastAPI bootstrap ticket."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    """Health endpoint returns the expected payload."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_v1_router_is_mounted():
    """A v1 stub route should be reachable."""
    response = client.get("/api/v1/data/_ping")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_correlation_routes_are_mounted_in_openapi():
    """Canonical app should expose correlation endpoints under /api/v1."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json().get("paths", {})
    assert "/api/v1/correlation/overview/all" in paths
    assert "/api/v1/sentiment/ticker-narrative/{ticker}" in paths
