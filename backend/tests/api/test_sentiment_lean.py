"""Lean sentiment router: ML routes return 503; narrative path is mounted."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.sentiment_lean import router as lean_sentiment_router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(lean_sentiment_router, prefix="/api/v1")
    return TestClient(app)


def test_lean_analyze_returns_503():
    r = _client().post("/api/v1/sentiment/analyze", json={"text": "hello"})
    assert r.status_code == 503
    assert "FinBERT" in r.json().get("detail", "")


def test_lean_explain_returns_503():
    r = _client().post("/api/v1/sentiment/explain", json={"text": "hello"})
    assert r.status_code == 503


def test_lean_ping():
    r = _client().get("/api/v1/sentiment/_ping")
    assert r.status_code == 200
    assert r.json().get("mode") == "lean"
