"""Canonical FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.v1.router import api_router
from app.middleware.http import APIKeyMiddleware, SecurityHeadersMiddleware
from app.settings import allowed_hosts, cors_allow_origins, load_dotenv_from_repo
from app.storage.database import get_engine

load_dotenv_from_repo()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Do not connect to the DB here — keeps the process alive for /health if DB is slow or misconfigured."""
    yield


app = FastAPI(
    title="Investor Sentiment Dashboard API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware: last registered is outermost on the request. CORS should run early;
# TrustedHost (if any) should be the outermost check.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_hosts = allowed_hosts()
if _hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_hosts)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    """Human-friendly root when someone opens the API host in a browser (avoids bare 404)."""
    return {
        "name": "Investor Sentiment Dashboard API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "health_ready": "/health/ready",
        "api_v1": "/api/v1",
    }


@app.head("/")
async def root_head() -> Response:
    """Some load balancers probe with HEAD; return 200 with an empty body."""
    return Response(status_code=200)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness: process is up (no dependency checks)."""
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    """Readiness: verify database connectivity (for load balancers / orchestrators)."""
    from fastapi.responses import JSONResponse
    from sqlalchemy import text

    try:
        engine = get_engine()
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "database": "unconfigured"},
        )

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "database": "unavailable"},
        )

    return {"status": "ready", "database": "ok"}
