"""Canonical FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.storage.database import get_engine


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize core services when the app boots."""
    get_engine()
    yield


app = FastAPI(
    title="Investor Sentiment Dashboard API",
    version="1.0.0",
    lifespan=lifespan,
)

# Dev frontend origins (Vite defaults and script-based port 3000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Minimal health endpoint required for bootstrap ticket."""
    return {"status": "ok"}
