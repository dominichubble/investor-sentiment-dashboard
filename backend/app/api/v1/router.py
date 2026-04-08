"""Main API v1 router assembly."""

from fastapi import APIRouter

from api.routers import correlation, stocks
from app.settings import lean_api_enabled

from . import data

api_router = APIRouter()
api_router.include_router(data.router)

if lean_api_enabled():
    from . import sentiment_lean

    api_router.include_router(sentiment_lean.router)
else:
    from . import sentiment

    api_router.include_router(sentiment.router)

api_router.include_router(correlation.router)
api_router.include_router(stocks.router)
