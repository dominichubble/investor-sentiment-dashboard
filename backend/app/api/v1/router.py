"""Main API v1 router assembly."""

from fastapi import APIRouter

from . import data, sentiment

api_router = APIRouter()
api_router.include_router(data.router)
api_router.include_router(sentiment.router)

