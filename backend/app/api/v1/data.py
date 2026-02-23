"""Data endpoints for API v1."""

from datetime import date, datetime, time
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.services.statistics_service import StatisticsService
from app.storage.sqlite_storage import SQLiteSentimentStorage

router = APIRouter(prefix="/data", tags=["data"])
storage = SQLiteSentimentStorage()
statistics_service = StatisticsService()


class SentimentInfo(BaseModel):
    """Sentiment payload returned for each record."""

    label: str
    score: float


class PredictionRecord(BaseModel):
    """Prediction record contract used by the frontend."""

    id: str
    record_type: str
    text: str
    sentiment: SentimentInfo
    source: str | None = None
    timestamp: str
    metadata: dict[str, Any] | None = None


class PredictionsResponse(BaseModel):
    """Paginated prediction records response."""

    predictions: list[PredictionRecord]
    total: int
    page: int
    page_size: int
    has_more: bool


class SentimentBreakdown(BaseModel):
    """Aggregate sentiment counts and percentages."""

    positive: int
    negative: int
    neutral: int
    positive_percentage: float
    negative_percentage: float
    neutral_percentage: float


class StockInfo(BaseModel):
    """Top-stock aggregate item."""

    ticker: str
    company_name: str
    count: int
    positive: int
    negative: int
    neutral: int


class StatisticsResponse(BaseModel):
    """Dashboard statistics payload."""

    total_predictions: int
    total_stocks_analyzed: int
    sentiment_distribution: SentimentBreakdown
    top_stocks: list[StockInfo]
    recent_activity: dict[str, int]
    date_range: dict[str, str | None]


@router.get("/_ping")
async def ping_data() -> dict[str, str]:
    """Temporary v1 data route proving router mount."""
    return {"status": "ok"}


@router.get("/predictions", response_model=PredictionsResponse)
async def get_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    source: str | None = Query(None),
    sentiment: str | None = Query(None),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
) -> PredictionsResponse:
    """Return paginated predictions from SQLite storage."""
    start_dt = datetime.combine(start_date, time.min) if start_date else None
    end_dt = datetime.combine(end_date, time.max) if end_date else None
    offset = (page - 1) * page_size

    rows, total = storage.query_records(
        record_types=None,
        source=(source.lower() if source else None),
        sentiment=(sentiment.lower() if sentiment else None),
        start_date=start_dt,
        end_date=end_dt,
        limit=page_size,
        offset=offset,
    )

    predictions: list[PredictionRecord] = []
    for row in rows:
        record_type = row.get("record_type", "document")
        metadata: dict[str, Any] = {}
        if record_type == "stock":
            metadata = {
                "ticker": row.get("ticker"),
                "mentioned_as": row.get("mentioned_as"),
                "document_id": row.get("document_id"),
            }
        elif row.get("document_id"):
            metadata = {"document_id": row.get("document_id")}

        predictions.append(
            PredictionRecord(
                id=row.get("id", ""),
                record_type=record_type,
                text=row.get("text", ""),
                sentiment=SentimentInfo(
                    label=row.get("sentiment_label", "neutral"),
                    score=float(row.get("sentiment_score", 0.0)),
                ),
                source=row.get("source"),
                timestamp=row.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                metadata=(metadata or None),
            )
        )

    return PredictionsResponse(
        predictions=predictions,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + page_size) < total,
    )


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics() -> StatisticsResponse:
    """Return aggregated dashboard statistics from SQLite."""
    stats = statistics_service.get_statistics()
    return StatisticsResponse(**stats)
