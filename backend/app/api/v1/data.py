"""Data endpoints for API v1."""

from datetime import date, datetime, time
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.prediction_metadata import build_prediction_metadata
from app.services.statistics_service import StatisticsService
from app.storage.sqlite_storage import SentimentStorage

router = APIRouter(prefix="/data", tags=["data"])
storage = SentimentStorage()
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
    published_at: str
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


class DailyTrendPoint(BaseModel):
    """One day in the market-wide sentiment series."""

    date: str
    count: int
    net_sentiment: float


class SourceSentimentBlock(BaseModel):
    total: int = 0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    positive_percentage: float = 0.0
    negative_percentage: float = 0.0
    neutral_percentage: float = 0.0


class SourceComparisonResponse(BaseModel):
    reddit: SourceSentimentBlock = Field(default_factory=SourceSentimentBlock)
    news: SourceSentimentBlock = Field(default_factory=SourceSentimentBlock)
    twitter: SourceSentimentBlock = Field(default_factory=SourceSentimentBlock)


class StatisticsResponse(BaseModel):
    """Dashboard statistics payload."""

    total_predictions: int
    total_stocks_analyzed: int
    sentiment_distribution: SentimentBreakdown
    top_stocks: list[StockInfo]
    recent_activity: dict[str, int]
    date_range: dict[str, str | None]
    daily_trend: list[DailyTrendPoint] = Field(default_factory=list)
    source_comparison: SourceComparisonResponse | None = None


@router.get("/_ping")
async def ping_data() -> dict[str, str]:
    """Temporary v1 data route proving router mount."""
    return {"status": "ok"}


@router.get("/predictions", response_model=PredictionsResponse)
async def get_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    source: str | None = Query(None),
    data_source: str | None = Query(
        None,
        description="Ingest channel: reddit, news, or twitter",
    ),
    sentiment: str | None = Query(None),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
) -> PredictionsResponse:
    """Return paginated predictions from database storage."""
    start_dt = datetime.combine(start_date, time.min) if start_date else None
    end_dt = datetime.combine(end_date, time.max) if end_date else None
    offset = (page - 1) * page_size

    rows, total = storage.query_records(
        source=(source.lower() if source else None),
        data_source=(data_source.lower() if data_source else None),
        sentiment=(sentiment.lower() if sentiment else None),
        start_date=start_dt,
        end_date=end_dt,
        limit=page_size,
        offset=offset,
    )

    predictions: list[PredictionRecord] = []
    for row in rows:
        record_type = row.get("record_type", "document")
        metadata = build_prediction_metadata(row)
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
                published_at=row.get("published_at", datetime.utcnow().isoformat() + "Z"),
                metadata=metadata,
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
async def get_statistics(
    days: int | None = Query(
        None,
        description="Limit to the last N days relative to the newest record. Omit for all data.",
        ge=1,
    ),
    data_source: str | None = Query(
        None,
        description="Ingest channel: reddit, news, twitter (omit for all)",
    ),
) -> StatisticsResponse:
    """Return aggregated dashboard statistics from the database."""
    raw = statistics_service.get_statistics(
        days=days,
        data_source=data_source,
        include_source_comparison=data_source is None,
    )
    sc = raw.pop("source_comparison", None)
    source_model = (
        SourceComparisonResponse(**sc) if isinstance(sc, dict) else None
    )
    return StatisticsResponse(**raw, source_comparison=source_model)
