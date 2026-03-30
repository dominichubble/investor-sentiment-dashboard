"""
Data retrieval API endpoints.

Provides access to historical predictions, stock sentiments, and statistics.
Includes TTL caching for the statistics endpoint.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional

from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.prediction_metadata import build_prediction_metadata
from app.entities.stock_database import StockDatabase
from app.services.statistics_service import StatisticsService
from app.storage import StockSentimentStorage

router = APIRouter(prefix="/data", tags=["data"])

# Initialize storage
storage = StockSentimentStorage()
storage.load()

# Initialize stock database for name lookup
stock_db = StockDatabase()
stock_db.load()

# Statistics cache (5 minute TTL)
_stats_cache: TTLCache = TTLCache(maxsize=64, ttl=300)
statistics_service = StatisticsService()


# Request/Response Models
class SentimentInfo(BaseModel):
    """Sentiment information."""

    label: str = Field(..., description="Sentiment label")
    score: float = Field(..., description="Confidence score (0-1)")


class PredictionRecord(BaseModel):
    """Historical prediction record."""

    id: str = Field(..., description="Prediction ID")
    record_type: str = Field(..., description="Record type (document or stock)")
    text: str = Field(..., description="Analyzed text")
    sentiment: SentimentInfo = Field(..., description="Sentiment prediction")
    source: Optional[str] = Field(None, description="Data source")
    timestamp: str = Field(..., description="Prediction timestamp (ISO 8601)")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class PredictionsResponse(BaseModel):
    """Response model for predictions list."""

    predictions: List[PredictionRecord] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_more: bool = Field(..., description="Whether more results are available")


class StockSentimentRecord(BaseModel):
    """Stock sentiment record."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Company name")
    sentiment: SentimentInfo = Field(..., description="Sentiment for this stock")
    text: str = Field("", description="Analyzed text")
    timestamp: str = Field(..., description="Analysis timestamp")


class StockSentimentsResponse(BaseModel):
    """Response model for stock sentiments."""

    ticker: str = Field(..., description="Stock ticker symbol")
    sentiments: List[StockSentimentRecord] = Field(..., description="Sentiment records")
    total: int = Field(..., description="Total number of records")
    summary: dict = Field(..., description="Sentiment summary statistics")


class SourceSentimentBlock(BaseModel):
    """Per-channel sentiment counts and percentages."""

    total: int = 0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    positive_percentage: float = 0.0
    negative_percentage: float = 0.0
    neutral_percentage: float = 0.0


class SourceComparisonResponse(BaseModel):
    """Sentiment breakdown by ingest channel (when viewing all sources)."""

    reddit: SourceSentimentBlock = Field(default_factory=SourceSentimentBlock)
    news: SourceSentimentBlock = Field(default_factory=SourceSentimentBlock)
    twitter: SourceSentimentBlock = Field(default_factory=SourceSentimentBlock)


class DailyTrendPoint(BaseModel):
    """One day in the market-wide sentiment series."""

    date: str
    count: int
    net_sentiment: float


class StatisticsResponse(BaseModel):
    """Response model for aggregate statistics."""

    total_predictions: int = Field(..., description="Total predictions made")
    total_stocks_analyzed: int = Field(..., description="Total unique stocks analyzed")
    sentiment_distribution: dict = Field(..., description="Overall sentiment distribution")
    top_stocks: List[dict] = Field(..., description="Most frequently mentioned stocks")
    recent_activity: dict = Field(..., description="Recent activity statistics")
    date_range: dict = Field(..., description="Data date range")
    daily_trend: List[DailyTrendPoint] = Field(
        default_factory=list,
        description="Daily mention volume and net sentiment for charts",
    )
    source_comparison: Optional[SourceComparisonResponse] = Field(
        None,
        description="Per-source sentiment when data_source filter is not applied",
    )


# Endpoints


@router.get("/predictions")
async def get_predictions(
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(20, description="Items per page", ge=1, le=100),
    source: Optional[str] = Query(
        None,
        description="Filter by stored source field (e.g. subreddit name or news outlet id)",
    ),
    data_source: Optional[str] = Query(
        None,
        description="Filter by ingest platform: reddit, news, or twitter",
    ),
    sentiment: Optional[str] = Query(
        None, description="Filter by sentiment (positive/negative/neutral)"
    ),
    start_date: Optional[date] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="Filter to date (YYYY-MM-DD)"),
) -> PredictionsResponse:
    """
    Get historical sentiment predictions with filtering and pagination.

    **Use Cases:**
    - Retrieving historical analysis results
    - Analyzing sentiment trends over time
    - Filtering by source, sentiment, or date range
    - Paginating through large result sets

    **Query Parameters:**
    - `page`: Page number (default: 1)
    - `page_size`: Items per page (default: 20, max: 100)
    - `source`: Filter by data source
    - `sentiment`: Filter by sentiment label
    - `start_date`, `end_date`: Date range filter

    **Example:**
    ```
    GET /api/v1/data/predictions?page=1&page_size=20&sentiment=positive
    ```
    """
    try:
        start_dt = (
            datetime.combine(start_date, datetime.min.time()) if start_date else None
        )
        end_dt = datetime.combine(end_date, datetime.max.time()) if end_date else None

        start_idx = (page - 1) * page_size

        sentiment_filter = sentiment.lower() if sentiment else None
        source_filter = source.lower() if source else None
        data_source_filter = data_source.lower() if data_source else None

        records, total = storage.query_records(
            source=source_filter,
            data_source=data_source_filter,
            sentiment=sentiment_filter,
            start_date=start_dt,
            end_date=end_dt,
            limit=page_size,
            offset=start_idx,
        )

        prediction_records = []
        for record in records:
            record_type = record.get("record_type", "document")
            metadata = build_prediction_metadata(record)

            prediction_records.append(
                PredictionRecord(
                    id=record.get("id"),
                    record_type=record_type,
                    text=record.get("text", ""),
                    sentiment=SentimentInfo(
                        label=record.get("sentiment_label", "neutral"),
                        score=record.get("sentiment_score", 0.0),
                    ),
                    source=record.get("source"),
                    timestamp=record.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                    metadata=metadata,
                )
            )

        end_idx = start_idx + page_size
        return PredictionsResponse(
            predictions=prediction_records,
            total=total,
            page=page,
            page_size=page_size,
            has_more=end_idx < total,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve predictions: {str(e)}",
        )


@router.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str) -> PredictionRecord:
    """
    Get a specific prediction by ID.

    **Use Cases:**
    - Retrieving details of a specific analysis
    - Looking up prediction by reference ID
    - Accessing full prediction metadata

    **Example:**
    ```
    GET /api/v1/data/predictions/abc123
    ```
    """
    try:
        record = storage.get_record_by_id(prediction_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"Prediction with ID '{prediction_id}' not found",
            )

        record_type = record.get("record_type", "document")
        metadata = build_prediction_metadata(record)

        return PredictionRecord(
            id=record.get("id"),
            record_type=record_type,
            text=record.get("text", ""),
            sentiment=SentimentInfo(
                label=record.get("sentiment_label", "neutral"),
                score=record.get("sentiment_score", 0.0),
            ),
            source=record.get("source"),
            timestamp=record.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve prediction: {str(e)}",
        )


@router.get("/stocks/{ticker}/sentiment")
async def get_stock_sentiment(
    ticker: str,
    limit: int = Query(50, description="Maximum number of results", ge=1, le=500),
    start_date: Optional[date] = Query(None, description="Filter from date"),
) -> StockSentimentsResponse:
    """
    Get sentiment analysis history for a specific stock ticker.

    Returns all sentiment mentions and analysis for the specified stock.

    **Use Cases:**
    - Tracking sentiment for a specific stock over time
    - Analyzing investor sentiment trends
    - Identifying sentiment drivers for a stock

    **Example:**
    ```
    GET /api/v1/data/stocks/AAPL/sentiment?limit=50
    ```
    """
    try:
        start_dt = (
            datetime.combine(start_date, datetime.min.time())
            if start_date
            else None
        )

        # Get sentiments for this ticker (stock records only)
        stock_data = storage.get_stock_sentiment(
            ticker.upper(), start_date=start_dt
        )

        if not stock_data:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment data found for ticker '{ticker}'",
            )

        # Apply limit
        stock_data = stock_data[:limit]

        # Lookup company name
        company_info = stock_db.get_by_ticker(ticker.upper()) or {}
        company_name = company_info.get("company_name", ticker.upper())

        # Build sentiment records
        sentiments = []
        for item in stock_data:
            sentiments.append(
                StockSentimentRecord(
                    ticker=item.get("ticker", ticker.upper()),
                    company_name=company_name,
                    sentiment=SentimentInfo(
                        label=item.get("sentiment_label", "neutral"),
                        score=item.get("sentiment_score", 0.0),
                    ),
                    text=item.get("text", ""),
                    timestamp=item.get("timestamp", ""),
                )
            )

        # Calculate summary statistics
        if sentiments:
            positive_count = sum(1 for s in sentiments if s.sentiment.label == "positive")
            negative_count = sum(1 for s in sentiments if s.sentiment.label == "negative")
            neutral_count = sum(1 for s in sentiments if s.sentiment.label == "neutral")
            total = len(sentiments)

            summary = {
                "total_mentions": total,
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "positive_percentage": round((positive_count / total) * 100, 2),
                "negative_percentage": round((negative_count / total) * 100, 2),
                "neutral_percentage": round((neutral_count / total) * 100, 2),
                "average_score": round(
                    sum(s.sentiment.score for s in sentiments) / total, 3
                ),
            }
        else:
            summary = {
                "total_mentions": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "positive_percentage": 0,
                "negative_percentage": 0,
                "neutral_percentage": 0,
                "average_score": 0,
            }

        return StockSentimentsResponse(
            ticker=ticker.upper(),
            sentiments=sentiments,
            total=len(sentiments),
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stock sentiment: {str(e)}",
        )


@router.get("/statistics")
async def get_statistics(
    days: Optional[int] = Query(
        None,
        description="Limit to the last N days relative to the newest record",
        ge=1,
    ),
    data_source: Optional[str] = Query(
        None,
        description="Ingest channel: all (omit), reddit, news, twitter (or x)",
    ),
) -> StatisticsResponse:
    """
    Aggregate statistics from the sentiment database (SQL).

    Optional ``data_source`` filters all aggregates to one channel. When omitted,
    ``source_comparison`` includes Reddit vs news vs X side-by-side sentiment.
    """
    cache_key = ("statistics", days, (data_source or "").lower() or None)
    cached = _stats_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = statistics_service.get_statistics(
            days=days,
            data_source=data_source,
            include_source_comparison=data_source is None,
        )
        sc = raw.pop("source_comparison", None)
        source_model = (
            SourceComparisonResponse(**sc) if isinstance(sc, dict) else None
        )
        response = StatisticsResponse(
            **raw,
            source_comparison=source_model,
        )
        _stats_cache[cache_key] = response
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}",
        )
