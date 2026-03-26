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
from app.storage import StockSentimentStorage

router = APIRouter(prefix="/data", tags=["data"])

# Initialize storage
storage = StockSentimentStorage()
storage.load()

# Initialize stock database for name lookup
stock_db = StockDatabase()
stock_db.load()

# Statistics cache (5 minute TTL)
_stats_cache: TTLCache = TTLCache(maxsize=4, ttl=300)


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


class StatisticsResponse(BaseModel):
    """Response model for aggregate statistics."""

    total_predictions: int = Field(..., description="Total predictions made")
    total_stocks_analyzed: int = Field(..., description="Total unique stocks analyzed")
    sentiment_distribution: dict = Field(..., description="Overall sentiment distribution")
    top_stocks: List[dict] = Field(..., description="Most frequently mentioned stocks")
    recent_activity: dict = Field(..., description="Recent activity statistics")
    date_range: dict = Field(..., description="Data date range")


# Endpoints


@router.get("/predictions")
async def get_predictions(
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(20, description="Items per page", ge=1, le=100),
    source: Optional[str] = Query(None, description="Filter by source (reddit/twitter/news)"),
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

        records, total = storage.query_records(
            source=source_filter,
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
async def get_statistics() -> StatisticsResponse:
    """
    Get aggregate statistics across all data.

    Provides high-level metrics and insights from the entire dataset.

    **Use Cases:**
    - Dashboard overview metrics
    - Understanding overall sentiment trends
    - Identifying most active stocks
    - Monitoring system activity

    **Response includes:**
    - Total predictions and stocks analyzed
    - Overall sentiment distribution
    - Top mentioned stocks
    - Recent activity metrics
    - Data date range

    **Example:**
    ```
    GET /api/v1/data/statistics
    ```
    """
    cached = _stats_cache.get("statistics")
    if cached is not None:
        return cached

    try:
        # Get all data
        all_data = storage.get_all_sentiments()

        # Calculate statistics
        total_predictions = len(all_data)

        # Count sentiments
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        stock_mentions = {}
        timestamps = []

        for item in all_data:
            label = item.get("sentiment_label", "neutral")
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

            ticker = item.get("ticker")
            if ticker:
                if ticker:
                    if ticker not in stock_mentions:
                        company_info = stock_db.get_by_ticker(ticker) or {}
                        company_name = company_info.get(
                            "company_name", item.get("mentioned_as", ticker)
                        )
                        stock_mentions[ticker] = {
                            "ticker": ticker,
                            "company_name": company_name,
                            "count": 0,
                            "positive": 0,
                            "negative": 0,
                            "neutral": 0,
                        }
                    stock_mentions[ticker]["count"] += 1
                    stock_mentions[ticker][label] = stock_mentions[ticker].get(label, 0) + 1

            # Collect timestamps
            ts = item.get("timestamp")
            if ts:
                timestamps.append(ts)

        # Get top stocks
        top_stocks = sorted(
            stock_mentions.values(),
            key=lambda x: x["count"],
            reverse=True,
        )[:10]

        # Calculate date range
        timestamps_dt: list[datetime] = []
        if timestamps:
            timestamps_dt = [
                datetime.fromisoformat(ts.replace("Z", "")) for ts in timestamps
            ]
            date_range = {
                "earliest": min(timestamps_dt).isoformat() + "Z",
                "latest": max(timestamps_dt).isoformat() + "Z",
            }
        else:
            date_range = {"earliest": None, "latest": None}

        # Recent activity — anchor to the latest data point so that
        # historical datasets (e.g. 2021-2022) show meaningful numbers.
        if timestamps_dt:
            anchor = max(timestamps_dt)
        else:
            anchor = datetime.utcnow()
        day_ago = anchor - timedelta(days=1)
        week_ago = anchor - timedelta(days=7)
        month_ago = anchor - timedelta(days=30)

        recent_counts = {"last_24h": 0, "last_7d": 0, "last_30d": 0}
        for ts in timestamps:
            ts_dt = datetime.fromisoformat(ts.replace("Z", ""))
            if ts_dt >= day_ago:
                recent_counts["last_24h"] += 1
            if ts_dt >= week_ago:
                recent_counts["last_7d"] += 1
            if ts_dt >= month_ago:
                recent_counts["last_30d"] += 1

        # Calculate sentiment distribution percentages
        total_with_sentiment = sum(sentiment_counts.values())
        sentiment_distribution = {
            "positive": sentiment_counts["positive"],
            "negative": sentiment_counts["negative"],
            "neutral": sentiment_counts["neutral"],
            "positive_percentage": (
                round((sentiment_counts["positive"] / total_with_sentiment) * 100, 2)
                if total_with_sentiment > 0
                else 0
            ),
            "negative_percentage": (
                round((sentiment_counts["negative"] / total_with_sentiment) * 100, 2)
                if total_with_sentiment > 0
                else 0
            ),
            "neutral_percentage": (
                round((sentiment_counts["neutral"] / total_with_sentiment) * 100, 2)
                if total_with_sentiment > 0
                else 0
            ),
        }

        response = StatisticsResponse(
            total_predictions=total_predictions,
            total_stocks_analyzed=len(stock_mentions),
            sentiment_distribution=sentiment_distribution,
            top_stocks=top_stocks,
            recent_activity=recent_counts,
            date_range=date_range,
        )
        _stats_cache["statistics"] = response
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}",
        )
