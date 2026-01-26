"""
Data retrieval API endpoints.

Provides access to historical predictions, stock sentiments, and statistics.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.storage import StockSentimentStorage

router = APIRouter(prefix="/data", tags=["data"])

# Initialize storage
storage = StockSentimentStorage()
storage.load()


# Request/Response Models
class SentimentInfo(BaseModel):
    """Sentiment information."""

    label: str = Field(..., description="Sentiment label")
    score: float = Field(..., description="Confidence score (0-1)")


class PredictionRecord(BaseModel):
    """Historical prediction record."""

    id: str = Field(..., description="Prediction ID")
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
    context: str = Field(..., description="Text context around stock mention")
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
        # Get all stock sentiments from storage
        all_data = storage.get_all_sentiments()

        # Convert to prediction records
        predictions = []
        for item in all_data:
            # Extract sentiment info
            overall_sent = item.get("overall_sentiment", {})

            predictions.append({
                "id": item.get("id", str(hash(item.get("text", "")))),
                "text": item.get("text", ""),
                "sentiment": {
                    "label": overall_sent.get("label", "neutral"),
                    "score": overall_sent.get("score", 0.0),
                },
                "source": item.get("source"),
                "timestamp": item.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                "metadata": {
                    "num_stocks": len(item.get("stocks", [])),
                    "has_stocks": len(item.get("stocks", [])) > 0,
                },
            })

        # Apply filters
        if source:
            predictions = [p for p in predictions if p.get("source") == source]

        if sentiment:
            predictions = [
                p for p in predictions 
                if p["sentiment"]["label"].lower() == sentiment.lower()
            ]

        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            predictions = [
                p for p in predictions
                if datetime.fromisoformat(p["timestamp"].replace("Z", "")) >= start_dt
            ]

        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            predictions = [
                p for p in predictions
                if datetime.fromisoformat(p["timestamp"].replace("Z", "")) <= end_dt
            ]

        # Calculate pagination
        total = len(predictions)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        # Get page of results
        page_predictions = predictions[start_idx:end_idx]

        # Convert to Pydantic models
        prediction_records = [
            PredictionRecord(
                id=p["id"],
                text=p["text"],
                sentiment=SentimentInfo(**p["sentiment"]),
                source=p.get("source"),
                timestamp=p["timestamp"],
                metadata=p.get("metadata"),
            )
            for p in page_predictions
        ]

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
        # Get all sentiments and find by ID
        all_data = storage.get_all_sentiments()

        for item in all_data:
            item_id = item.get("id", str(hash(item.get("text", ""))))
            if item_id == prediction_id:
                overall_sent = item.get("overall_sentiment", {})

                return PredictionRecord(
                    id=item_id,
                    text=item.get("text", ""),
                    sentiment=SentimentInfo(
                        label=overall_sent.get("label", "neutral"),
                        score=overall_sent.get("score", 0.0),
                    ),
                    source=item.get("source"),
                    timestamp=item.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                    metadata={
                        "num_stocks": len(item.get("stocks", [])),
                        "stocks": item.get("stocks", []),
                    },
                )

        raise HTTPException(
            status_code=404,
            detail=f"Prediction with ID '{prediction_id}' not found",
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
        # Get sentiments for this ticker
        stock_data = storage.get_stock_sentiments(ticker.upper())

        if not stock_data:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment data found for ticker '{ticker}'",
            )

        # Apply date filter if specified
        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            stock_data = [
                item for item in stock_data
                if datetime.fromisoformat(
                    item.get("timestamp", "").replace("Z", "")
                ) >= start_dt
            ]

        # Apply limit
        stock_data = stock_data[:limit]

        # Build sentiment records
        sentiments = []
        for item in stock_data:
            for stock in item.get("stocks", []):
                if stock.get("ticker", "").upper() == ticker.upper():
                    sentiments.append(
                        StockSentimentRecord(
                            ticker=stock["ticker"],
                            company_name=stock.get("company_name", ticker),
                            sentiment=SentimentInfo(
                                label=stock["sentiment"]["label"],
                                score=stock["sentiment"]["score"],
                            ),
                            context=stock.get("context", ""),
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
            # Count overall sentiment
            overall_sent = item.get("overall_sentiment", {})
            label = overall_sent.get("label", "neutral")
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

            # Count stock mentions
            for stock in item.get("stocks", []):
                ticker = stock.get("ticker", "")
                if ticker:
                    if ticker not in stock_mentions:
                        stock_mentions[ticker] = {
                            "ticker": ticker,
                            "company_name": stock.get("company_name", ticker),
                            "count": 0,
                            "positive": 0,
                            "negative": 0,
                            "neutral": 0,
                        }
                    stock_mentions[ticker]["count"] += 1
                    stock_sentiment = stock.get("sentiment", {}).get("label", "neutral")
                    stock_mentions[ticker][stock_sentiment] += 1

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

        # Recent activity (last 24 hours, 7 days, 30 days)
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

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

        return StatisticsResponse(
            total_predictions=total_predictions,
            total_stocks_analyzed=len(stock_mentions),
            sentiment_distribution=sentiment_distribution,
            top_stocks=top_stocks,
            recent_activity=recent_counts,
            date_range=date_range,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}",
        )
