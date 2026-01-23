"""
Stock sentiment API endpoints.

Provides REST API for stock entity extraction and sentiment analysis.
"""

from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.stocks import analyze_stock_sentiment
from app.storage import StockSentimentStorage

router = APIRouter(prefix="/stocks", tags=["stocks"])

# Initialize storage
storage = StockSentimentStorage()
storage.load()


# Request/Response Models
class StockAnalysisRequest(BaseModel):
    """Request model for stock sentiment analysis."""

    text: str = Field(..., description="Text to analyze", min_length=1)
    extract_context: bool = Field(
        True, description="Extract context around stock mentions"
    )
    include_movements: bool = Field(
        True, description="Include stock price movement detection"
    )
    source: Optional[str] = Field(
        None, description="Source type (reddit/twitter/news)"
    )


class SentimentInfo(BaseModel):
    """Sentiment information."""

    label: str = Field(..., description="Sentiment label")
    score: float = Field(..., description="Confidence score (0-1)")


class StockMention(BaseModel):
    """Stock mention with sentiment."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Company full name")
    mentioned_as: str = Field(
        ..., description="How stock was mentioned in text"
    )
    sentiment: SentimentInfo
    context: Optional[str] = Field(None, description="Context around mention")
    position: Optional[dict] = Field(
        None, description="Character position in text"
    )


class StockAnalysisResponse(BaseModel):
    """Response model for stock sentiment analysis."""

    text: str = Field(..., description="Original text")
    overall_sentiment: dict = Field(
        ..., description="Overall text sentiment"
    )
    stocks: List[StockMention] = Field(
        ..., description="List of stock mentions with sentiment"
    )
    metadata: dict = Field(..., description="Analysis metadata")


class StockSentimentResponse(BaseModel):
    """Response model for stock sentiment query."""

    ticker: str
    total_mentions: int
    average_score: float
    sentiment_distribution: dict
    records: Optional[List[dict]] = None


class TrendingStock(BaseModel):
    """Trending stock information."""

    ticker: str
    mentions: int


class TrendingStocksResponse(BaseModel):
    """Response model for trending stocks."""

    trending: List[TrendingStock]
    period_hours: int
    total_stocks: int


# Endpoints


@router.post("/analyze", response_model=StockAnalysisResponse)
async def analyze_stock_sentiment_endpoint(
    request: StockAnalysisRequest,
) -> StockAnalysisResponse:
    """
    Analyze text and extract stock-sentiment pairs.

    Extracts stocks by:
    - Ticker symbols ($AAPL)
    - Company names (Apple) using NER

    Analyzes sentiment for each stock mention.

    **Example Request:**
    ```json
    {
        "text": "Apple reported strong earnings while Tesla faced delays",
        "extract_context": true,
        "include_movements": true,
        "source": "news"
    }
    ```

    **Example Response:**
    ```json
    {
        "text": "Apple reported strong earnings...",
        "overall_sentiment": {
            "label": "positive",
            "score": 0.89
        },
        "stocks": [
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "mentioned_as": "Apple",
                "sentiment": {"label": "positive", "score": 0.92},
                "context": "Apple reported strong earnings"
            }
        ]
    }
    ```
    """
    try:
        # Analyze stock sentiment
        result = analyze_stock_sentiment(
            text=request.text,
            extract_context=request.extract_context,
            include_movements=request.include_movements,
        )

        # Save to storage if source is provided
        if request.source:
            storage.save_analysis_result(result, source=request.source)

        return StockAnalysisResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/sentiment", response_model=StockSentimentResponse)
async def get_stock_sentiment(
    ticker: str,
    start_date: Optional[date] = Query(
        None, description="Filter by start date (YYYY-MM-DD)"
    ),
    end_date: Optional[date] = Query(
        None, description="Filter by end date (YYYY-MM-DD)"
    ),
    source: Optional[str] = Query(
        None, description="Filter by source (reddit/twitter/news)"
    ),
    include_records: bool = Query(
        False, description="Include individual records"
    ),
) -> StockSentimentResponse:
    """
    Get aggregated sentiment for a specific stock.

    Returns:
    - Total mentions
    - Average sentiment score
    - Sentiment distribution (positive/negative/neutral)
    - Optional: Individual records

    **Example:**
    ```
    GET /stocks/AAPL/sentiment?start_date=2026-01-01&end_date=2026-01-23
    ```
    """
    try:
        # Convert dates to datetime
        start_dt = (
            datetime.combine(start_date, datetime.min.time())
            if start_date
            else None
        )
        end_dt = (
            datetime.combine(end_date, datetime.max.time())
            if end_date
            else None
        )

        # Get aggregated sentiment
        aggregated = storage.aggregate_sentiment(
            ticker=ticker.upper(), start_date=start_dt, end_date=end_dt
        )

        # Get individual records if requested
        records = None
        if include_records:
            records = storage.get_stock_sentiment(
                ticker=ticker.upper(),
                start_date=start_dt,
                end_date=end_dt,
                source=source,
            )

        return StockSentimentResponse(**aggregated, records=records)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/mentions")
async def get_stock_mentions(
    ticker: str,
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    source: Optional[str] = Query(
        None, description="Filter by source (reddit/twitter/news)"
    ),
):
    """
    Get recent mentions of a stock.

    Returns list of individual stock mentions with sentiment.

    **Example:**
    ```
    GET /stocks/AAPL/mentions?limit=10&source=reddit
    ```
    """
    try:
        records = storage.get_stock_sentiment(
            ticker=ticker.upper(), source=source
        )

        # Apply pagination
        paginated = records[offset : offset + limit]

        return {
            "ticker": ticker.upper(),
            "total_count": len(records),
            "returned_count": len(paginated),
            "offset": offset,
            "mentions": paginated,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trending", response_model=TrendingStocksResponse)
async def get_trending_stocks(
    period: str = Query(
        "24h",
        description="Time period (24h, 7d, 30d)",
        regex="^(24h|7d|30d)$",
    ),
    min_mentions: int = Query(
        5, ge=1, description="Minimum mentions required"
    ),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
) -> TrendingStocksResponse:
    """
    Get stocks with most mentions in recent period.

    Useful for identifying popular/trending stocks in investor discussions.

    **Example:**
    ```
    GET /stocks/trending?period=24h&min_mentions=10&limit=20
    ```

    Returns stocks sorted by mention count (descending).
    """
    try:
        # Parse period to hours
        period_map = {"24h": 24, "7d": 168, "30d": 720}
        hours = period_map.get(period, 24)

        # Get trending stocks
        trending_data = storage.get_trending_stocks(
            min_mentions=min_mentions, hours=hours
        )

        # Convert dicts to TrendingStock models
        trending = [
            TrendingStock(ticker=item["ticker"], mentions=item["mentions"])
            for item in trending_data
        ]

        # Apply limit
        trending = trending[:limit]

        return TrendingStocksResponse(
            trending=trending, period_hours=hours, total_stocks=len(trending)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_stocks(
    tickers: List[str] = Query(..., description="List of ticker symbols"),
    start_date: Optional[date] = Query(None, description="Start date"),
    end_date: Optional[date] = Query(None, description="End date"),
):
    """
    Compare sentiment across multiple stocks.

    **Example:**
    ```
    POST /stocks/compare?tickers=AAPL&tickers=TSLA&tickers=MSFT
    ```

    Returns sentiment comparison for all requested tickers.
    """
    try:
        # Convert dates
        start_dt = (
            datetime.combine(start_date, datetime.min.time())
            if start_date
            else None
        )
        end_dt = (
            datetime.combine(end_date, datetime.max.time())
            if end_date
            else None
        )

        # Get sentiment for each ticker
        comparison = []
        for ticker in tickers:
            aggregated = storage.aggregate_sentiment(
                ticker=ticker.upper(), start_date=start_dt, end_date=end_dt
            )
            comparison.append(aggregated)

        return {"tickers": [t.upper() for t in tickers], "comparison": comparison}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics():
    """
    Get overall stock sentiment statistics.

    Returns:
    - Total stock sentiment records
    - Number of unique tickers tracked
    - Last update timestamp
    """
    try:
        stats = storage.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
