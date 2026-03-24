"""
Correlation analysis API endpoints.

Provides REST API for sentiment-price correlation analysis,
lag analysis, and merged time-series data.

Includes TTL caching for expensive operations to reduce response times.
"""

import logging
from datetime import date, datetime, time
from functools import lru_cache
from typing import List, Optional, Tuple

from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.analysis.correlation import CorrelationAnalyzer
from app.analysis.price_service import PriceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/correlation", tags=["correlation"])


@lru_cache(maxsize=1)
def get_analyzer() -> CorrelationAnalyzer:
    """Lazily initialize heavy analyzer state on first correlation request."""
    return CorrelationAnalyzer()


def _optional_range_datetimes(
    start_date: Optional[date],
    end_date: Optional[date],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Validate optional calendar range and return inclusive UTC-naive datetimes."""
    if start_date is None and end_date is None:
        return None, None
    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=400,
            detail="Both start_date and end_date are required for a custom range.",
        )
    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date must be on or before end_date.",
        )
    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max.replace(microsecond=0))
    return start_dt, end_dt


def _range_cache_suffix(
    start_date: Optional[date], end_date: Optional[date]
) -> str:
    if start_date is not None and end_date is not None:
        return f":{start_date.isoformat()}:{end_date.isoformat()}"
    return ""


# --- TTL Caches ---
# Correlation results: cache for 1 hour (3600 seconds)
_correlation_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)
# Overview results: cache for 30 minutes
_overview_cache: TTLCache = TTLCache(maxsize=16, ttl=1800)
# Granger results: cache for 1 hour
_granger_cache: TTLCache = TTLCache(maxsize=128, ttl=3600)
# Rolling correlation: cache for 1 hour
_rolling_cache: TTLCache = TTLCache(maxsize=128, ttl=3600)
# Time series: cache for 30 minutes
_timeseries_cache: TTLCache = TTLCache(maxsize=128, ttl=1800)
# Statistics: cache for 5 minutes
_statistics_cache: TTLCache = TTLCache(maxsize=16, ttl=300)


# --- Response Models ---


class CorrelationResult(BaseModel):
    """Correlation coefficient with statistical significance."""

    coefficient: float = Field(..., description="Correlation coefficient (-1 to +1)")
    p_value: float = Field(..., description="Statistical p-value")
    significant: bool = Field(
        ..., description="Whether correlation is statistically significant (p < 0.05)"
    )
    interpretation: str = Field(
        ..., description="Human-readable interpretation of the correlation"
    )


class CorrelationResponse(BaseModel):
    """Response model for correlation analysis."""

    ticker: str
    data_points: int
    period: Optional[str] = None
    sentiment_metric: Optional[str] = None
    price_metric: Optional[str] = None
    pearson: Optional[CorrelationResult] = None
    spearman: Optional[CorrelationResult] = None
    error: Optional[str] = None


class LagResult(BaseModel):
    """Single lag correlation result."""

    lag_days: int
    data_points: int
    pearson_r: Optional[float] = None
    p_value: Optional[float] = None
    significant: Optional[bool] = None
    description: str


class LagAnalysisResponse(BaseModel):
    """Response model for lag analysis."""

    ticker: str
    max_lag_days: int
    lags: List[LagResult]
    best_lag: Optional[LagResult] = None
    error: Optional[str] = None


class TimeSeriesPoint(BaseModel):
    """Single data point in the merged time series."""

    date: str
    close: float
    returns: Optional[float] = None
    avg_sentiment_score: float
    net_sentiment: float
    mention_count: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float


class TimeSeriesResponse(BaseModel):
    """Response model for merged time-series data."""

    ticker: str
    data_points: int
    series: List[TimeSeriesPoint]


class CorrelationOverviewItem(BaseModel):
    """Single stock in the correlation overview."""

    ticker: str
    mentions: int
    data_points: int
    pearson_r: float
    pearson_p: float
    significant: bool
    interpretation: str
    spearman_r: float


class GrangerLagResult(BaseModel):
    """Single lag result from Granger causality test."""
    lag: int
    f_statistic: float
    p_value: float
    significant: bool


class GrangerSummary(BaseModel):
    """Summary of Granger causality test."""
    sentiment_predicts_price: bool
    price_predicts_sentiment: bool
    best_sentiment_to_price_lag: Optional[GrangerLagResult] = None
    best_price_to_sentiment_lag: Optional[GrangerLagResult] = None
    interpretation: str


class GrangerCausalityResponse(BaseModel):
    """Response model for Granger causality analysis."""
    ticker: str
    max_lag: Optional[int] = None
    data_points: Optional[int] = None
    sentiment_to_price: Optional[List[GrangerLagResult]] = None
    price_to_sentiment: Optional[List[GrangerLagResult]] = None
    summary: Optional[GrangerSummary] = None
    error: Optional[str] = None


class RollingCorrelationPoint(BaseModel):
    """Single point in rolling correlation time series."""
    date: str
    correlation: float
    window_start: str


class RollingCorrelationStats(BaseModel):
    """Statistics for rolling correlation series."""
    mean_correlation: float
    std_correlation: float
    min_correlation: float
    max_correlation: float
    periods_positive: int
    periods_negative: int


class RollingCorrelationResponse(BaseModel):
    """Response model for rolling correlation analysis."""
    ticker: str
    window: int
    period: Optional[str] = None
    data_points: int
    series: List[RollingCorrelationPoint]
    statistics: Optional[RollingCorrelationStats] = None
    error: Optional[str] = None


class PriceHistoryPoint(BaseModel):
    """Single price data point."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    returns: Optional[float] = None


class PriceHistoryResponse(BaseModel):
    """Response model for price history."""

    ticker: str
    data_points: int
    series: List[PriceHistoryPoint]


class StockInfoResponse(BaseModel):
    """Response model for stock info."""

    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    currency: Optional[str] = None


# --- Endpoints ---


@router.get("/{ticker}", response_model=CorrelationResponse)
async def get_correlation(
    ticker: str,
    period: str = Query(
        "90d",
        description="Price data period (7d, 30d, 90d, 6mo, 1y); ignored if start_date+end_date set",
    ),
    start_date: Optional[date] = Query(
        None, description="Custom range start (inclusive), requires end_date"
    ),
    end_date: Optional[date] = Query(
        None, description="Custom range end (inclusive), requires start_date"
    ),
    sentiment_metric: str = Query(
        "net_sentiment",
        description="Sentiment metric to correlate (net_sentiment, avg_score)",
    ),
    price_metric: str = Query(
        "returns",
        description="Price metric to correlate (returns, close)",
    ),
) -> CorrelationResponse:
    """
    Calculate correlation between sentiment and stock price for a ticker.

    Returns Pearson and Spearman correlation coefficients with p-values
    and statistical significance.

    **Example:**
    ```
    GET /api/v1/correlation/AAPL?period=90d&sentiment_metric=net_sentiment
    ```
    """
    start_dt, end_dt = _optional_range_datetimes(start_date, end_date)
    cache_key = (
        f"{ticker.upper()}:{period}:{sentiment_metric}:{price_metric}"
        f"{_range_cache_suffix(start_date, end_date)}"
    )
    cached = _correlation_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        result = get_analyzer().calculate_correlation(
            ticker=ticker.upper(),
            period=period,
            sentiment_metric=sentiment_metric,
            price_metric=price_metric,
            start_date=start_dt,
            end_date=end_dt,
        )
        response = CorrelationResponse(**result)
        _correlation_cache[cache_key] = response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/timeseries", response_model=TimeSeriesResponse)
async def get_correlation_timeseries(
    ticker: str,
    period: str = Query("90d", description="Price data period"),
) -> TimeSeriesResponse:
    """
    Get merged sentiment + price time-series data for charting.

    Returns daily data points with both sentiment scores and price data,
    suitable for dual-axis visualization.

    **Example:**
    ```
    GET /api/v1/correlation/AAPL/timeseries?period=90d
    ```
    """
    start_dt, end_dt = _optional_range_datetimes(start_date, end_date)
    cache_key = (
        f"ts:{ticker.upper()}:{period}{_range_cache_suffix(start_date, end_date)}"
    )
    cached = _timeseries_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        result = get_analyzer().get_timeseries_response(
            ticker=ticker.upper(),
            period=period,
            start_date=start_dt,
            end_date=end_dt,
        )
        response = TimeSeriesResponse(**result)
        _timeseries_cache[cache_key] = response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/lag-analysis", response_model=LagAnalysisResponse)
async def get_lag_analysis(
    ticker: str,
    max_lag_days: int = Query(5, ge=1, le=14, description="Maximum lag in days"),
    period: str = Query("90d", description="Price data period"),
    start_date: Optional[date] = Query(None, description="Custom range start (inclusive)"),
    end_date: Optional[date] = Query(None, description="Custom range end (inclusive)"),
    sentiment_metric: str = Query(
        "net_sentiment", description="Sentiment metric to use"
    ),
) -> LagAnalysisResponse:
    """
    Perform lag correlation analysis.

    Tests whether sentiment at time t predicts price at time t+lag.
    Helps determine if sentiment is a leading indicator for price.

    - Positive lag: sentiment leads price
    - Negative lag: price leads sentiment
    - Zero lag: same-day correlation

    **Example:**
    ```
    GET /api/v1/correlation/AAPL/lag-analysis?max_lag_days=5
    ```
    """
    try:
        result = get_analyzer().lag_analysis(
            ticker=ticker.upper(),
            max_lag_days=max_lag_days,
            period=period,
            sentiment_metric=sentiment_metric,
        )
        return LagAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/granger", response_model=GrangerCausalityResponse)
async def get_granger_causality(
    ticker: str,
    max_lag: int = Query(5, ge=1, le=10, description="Maximum lag in days"),
    period: str = Query("90d", description="Price data period"),
    start_date: Optional[date] = Query(None, description="Custom range start (inclusive)"),
    end_date: Optional[date] = Query(None, description="Custom range end (inclusive)"),
    sentiment_metric: str = Query(
        "net_sentiment", description="Sentiment metric to use"
    ),
) -> GrangerCausalityResponse:
    """
    Test Granger causality between sentiment and price returns.

    Tests whether past sentiment helps predict future price (and vice versa).
    This is a key test for the research question of whether sentiment leads price.

    **Example:**
    ```
    GET /api/v1/correlation/AAPL/granger?max_lag=5
    ```
    """
    start_dt, end_dt = _optional_range_datetimes(start_date, end_date)
    cache_key = (
        f"granger:{ticker.upper()}:{max_lag}:{period}:{sentiment_metric}"
        f"{_range_cache_suffix(start_date, end_date)}"
    )
    cached = _granger_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        result = get_analyzer().granger_causality(
            ticker=ticker.upper(),
            max_lag=max_lag,
            period=period,
            sentiment_metric=sentiment_metric,
            start_date=start_dt,
            end_date=end_dt,
        )
        response = GrangerCausalityResponse(**result)
        _granger_cache[cache_key] = response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/rolling", response_model=RollingCorrelationResponse)
async def get_rolling_correlation(
    ticker: str,
    window: int = Query(14, ge=5, le=60, description="Rolling window size in days"),
    period: str = Query("90d", description="Price data period"),
    start_date: Optional[date] = Query(None, description="Custom range start (inclusive)"),
    end_date: Optional[date] = Query(None, description="Custom range end (inclusive)"),
    sentiment_metric: str = Query(
        "net_sentiment", description="Sentiment metric to use"
    ),
    price_metric: str = Query(
        "returns", description="Price metric to use"
    ),
) -> RollingCorrelationResponse:
    """
    Calculate rolling windowed correlation between sentiment and price.

    Shows how the correlation changes over time using a sliding window.
    Reveals whether correlation strengthens during certain market periods.

    **Example:**
    ```
    GET /api/v1/correlation/AAPL/rolling?window=14&period=90d
    ```
    """
    start_dt, end_dt = _optional_range_datetimes(start_date, end_date)
    cache_key = (
        f"rolling:{ticker.upper()}:{window}:{period}:{sentiment_metric}:{price_metric}"
        f"{_range_cache_suffix(start_date, end_date)}"
    )
    cached = _rolling_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        result = get_analyzer().rolling_correlation(
            ticker=ticker.upper(),
            window=window,
            period=period,
            sentiment_metric=sentiment_metric,
            price_metric=price_metric,
            start_date=start_dt,
            end_date=end_dt,
        )
        response = RollingCorrelationResponse(**result)
        _rolling_cache[cache_key] = response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overview/all")
async def get_correlation_overview(
    min_mentions: int = Query(3, ge=1, description="Minimum mentions required"),
    period: str = Query("90d", description="Price data period"),
) -> List[CorrelationOverviewItem]:
    """
    Get correlation overview for all tracked stocks.

    Returns correlation summaries sorted by absolute correlation strength.
    Only includes stocks with sufficient data points.

    **Example:**
    ```
    GET /api/v1/correlation/overview/all?min_mentions=5
    ```
    """
    cache_key = f"overview:{min_mentions}:{period}"
    cached = _overview_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        results = get_analyzer().get_correlation_overview(
            min_mentions=min_mentions, period=period
        )
        response = [CorrelationOverviewItem(**r) for r in results]
        _overview_cache[cache_key] = response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/price-history", response_model=PriceHistoryResponse)
async def get_price_history(
    ticker: str,
    period: str = Query("90d", description="Price data period"),
) -> PriceHistoryResponse:
    """
    Get historical price data for a stock.

    **Example:**
    ```
    GET /api/v1/correlation/AAPL/price-history?period=90d
    ```
    """
    try:
        df = PriceService.get_price_history(ticker.upper(), period=period)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No price data found for ticker '{ticker}'",
            )

        series = []
        for _, row in df.iterrows():
            import math

            returns_val = row.get("returns")
            series.append(
                PriceHistoryPoint(
                    date=row["date"].strftime("%Y-%m-%d"),
                    open=round(float(row["Open"]), 2),
                    high=round(float(row["High"]), 2),
                    low=round(float(row["Low"]), 2),
                    close=round(float(row["Close"]), 2),
                    volume=int(row["Volume"]),
                    returns=round(float(returns_val), 6)
                    if returns_val is not None and not math.isnan(returns_val)
                    else None,
                )
            )

        return PriceHistoryResponse(
            ticker=ticker.upper(),
            data_points=len(series),
            series=series,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/info", response_model=StockInfoResponse)
async def get_stock_info(ticker: str) -> StockInfoResponse:
    """
    Get basic stock information (name, sector, market cap).

    **Example:**
    ```
    GET /api/v1/correlation/AAPL/info
    ```
    """
    try:
        info = PriceService.get_stock_info(ticker.upper())
        return StockInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
