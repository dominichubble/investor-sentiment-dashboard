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


def _methodology_cache_suffix(
    *,
    data_source: Optional[str],
    min_mentions_per_day: int,
    align_mode: str,
    market_adjustment: str,
    price_metric: Optional[str] = None,
) -> str:
    pm = price_metric or "auto"
    ds = data_source or "all"
    return (
        f":ds={ds}:mnp={min_mentions_per_day}:am={align_mode}:mm={market_adjustment}:pm={pm}"
    )


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
# Out-of-sample split: 1 hour
_oos_cache: TTLCache = TTLCache(maxsize=64, ttl=3600)


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
    effective_price_metric: Optional[str] = Field(
        None,
        description="Column actually used (e.g. forward_1d_return, excess_returns)",
    )
    align_mode: Optional[str] = None
    market_adjustment: Optional[str] = None
    spy_beta: Optional[float] = Field(
        None, description="OLS beta of stock on SPY returns when market-adjusted"
    )
    data_source: Optional[str] = None
    min_mentions_per_day: Optional[int] = None
    trailing_days: Optional[int] = Field(
        None,
        description="Trailing window (days) for net_sentiment rolling mean used in correlation",
    )
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
    trailing_days: Optional[int] = None
    error: Optional[str] = None


class TimeSeriesPoint(BaseModel):
    """Single data point in the merged time series."""

    date: str
    close: float
    returns: Optional[float] = None
    spy_returns: Optional[float] = None
    excess_returns: Optional[float] = None
    forward_1d_return: Optional[float] = None
    forward_excess_return: Optional[float] = None
    avg_sentiment_score: float
    net_sentiment: float
    trailing_net_sentiment: float = Field(
        ...,
        description="Causal rolling mean of net_sentiment over trailing_days",
    )
    mention_count: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float


class TimeSeriesResponse(BaseModel):
    """Response model for merged time-series data."""

    ticker: str
    data_points: int
    trailing_days: int = Field(
        1, description="Window length used for trailing_net_sentiment"
    )
    spy_beta: Optional[float] = None
    align_mode: Optional[str] = None
    market_adjustment: Optional[str] = None
    series: List[TimeSeriesPoint]


class CorrelationOverviewItem(BaseModel):
    """Single stock in the correlation overview."""

    ticker: str
    mentions: int
    data_points: int
    pearson_r: float
    pearson_p: float
    significant: bool
    significant_bonferroni: bool = Field(
        False,
        description="True if pearson_p < 0.05/n_tickers (Bonferroni family-wise check)",
    )
    interpretation: str
    spearman_r: float


class CorrelationOverviewResponse(BaseModel):
    """Multi-ticker correlation scan with multiple-testing metadata."""

    n_tickers_tested: int
    alpha_individual: float = 0.05
    alpha_bonferroni: Optional[float] = None
    align_mode: str = "same_day"
    market_adjustment: str = "none"
    data_source: Optional[str] = None
    items: List[CorrelationOverviewItem]


class OutOfSampleBlock(BaseModel):
    label: str
    n: int
    pearson_r: Optional[float] = None
    pearson_p: Optional[float] = None
    significant: bool = False


class OutOfSampleResponse(BaseModel):
    ticker: str
    error: Optional[str] = None
    train_ratio: Optional[float] = None
    split_date: Optional[str] = None
    effective_price_metric: Optional[str] = None
    sentiment_metric: Optional[str] = None
    align_mode: Optional[str] = None
    market_adjustment: Optional[str] = None
    spy_beta: Optional[float] = None
    train: Optional[OutOfSampleBlock] = None
    test: Optional[OutOfSampleBlock] = None
    trailing_days: Optional[int] = None


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
    trailing_days: Optional[int] = None
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
    trailing_days: Optional[int] = None
    effective_price_metric: Optional[str] = None
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
    sentiment_metric: str = Query(
        "net_sentiment",
        description="Sentiment metric to correlate (net_sentiment, avg_score)",
    ),
    price_metric: Optional[str] = Query(
        None,
        description="Price column: omit or 'auto' to derive from align_mode and market_adjustment",
    ),
    start_date: Optional[date] = Query(
        None, description="Custom range start (inclusive), requires end_date"
    ),
    end_date: Optional[date] = Query(
        None, description="Custom range end (inclusive), requires start_date"
    ),
    trailing_days: int = Query(
        1,
        ge=1,
        le=30,
        description="Trailing window (days) for net_sentiment rolling mean; 1 = same-day only",
    ),
    data_source: Optional[str] = Query(
        None,
        description="Filter sentiment rows (e.g. reddit, news, twitter); omit for all channels",
    ),
    min_mentions_per_day: int = Query(
        1,
        ge=1,
        le=500,
        description="Drop calendar days with fewer than this many mentions before merge",
    ),
    align_mode: str = Query(
        "same_day",
        description="same_day (concurrent) or sentiment_leads_1d (sentiment vs next close-to-close return)",
    ),
    market_adjustment: str = Query(
        "none",
        description="none or spy_beta_residual (correlate vs stock return minus beta*SPY)",
    ),
) -> CorrelationResponse:
    """
    Calculate correlation between sentiment and stock price for a ticker.

    Returns Pearson and Spearman correlation coefficients with p-values
    and statistical significance.
    """
    start_dt, end_dt = _optional_range_datetimes(start_date, end_date)
    cache_key = (
        f"{ticker.upper()}:{period}:{sentiment_metric}:td{trailing_days}"
        f"{_range_cache_suffix(start_date, end_date)}"
        f"{_methodology_cache_suffix(data_source=data_source, min_mentions_per_day=min_mentions_per_day, align_mode=align_mode, market_adjustment=market_adjustment, price_metric=price_metric)}"
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
            trailing_days=trailing_days,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=align_mode,
            market_adjustment=market_adjustment,
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
    start_date: Optional[date] = Query(
        None, description="Custom range start (inclusive), requires end_date"
    ),
    end_date: Optional[date] = Query(
        None, description="Custom range end (inclusive), requires start_date"
    ),
    trailing_days: int = Query(
        1,
        ge=1,
        le=30,
        description="Trailing window (days) for trailing_net_sentiment",
    ),
    data_source: Optional[str] = Query(None, description="Filter sentiment by data_source"),
    min_mentions_per_day: int = Query(1, ge=1, le=500),
    align_mode: str = Query("same_day"),
    market_adjustment: str = Query("none"),
) -> TimeSeriesResponse:
    """
    Get merged sentiment + price time-series data for charting.

    Returns daily data points with both sentiment scores and price data,
    suitable for dual-axis visualization.
    """
    start_dt, end_dt = _optional_range_datetimes(start_date, end_date)
    cache_key = (
        f"ts:{ticker.upper()}:{period}:td{trailing_days}"
        f"{_range_cache_suffix(start_date, end_date)}"
        f"{_methodology_cache_suffix(data_source=data_source, min_mentions_per_day=min_mentions_per_day, align_mode=align_mode, market_adjustment=market_adjustment)}"
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
            trailing_days=trailing_days,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=align_mode,
            market_adjustment=market_adjustment,
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
    sentiment_metric: str = Query(
        "net_sentiment", description="Sentiment metric to use"
    ),
    start_date: Optional[date] = Query(None, description="Custom range start (inclusive)"),
    end_date: Optional[date] = Query(None, description="Custom range end (inclusive)"),
    trailing_days: int = Query(
        1,
        ge=1,
        le=30,
        description="Trailing window (days) for net_sentiment when using net_sentiment metric",
    ),
    data_source: Optional[str] = Query(None),
    min_mentions_per_day: int = Query(1, ge=1, le=500),
    align_mode: str = Query("same_day"),
    market_adjustment: str = Query("none"),
) -> LagAnalysisResponse:
    """
    Perform lag correlation analysis.

    Tests whether sentiment at time t predicts price at time t+lag.
    """
    start_dt, end_dt = _optional_range_datetimes(start_date, end_date)
    try:
        result = get_analyzer().lag_analysis(
            ticker=ticker.upper(),
            max_lag_days=max_lag_days,
            period=period,
            sentiment_metric=sentiment_metric,
            start_date=start_dt,
            end_date=end_dt,
            trailing_days=trailing_days,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=align_mode,
            market_adjustment=market_adjustment,
        )
        return LagAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/granger", response_model=GrangerCausalityResponse)
async def get_granger_causality(
    ticker: str,
    max_lag: int = Query(5, ge=1, le=10, description="Maximum lag in days"),
    period: str = Query("90d", description="Price data period"),
    sentiment_metric: str = Query(
        "net_sentiment", description="Sentiment metric to use"
    ),
    start_date: Optional[date] = Query(None, description="Custom range start (inclusive)"),
    end_date: Optional[date] = Query(None, description="Custom range end (inclusive)"),
    trailing_days: int = Query(
        1,
        ge=1,
        le=30,
        description="Trailing window (days) for net_sentiment when using net_sentiment metric",
    ),
    data_source: Optional[str] = Query(None),
    min_mentions_per_day: int = Query(1, ge=1, le=500),
    align_mode: str = Query("same_day"),
    market_adjustment: str = Query("none"),
) -> GrangerCausalityResponse:
    """
    Test Granger causality between sentiment and price returns.

    Tests whether past sentiment helps predict future price (and vice versa).
    """
    start_dt, end_dt = _optional_range_datetimes(start_date, end_date)
    cache_key = (
        f"granger:{ticker.upper()}:{max_lag}:{period}:{sentiment_metric}:td{trailing_days}"
        f"{_range_cache_suffix(start_date, end_date)}"
        f"{_methodology_cache_suffix(data_source=data_source, min_mentions_per_day=min_mentions_per_day, align_mode=align_mode, market_adjustment=market_adjustment)}"
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
            trailing_days=trailing_days,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=align_mode,
            market_adjustment=market_adjustment,
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
    sentiment_metric: str = Query(
        "net_sentiment", description="Sentiment metric to use"
    ),
    price_metric: Optional[str] = Query(
        None, description="Omit for auto based on align_mode and market_adjustment"
    ),
    start_date: Optional[date] = Query(None, description="Custom range start (inclusive)"),
    end_date: Optional[date] = Query(None, description="Custom range end (inclusive)"),
    trailing_days: int = Query(
        1,
        ge=1,
        le=30,
        description="Trailing window (days) for net_sentiment when using net_sentiment metric",
    ),
    data_source: Optional[str] = Query(None),
    min_mentions_per_day: int = Query(1, ge=1, le=500),
    align_mode: str = Query("same_day"),
    market_adjustment: str = Query("none"),
) -> RollingCorrelationResponse:
    """
    Calculate rolling windowed correlation between sentiment and price.

    Shows how the correlation changes over time using a sliding window.
    """
    start_dt, end_dt = _optional_range_datetimes(start_date, end_date)
    cache_key = (
        f"rolling:{ticker.upper()}:{window}:{period}:{sentiment_metric}:td{trailing_days}"
        f"{_range_cache_suffix(start_date, end_date)}"
        f"{_methodology_cache_suffix(data_source=data_source, min_mentions_per_day=min_mentions_per_day, align_mode=align_mode, market_adjustment=market_adjustment, price_metric=price_metric)}"
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
            trailing_days=trailing_days,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=align_mode,
            market_adjustment=market_adjustment,
        )
        response = RollingCorrelationResponse(**result)
        _rolling_cache[cache_key] = response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overview/all", response_model=CorrelationOverviewResponse)
async def get_correlation_overview(
    min_mentions: int = Query(3, ge=1, description="Minimum mentions required (ticker-level)"),
    period: str = Query("90d", description="Price data period"),
    data_source: Optional[str] = Query(None),
    min_mentions_per_day: int = Query(1, ge=1, le=500),
    align_mode: str = Query("same_day"),
    market_adjustment: str = Query("none"),
) -> CorrelationOverviewResponse:
    """
    Correlation scan across tickers with Bonferroni-adjusted significance flags.
    """
    cache_key = (
        f"overview:{min_mentions}:{period}"
        f"{_methodology_cache_suffix(data_source=data_source, min_mentions_per_day=min_mentions_per_day, align_mode=align_mode, market_adjustment=market_adjustment)}"
    )
    cached = _overview_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = get_analyzer().get_correlation_overview(
            min_mentions=min_mentions,
            period=period,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=align_mode,
            market_adjustment=market_adjustment,
        )
        items = [CorrelationOverviewItem(**r) for r in raw["items"]]
        response = CorrelationOverviewResponse(
            n_tickers_tested=raw["n_tickers_tested"],
            alpha_individual=raw["alpha_individual"],
            alpha_bonferroni=raw["alpha_bonferroni"],
            align_mode=raw["align_mode"],
            market_adjustment=raw["market_adjustment"],
            data_source=raw["data_source"],
            items=items,
        )
        _overview_cache[cache_key] = response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/out-of-sample", response_model=OutOfSampleResponse)
async def get_out_of_sample_correlation(
    ticker: str,
    period: str = Query("90d"),
    sentiment_metric: str = Query("net_sentiment"),
    price_metric: Optional[str] = Query(None),
    train_ratio: float = Query(0.7, gt=0.05, lt=0.95),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    trailing_days: int = Query(1, ge=1, le=30),
    data_source: Optional[str] = Query(None),
    min_mentions_per_day: int = Query(1, ge=1, le=500),
    align_mode: str = Query("same_day"),
    market_adjustment: str = Query("none"),
) -> OutOfSampleResponse:
    """Early vs late window Pearson correlation (simple holdout check)."""
    start_dt, end_dt = _optional_range_datetimes(start_date, end_date)
    cache_key = (
        f"oos:{ticker.upper()}:{period}:tr{train_ratio}:td{trailing_days}"
        f"{_range_cache_suffix(start_date, end_date)}"
        f"{_methodology_cache_suffix(data_source=data_source, min_mentions_per_day=min_mentions_per_day, align_mode=align_mode, market_adjustment=market_adjustment, price_metric=price_metric)}"
    )
    cached = _oos_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw = get_analyzer().out_of_sample_correlation(
            ticker=ticker.upper(),
            period=period,
            sentiment_metric=sentiment_metric,
            price_metric=price_metric,
            start_date=start_dt,
            end_date=end_dt,
            trailing_days=trailing_days,
            train_ratio=train_ratio,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=align_mode,
            market_adjustment=market_adjustment,
        )
        train_b = raw.get("train")
        test_b = raw.get("test")
        response = OutOfSampleResponse(
            ticker=raw["ticker"],
            error=raw.get("error"),
            train_ratio=raw.get("train_ratio"),
            split_date=raw.get("split_date"),
            effective_price_metric=raw.get("effective_price_metric"),
            sentiment_metric=raw.get("sentiment_metric"),
            align_mode=raw.get("align_mode"),
            market_adjustment=raw.get("market_adjustment"),
            spy_beta=raw.get("spy_beta"),
            train=OutOfSampleBlock(**train_b) if isinstance(train_b, dict) else None,
            test=OutOfSampleBlock(**test_b) if isinstance(test_b, dict) else None,
            trailing_days=raw.get("trailing_days"),
        )
        _oos_cache[cache_key] = response
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
