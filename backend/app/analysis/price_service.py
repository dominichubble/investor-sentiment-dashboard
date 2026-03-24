"""
Stock price data service using yfinance.

Fetches historical stock prices and calculates returns for correlation analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceService:
    """Fetches and processes stock price data."""

    _cache: Dict[str, pd.DataFrame] = {}
    _cache_expiry: Dict[str, datetime] = {}
    CACHE_TTL_MINUTES = 30

    @classmethod
    def get_price_history(
        cls,
        ticker: str,
        period: str = "90d",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a stock.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").
            period: yfinance period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max).
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo).

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume, Returns.
        """
        cache_key = f"{ticker}_{period}_{interval}"
        now = datetime.now()

        if (
            cache_key in cls._cache
            and cache_key in cls._cache_expiry
            and cls._cache_expiry[cache_key] > now
        ):
            logger.debug(f"Using cached price data for {ticker}")
            return cls._cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()

            df = df.reset_index()
            df = df.rename(columns={"Date": "date"})

            # Ensure date column is timezone-naive for compatibility
            if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)

            df["date"] = pd.to_datetime(df["date"]).dt.normalize()

            # Calculate daily returns
            df["returns"] = df["Close"].pct_change()

            # Calculate simple moving averages
            df["sma_5"] = df["Close"].rolling(window=5).mean()
            df["sma_20"] = df["Close"].rolling(window=20).mean()

            # Cache the result
            cls._cache[cache_key] = df
            cls._cache_expiry[cache_key] = now + timedelta(
                minutes=cls.CACHE_TTL_MINUTES
            )

            logger.info(f"Fetched {len(df)} price records for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return pd.DataFrame()

    @classmethod
    def get_price_history_range(
        cls,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch daily prices between start_date and end_date (inclusive on both ends).

        Uses yfinance start/end; end is passed as exclusive upper bound internally.
        """
        cache_key = (
            f"{ticker}_range_{start_date.date().isoformat()}_"
            f"{end_date.date().isoformat()}_{interval}"
        )
        now = datetime.now()
        if (
            cache_key in cls._cache
            and cache_key in cls._cache_expiry
            and cls._cache_expiry[cache_key] > now
        ):
            logger.debug(f"Using cached range price data for {ticker}")
            return cls._cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            start_s = pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")
            end_exclusive = (
                pd.Timestamp(end_date).normalize() + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            df = stock.history(start=start_s, end=end_exclusive, interval=interval)

            if df.empty:
                logger.warning(f"No price data found for {ticker} in range")
                return pd.DataFrame()

            df = df.reset_index()
            df = df.rename(columns={"Date": "date"})
            if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["returns"] = df["Close"].pct_change()

            cls._cache[cache_key] = df
            cls._cache_expiry[cache_key] = now + timedelta(
                minutes=cls.CACHE_TTL_MINUTES
            )
            logger.info(f"Fetched {len(df)} price records for {ticker} (custom range)")
            return df
        except Exception as e:
            logger.error(f"Error fetching range price data for {ticker}: {e}")
            return pd.DataFrame()

    @classmethod
    def get_daily_returns(
        cls,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "90d",
    ) -> pd.DataFrame:
        """
        Get daily returns for a stock within a date range.

        When both start_date and end_date are set, fetches that window directly.
        Otherwise uses period-based history and optional start/end filters.

        Returns:
            DataFrame with columns: date, close, returns.
        """
        if start_date is not None and end_date is not None:
            df = cls.get_price_history_range(ticker, start_date, end_date)
        else:
            df = cls.get_price_history(ticker, period=period)

        if df.empty:
            return pd.DataFrame()

        result = df[["date", "Close", "returns"]].copy()
        result = result.rename(columns={"Close": "close"})

        if start_date:
            start_ts = pd.Timestamp(start_date).normalize()
            result = result[result["date"] >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date).normalize()
            result = result[result["date"] <= end_ts]

        return result.dropna(subset=["returns"])

    @classmethod
    def get_stock_info(cls, ticker: str) -> Dict:
        """Get basic stock information."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "ticker": ticker,
                "name": info.get("shortName", info.get("longName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "currency": info.get("currency", "USD"),
            }
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return {"ticker": ticker, "name": ticker}

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the price data cache."""
        cls._cache.clear()
        cls._cache_expiry.clear()
        logger.info("Price data cache cleared")
