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
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a stock.

        When *start_date* and *end_date* are both provided the explicit date
        window is used and *period* is ignored.  Otherwise *period* is forwarded
        to yfinance as before.
        """
        if start_date and end_date:
            cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}_{interval}"
        else:
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
            if start_date and end_date:
                # Add one day buffer to end_date so the end day is included
                end_plus = end_date + timedelta(days=1)
                df = stock.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_plus.strftime("%Y-%m-%d"),
                    interval=interval,
                )
            else:
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
    def get_daily_returns(
        cls,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "90d",
    ) -> pd.DataFrame:
        """
        Get daily returns for a stock within a date range.

        Returns:
            DataFrame with columns: date, close, returns.
        """
        df = cls.get_price_history(
            ticker, period=period, start_date=start_date, end_date=end_date
        )

        if df.empty:
            return pd.DataFrame()

        result = df[["date", "Close", "returns"]].copy()
        result = result.rename(columns={"Close": "close"})

        # When dates were already used in get_price_history, no further
        # filtering is needed, but we still apply it as a safety net.
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
