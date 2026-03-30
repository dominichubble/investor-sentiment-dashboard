"""
Correlation analysis between sentiment scores and stock price movements.

Calculates Pearson and Spearman correlations, supports lag analysis,
and provides merged time-series data for visualization.
"""

import logging
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from app.storage import StockSentimentStorage

from .price_service import PriceService

try:
    from statsmodels.tsa.stattools import grangercausalitytests

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logger = logging.getLogger(__name__)

_TRAILING_MAX_DAYS = 30


def _trailing_window_days(trailing_days: Optional[int]) -> int:
    """Clamp trailing window for causal rolling mean of net_sentiment (1 = same-day only)."""
    if trailing_days is None or trailing_days < 1:
        return 1
    return min(_TRAILING_MAX_DAYS, int(trailing_days))


def _sentiment_column_for_correlation(sentiment_metric: str) -> str:
    """Map API sentiment_metric to merged DataFrame column."""
    if sentiment_metric == "net_sentiment":
        return "trailing_net_sentiment"
    return sentiment_metric


class CorrelationAnalyzer:
    """Analyzes correlation between sentiment and stock price movements."""

    def __init__(self, storage: Optional[StockSentimentStorage] = None):
        self.storage = storage or StockSentimentStorage()
        if not self.storage._loaded:
            self.storage.load()

    def _aggregate_daily_sentiment(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Aggregate sentiment records into daily averages.

        Returns:
            DataFrame with columns: date, avg_score, mention_count,
            positive_ratio, negative_ratio, neutral_ratio, net_sentiment.
        """
        records = self.storage.get_stock_sentiment(
            ticker=ticker, start_date=start_date, end_date=end_date
        )

        if not records:
            return pd.DataFrame()

        daily: Dict[str, List[Dict]] = defaultdict(list)

        for record in records:
            ts = record.get("published_at") or record.get("timestamp") or ""
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", ""))
                day_key = dt.strftime("%Y-%m-%d")
                daily[day_key].append(record)
            except (ValueError, TypeError):
                continue

        rows = []
        for day_str, day_records in sorted(daily.items()):
            scores = [r["sentiment_score"] for r in day_records]
            labels = [r["sentiment_label"] for r in day_records]
            total = len(day_records)

            positive_count = sum(1 for lbl in labels if lbl == "positive")
            negative_count = sum(1 for lbl in labels if lbl == "negative")
            neutral_count = sum(1 for lbl in labels if lbl == "neutral")

            rows.append(
                {
                    "date": pd.Timestamp(day_str),
                    "avg_score": np.mean(scores),
                    "mention_count": total,
                    "positive_ratio": positive_count / total if total else 0,
                    "negative_ratio": negative_count / total if total else 0,
                    "neutral_ratio": neutral_count / total if total else 0,
                    "net_sentiment": (
                        (positive_count - negative_count) / total if total else 0
                    ),
                }
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_merged_timeseries(
        self,
        ticker: str,
        period: str = "90d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trailing_days: int = 1,
    ) -> pd.DataFrame:
        """
        Merge sentiment data with price data on a daily basis.

        Adds ``trailing_net_sentiment``: causal rolling mean of ``net_sentiment``
        over ``trailing_days`` (min_periods=1). For ``net_sentiment`` correlations,
        use this column via ``_sentiment_column_for_correlation``.

        Returns:
            DataFrame with columns: date, close, returns, avg_score,
            mention_count, net_sentiment, trailing_net_sentiment,
            positive_ratio, negative_ratio, neutral_ratio.
        """
        sentiment_df = self._aggregate_daily_sentiment(ticker, start_date, end_date)
        price_df = PriceService.get_daily_returns(
            ticker, start_date=start_date, end_date=end_date, period=period
        )

        if sentiment_df.empty or price_df.empty:
            return pd.DataFrame()

        # Normalize dates
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.normalize()
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()

        merged = pd.merge(sentiment_df, price_df, on="date", how="inner")
        merged = merged.sort_values("date").reset_index(drop=True)

        w = _trailing_window_days(trailing_days)
        merged["trailing_net_sentiment"] = (
            merged["net_sentiment"].rolling(window=w, min_periods=1).mean()
        )

        return merged

    def calculate_correlation(
        self,
        ticker: str,
        period: str = "90d",
        sentiment_metric: str = "net_sentiment",
        price_metric: str = "returns",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trailing_days: int = 1,
    ) -> Dict:
        """
        Calculate Pearson and Spearman correlations between sentiment and price.

        Args:
            ticker: Stock ticker symbol.
            period: Price data period (used when start_date/end_date not both set).
            sentiment_metric: Column to use for sentiment (net_sentiment, avg_score).
            price_metric: Column to use for price (returns, close).
            start_date: Inclusive range start (optional, with end_date).
            end_date: Inclusive range end (optional, with start_date).
            trailing_days: For ``net_sentiment``, rolling window length (causal, 1=same day).

        Returns:
            Dictionary with correlation results and statistical significance.
        """
        w = _trailing_window_days(trailing_days)
        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
        )

        if merged.empty or len(merged) < 5:
            return {
                "ticker": ticker,
                "data_points": len(merged) if not merged.empty else 0,
                "error": "Insufficient data for correlation analysis (need at least 5 overlapping days)",
                "pearson": None,
                "spearman": None,
                "trailing_days": w,
            }

        sent_col = _sentiment_column_for_correlation(sentiment_metric)
        sentiment_values = merged[sent_col].values
        price_values = merged[price_metric].values

        # Remove any NaN pairs
        valid_mask = ~(np.isnan(sentiment_values) | np.isnan(price_values))
        sentiment_clean = sentiment_values[valid_mask]
        price_clean = price_values[valid_mask]

        if len(sentiment_clean) < 5:
            return {
                "ticker": ticker,
                "data_points": len(sentiment_clean),
                "error": "Insufficient valid data points after cleaning",
                "pearson": None,
                "spearman": None,
                "trailing_days": w,
            }

        # Check for constant arrays (would cause warnings)
        if np.std(sentiment_clean) == 0 or np.std(price_clean) == 0:
            return {
                "ticker": ticker,
                "data_points": int(len(sentiment_clean)),
                "error": "One or both variables are constant (no variance)",
                "pearson": None,
                "spearman": None,
                "trailing_days": w,
            }

        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = stats.pearsonr(sentiment_clean, price_clean)

        # Spearman correlation (monotonic relationship)
        spearman_r, spearman_p = stats.spearmanr(sentiment_clean, price_clean)

        # Interpret strength
        def interpret_correlation(r: float) -> str:
            abs_r = abs(r)
            if abs_r >= 0.7:
                direction = "positive" if r > 0 else "negative"
                return f"Strong {direction} correlation"
            elif abs_r >= 0.4:
                direction = "positive" if r > 0 else "negative"
                return f"Moderate {direction} correlation"
            elif abs_r >= 0.2:
                direction = "positive" if r > 0 else "negative"
                return f"Weak {direction} correlation"
            else:
                return "No significant correlation"

        period_label = (
            "custom"
            if start_date is not None and end_date is not None
            else period
        )
        return {
            "ticker": ticker,
            "data_points": int(len(sentiment_clean)),
            "period": period_label,
            "sentiment_metric": sentiment_metric,
            "trailing_days": w,
            "price_metric": price_metric,
            "pearson": {
                "coefficient": round(float(pearson_r), 4),
                "p_value": round(float(pearson_p), 6),
                "significant": pearson_p < 0.05,
                "interpretation": interpret_correlation(pearson_r),
            },
            "spearman": {
                "coefficient": round(float(spearman_r), 4),
                "p_value": round(float(spearman_p), 6),
                "significant": spearman_p < 0.05,
                "interpretation": interpret_correlation(spearman_r),
            },
        }

    def lag_analysis(
        self,
        ticker: str,
        max_lag_days: int = 5,
        period: str = "90d",
        sentiment_metric: str = "net_sentiment",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trailing_days: int = 1,
    ) -> Dict:
        """
        Analyze correlation at different lag periods.

        Tests whether sentiment at time t predicts price movement at time t+lag.
        Positive lag = sentiment leads price. Negative lag = price leads sentiment.

        Returns:
            Dictionary with lag correlation results.
        """
        w = _trailing_window_days(trailing_days)
        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
        )

        if merged.empty or len(merged) < max_lag_days + 5:
            return {
                "ticker": ticker,
                "error": "Insufficient data for lag analysis",
                "lags": [],
                "trailing_days": w,
            }

        sent_col = _sentiment_column_for_correlation(sentiment_metric)
        sentiment_values = merged[sent_col].values
        price_returns = merged["returns"].values

        lag_results: List[Dict] = []

        for lag in range(-max_lag_days, max_lag_days + 1):
            if lag > 0:
                # Sentiment leads price by `lag` days
                sent = sentiment_values[:-lag]
                price = price_returns[lag:]
            elif lag < 0:
                # Price leads sentiment by `|lag|` days
                sent = sentiment_values[-lag:]
                price = price_returns[:lag]
            else:
                sent = sentiment_values
                price = price_returns

            # Clean NaN
            valid = ~(np.isnan(sent) | np.isnan(price))
            sent_clean = sent[valid]
            price_clean = price[valid]

            if (
                len(sent_clean) < 5
                or np.std(sent_clean) == 0
                or np.std(price_clean) == 0
            ):
                lag_results.append(
                    {
                        "lag_days": lag,
                        "data_points": len(sent_clean),
                        "pearson_r": None,
                        "p_value": None,
                    }
                )
                continue

            r, p = stats.pearsonr(sent_clean, price_clean)

            lag_results.append(
                {
                    "lag_days": lag,
                    "data_points": int(len(sent_clean)),
                    "pearson_r": round(float(r), 4),
                    "p_value": round(float(p), 6),
                    "significant": p < 0.05,
                    "description": (
                        f"Sentiment leads price by {lag} day(s)"
                        if lag > 0
                        else (
                            f"Price leads sentiment by {abs(lag)} day(s)"
                            if lag < 0
                            else "Same-day correlation"
                        )
                    ),
                }
            )

        # Find the optimal lag (highest absolute correlation)
        valid_lags = [l for l in lag_results if l["pearson_r"] is not None]
        best_lag = None
        if valid_lags:
            best_lag = max(valid_lags, key=lambda x: abs(x["pearson_r"]))

        return {
            "ticker": ticker,
            "max_lag_days": max_lag_days,
            "lags": lag_results,
            "best_lag": best_lag,
            "trailing_days": w,
        }

    def granger_causality(
        self,
        ticker: str,
        max_lag: int = 5,
        period: str = "90d",
        sentiment_metric: str = "net_sentiment",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trailing_days: int = 1,
    ) -> Dict:
        """
        Test Granger causality between sentiment and price returns.

        Tests both directions:
        - sentiment -> price (does past sentiment predict future returns?)
        - price -> sentiment (do past returns predict future sentiment?)

        Args:
            ticker: Stock ticker symbol.
            max_lag: Maximum number of lag days to test.
            period: Price data period (ignored when start_date/end_date given).
            sentiment_metric: Sentiment column to use.
            start_date: Optional absolute start date.
            end_date: Optional absolute end date.

        Returns:
            Dictionary with Granger causality test results for both directions.
        """
        if not HAS_STATSMODELS:
            return {
                "ticker": ticker,
                "error": "statsmodels is required for Granger causality testing. "
                "Install with: pip install statsmodels",
            }

        w = _trailing_window_days(trailing_days)
        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
        )

        if merged.empty or len(merged) < max_lag + 10:
            return {
                "ticker": ticker,
                "error": f"Insufficient data for Granger causality "
                f"(need at least {max_lag + 10} data points, have {len(merged)})",
                "trailing_days": w,
            }

        sent_col = _sentiment_column_for_correlation(sentiment_metric)
        sentiment_values = merged[sent_col].values
        price_returns = merged["returns"].values

        valid_mask = ~(np.isnan(sentiment_values) | np.isnan(price_returns))
        sentiment_clean = sentiment_values[valid_mask]
        price_clean = price_returns[valid_mask]

        if len(sentiment_clean) < max_lag + 10:
            return {
                "ticker": ticker,
                "error": "Insufficient valid data after NaN removal",
                "trailing_days": w,
            }

        results: Dict = {
            "ticker": ticker,
            "max_lag": max_lag,
            "data_points": int(len(sentiment_clean)),
            "trailing_days": w,
            "sentiment_to_price": [],
            "price_to_sentiment": [],
            "summary": {},
        }

        # Test: sentiment -> price (does sentiment Granger-cause price?)
        try:
            data_sp = np.column_stack([price_clean, sentiment_clean])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_sp = grangercausalitytests(data_sp, maxlag=max_lag, verbose=False)

            for lag in range(1, max_lag + 1):
                f_stat = gc_sp[lag][0]["ssr_ftest"][0]
                p_value = gc_sp[lag][0]["ssr_ftest"][1]
                results["sentiment_to_price"].append(
                    {
                        "lag": lag,
                        "f_statistic": round(float(f_stat), 4),
                        "p_value": round(float(p_value), 6),
                        "significant": p_value < 0.05,
                    }
                )
        except Exception as e:
            logger.warning(f"Granger test (sentiment->price) failed for {ticker}: {e}")
            results["sentiment_to_price"] = [{"error": str(e)}]

        # Test: price -> sentiment (does price Granger-cause sentiment?)
        try:
            data_ps = np.column_stack([sentiment_clean, price_clean])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_ps = grangercausalitytests(data_ps, maxlag=max_lag, verbose=False)

            for lag in range(1, max_lag + 1):
                f_stat = gc_ps[lag][0]["ssr_ftest"][0]
                p_value = gc_ps[lag][0]["ssr_ftest"][1]
                results["price_to_sentiment"].append(
                    {
                        "lag": lag,
                        "f_statistic": round(float(f_stat), 4),
                        "p_value": round(float(p_value), 6),
                        "significant": p_value < 0.05,
                    }
                )
        except Exception as e:
            logger.warning(f"Granger test (price->sentiment) failed for {ticker}: {e}")
            results["price_to_sentiment"] = [{"error": str(e)}]

        # Summary: find the best (most significant) lag for each direction
        sp_significant = [
            r
            for r in results["sentiment_to_price"]
            if isinstance(r, dict) and r.get("significant")
        ]
        ps_significant = [
            r
            for r in results["price_to_sentiment"]
            if isinstance(r, dict) and r.get("significant")
        ]

        results["summary"] = {
            "sentiment_predicts_price": len(sp_significant) > 0,
            "price_predicts_sentiment": len(ps_significant) > 0,
            "best_sentiment_to_price_lag": (
                min(sp_significant, key=lambda x: x["p_value"])
                if sp_significant
                else None
            ),
            "best_price_to_sentiment_lag": (
                min(ps_significant, key=lambda x: x["p_value"])
                if ps_significant
                else None
            ),
            "interpretation": self._interpret_granger(sp_significant, ps_significant),
        }

        return results

    @staticmethod
    def _interpret_granger(
        sp_significant: List[Dict], ps_significant: List[Dict]
    ) -> str:
        """Generate a human-readable interpretation of Granger causality results."""
        sp = len(sp_significant) > 0
        ps = len(ps_significant) > 0

        if sp and ps:
            return (
                "Bidirectional: sentiment and price returns appear to "
                "Granger-cause each other (feedback loop)."
            )
        elif sp:
            return (
                "Sentiment Granger-causes price returns: past sentiment "
                "contains information useful for predicting future price movements."
            )
        elif ps:
            return (
                "Price returns Granger-cause sentiment: past price movements "
                "appear to influence subsequent sentiment (reactive sentiment)."
            )
        else:
            return (
                "No significant Granger causality detected in either direction "
                "at the tested lag periods."
            )

    def rolling_correlation(
        self,
        ticker: str,
        window: int = 14,
        period: str = "90d",
        sentiment_metric: str = "net_sentiment",
        price_metric: str = "returns",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trailing_days: int = 1,
    ) -> Dict:
        """
        Calculate rolling windowed correlation between sentiment and price.

        Args:
            ticker: Stock ticker symbol.
            window: Rolling window size in days.
            period: Price data period (ignored when start_date/end_date given).
            sentiment_metric: Sentiment column.
            price_metric: Price column.
            start_date: Optional absolute start date.
            end_date: Optional absolute end date.

        Returns:
            Dictionary with time-series of rolling correlation values.
        """
        w = _trailing_window_days(trailing_days)
        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
        )

        if merged.empty or len(merged) < window + 2:
            return {
                "ticker": ticker,
                "window": window,
                "data_points": 0,
                "error": f"Insufficient data (need at least {window + 2} days, "
                f"have {len(merged)})",
                "series": [],
                "trailing_days": w,
            }

        sent_col = _sentiment_column_for_correlation(sentiment_metric)
        sent_series = merged[sent_col]
        price_series = merged[price_metric]

        rolling_corr = sent_series.rolling(window=window).corr(price_series)

        period_label = (
            "custom"
            if start_date is not None and end_date is not None
            else period
        )

        series = []
        for i, (_, row) in enumerate(merged.iterrows()):
            corr_val = rolling_corr.iloc[i]
            if pd.notna(corr_val):
                series.append(
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "correlation": round(float(corr_val), 4),
                        "window_start": merged.iloc[max(0, i - window + 1)][
                            "date"
                        ].strftime("%Y-%m-%d"),
                    }
                )

        # Statistics
        valid_corrs = [s["correlation"] for s in series]
        stats_summary = {}
        if valid_corrs:
            stats_summary = {
                "mean_correlation": round(float(np.mean(valid_corrs)), 4),
                "std_correlation": round(float(np.std(valid_corrs)), 4),
                "min_correlation": round(float(np.min(valid_corrs)), 4),
                "max_correlation": round(float(np.max(valid_corrs)), 4),
                "periods_positive": sum(1 for c in valid_corrs if c > 0),
                "periods_negative": sum(1 for c in valid_corrs if c < 0),
            }

        return {
            "ticker": ticker,
            "window": window,
            "period": period_label,
            "data_points": len(series),
            "series": series,
            "statistics": stats_summary,
            "trailing_days": w,
        }

    def get_correlation_overview(
        self,
        min_mentions: int = 3,
        period: str = "90d",
    ) -> List[Dict]:
        """
        Calculate correlation for all tracked stocks with sufficient data.

        Returns:
            List of correlation summaries sorted by absolute correlation strength.
        """
        all_sentiments = self.storage.get_all_sentiments()

        # Count mentions per ticker
        ticker_counts: Dict[str, int] = defaultdict(int)
        for record in all_sentiments:
            ticker = record.get("ticker", "")
            if ticker:
                ticker_counts[ticker] += 1

        # Filter tickers with enough mentions
        qualified_tickers = [t for t, c in ticker_counts.items() if c >= min_mentions]

        results = []
        for ticker in qualified_tickers:
            try:
                corr = self.calculate_correlation(
                    ticker, period=period, sentiment_metric="net_sentiment"
                )
                if corr.get("pearson") is not None:
                    results.append(
                        {
                            "ticker": ticker,
                            "mentions": ticker_counts[ticker],
                            "data_points": corr["data_points"],
                            "pearson_r": corr["pearson"]["coefficient"],
                            "pearson_p": corr["pearson"]["p_value"],
                            "significant": corr["pearson"]["significant"],
                            "interpretation": corr["pearson"]["interpretation"],
                            "spearman_r": corr["spearman"]["coefficient"],
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed correlation for {ticker}: {e}")
                continue

        # Sort by absolute correlation strength
        results.sort(key=lambda x: abs(x.get("pearson_r", 0)), reverse=True)
        return results

    def get_timeseries_response(
        self,
        ticker: str,
        period: str = "90d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trailing_days: int = 1,
    ) -> Dict:
        """
        Get formatted time-series data for API response.

        Returns dictionary suitable for JSON serialization and frontend charting.
        """
        w = _trailing_window_days(trailing_days)
        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
        )

        if merged.empty:
            return {
                "ticker": ticker,
                "data_points": 0,
                "series": [],
                "trailing_days": w,
            }

        series = []
        for _, row in merged.iterrows():
            series.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "close": round(float(row["close"]), 2),
                    "returns": (
                        round(float(row["returns"]), 6)
                        if not np.isnan(row["returns"])
                        else None
                    ),
                    "avg_sentiment_score": round(float(row["avg_score"]), 4),
                    "net_sentiment": round(float(row["net_sentiment"]), 4),
                    "trailing_net_sentiment": round(
                        float(row["trailing_net_sentiment"]), 4
                    ),
                    "mention_count": int(row["mention_count"]),
                    "positive_ratio": round(float(row["positive_ratio"]), 4),
                    "negative_ratio": round(float(row["negative_ratio"]), 4),
                    "neutral_ratio": round(float(row["neutral_ratio"]), 4),
                }
            )

        return {
            "ticker": ticker,
            "data_points": len(series),
            "series": series,
            "trailing_days": w,
        }
