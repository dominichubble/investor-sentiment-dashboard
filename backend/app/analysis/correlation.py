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
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def _adf_stationarity(series: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Augmented Dickey-Fuller stationarity pre-check.

    Returns a dict with the ADF test statistic, p-value, and a boolean
    ``stationary`` flag (True when the null of a unit root can be rejected
    at the supplied alpha). If statsmodels is unavailable or the series is
    too short to test, returns ``{"available": False}``; callers should treat
    that as "no pre-check applied" rather than as an assertion of stationarity.
    """
    if not HAS_STATSMODELS:
        return {"available": False, "reason": "statsmodels not installed"}
    clean = series[~np.isnan(series)]
    if clean.size < 12 or np.nanstd(clean) == 0.0:
        return {"available": False, "reason": "series too short or constant"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value, _, _, _, _ = adfuller(clean, autolag="AIC")
    except Exception as exc:
        return {"available": False, "reason": f"adfuller failed: {exc}"}
    return {
        "available": True,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "stationary": bool(p_value < alpha),
        "alpha": alpha,
    }

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


_VALID_ALIGN = frozenset({"same_day", "sentiment_leads_1d"})
_VALID_MARKET = frozenset({"none", "spy_beta_residual"})


def _norm_align(align_mode: str) -> str:
    k = (align_mode or "same_day").strip().lower()
    if k not in _VALID_ALIGN:
        raise ValueError(
            f"align_mode must be one of {sorted(_VALID_ALIGN)}, got {align_mode!r}"
        )
    return k


def _norm_market(market_adjustment: str) -> str:
    k = (market_adjustment or "none").strip().lower()
    if k not in _VALID_MARKET:
        raise ValueError(
            f"market_adjustment must be one of {sorted(_VALID_MARKET)}, "
            f"got {market_adjustment!r}"
        )
    return k


def _ols_beta_y_on_x(y: np.ndarray, x: np.ndarray) -> float:
    """Scalar beta in y ≈ beta * x (through origin); fallback 1.0 if degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vx = np.var(x)
    if vx < 1e-16 or len(x) < 2:
        return 1.0
    return float(np.cov(y, x, bias=True)[0, 1] / vx)


def _default_price_column(align_mode: str, market_adjustment: str) -> str:
    if align_mode == "sentiment_leads_1d":
        return (
            "forward_excess_return"
            if market_adjustment == "spy_beta_residual"
            else "forward_1d_return"
        )
    return "excess_returns" if market_adjustment == "spy_beta_residual" else "returns"


def _resolve_price_metric(
    price_metric: Optional[str],
    align_mode: str,
    market_adjustment: str,
) -> str:
    """Pick DataFrame column for price side of correlation."""
    pm = (price_metric or "").strip().lower()
    if pm in ("", "auto"):
        return _default_price_column(align_mode, market_adjustment)
    # Plain "returns" with next-day alignment means forward return, not same-day.
    if pm == "returns" and align_mode == "sentiment_leads_1d":
        return (
            "forward_excess_return"
            if market_adjustment == "spy_beta_residual"
            else "forward_1d_return"
        )
    if pm == "excess_returns" and align_mode == "sentiment_leads_1d":
        return (
            "forward_excess_return"
            if market_adjustment == "spy_beta_residual"
            else "forward_1d_return"
        )
    allowed = {
        "returns",
        "close",
        "forward_1d_return",
        "excess_returns",
        "forward_excess_return",
    }
    if pm not in allowed:
        raise ValueError(
            f"price_metric must be one of {sorted(allowed)} or auto, got {price_metric!r}"
        )
    return pm


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
        data_source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Aggregate sentiment records into daily averages.

        Returns:
            DataFrame with columns: date, avg_score, mention_count,
            positive_ratio, negative_ratio, neutral_ratio, net_sentiment.
        """
        records = self.storage.get_stock_sentiment(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            data_source=data_source,
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
        data_source: Optional[str] = None,
        min_mentions_per_day: int = 1,
        align_mode: str = "same_day",
        market_adjustment: str = "none",
    ) -> pd.DataFrame:
        """
        Merge sentiment data with price data on a daily basis.

        Adds ``trailing_net_sentiment``: causal rolling mean of ``net_sentiment``
        over ``trailing_days`` (min_periods=1). For ``net_sentiment`` correlations,
        use this column via ``_sentiment_column_for_correlation``.

        When ``market_adjustment`` is ``spy_beta_residual``, fetches SPY and sets
        ``excess_returns`` / ``forward_excess_return`` using a single OLS beta
        (stock return on SPY return) over the merged window.

        ``sentiment_leads_1d`` adds ``forward_1d_return`` (close-to-close return
        realized on the **next** trading row) for causal alignment.

        Returns:
            DataFrame with price/sentiment columns; metadata in ``.attrs``.
        """
        am = _norm_align(align_mode)
        mm = _norm_market(market_adjustment)

        sentiment_df = self._aggregate_daily_sentiment(
            ticker, start_date, end_date, data_source=data_source
        )
        price_df = PriceService.get_daily_returns(
            ticker, start_date=start_date, end_date=end_date, period=period
        )

        if sentiment_df.empty or price_df.empty:
            return pd.DataFrame()

        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.normalize()
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()

        mnp = max(1, int(min_mentions_per_day))
        if mnp > 1:
            sentiment_df = sentiment_df[sentiment_df["mention_count"] >= mnp]

        if sentiment_df.empty:
            return pd.DataFrame()

        merged = pd.merge(sentiment_df, price_df, on="date", how="inner")
        merged = merged.sort_values("date").reset_index(drop=True)

        w = _trailing_window_days(trailing_days)
        merged["trailing_net_sentiment"] = (
            merged["net_sentiment"].rolling(window=w, min_periods=1).mean()
        )

        spy_beta: Optional[float] = None
        if mm == "spy_beta_residual":
            spy_df = PriceService.get_daily_returns(
                "SPY",
                start_date=start_date,
                end_date=end_date,
                period=period,
            )
            if spy_df.empty:
                merged["spy_returns"] = np.nan
                merged["excess_returns"] = merged["returns"]
            else:
                spy_df["date"] = pd.to_datetime(spy_df["date"]).dt.normalize()
                spy_df = spy_df.rename(columns={"returns": "spy_returns"})
                merged = merged.merge(
                    spy_df[["date", "spy_returns"]], on="date", how="inner"
                )
                valid = merged["returns"].notna() & merged["spy_returns"].notna()
                if valid.sum() >= 2:
                    spy_beta = _ols_beta_y_on_x(
                        merged.loc[valid, "returns"].values,
                        merged.loc[valid, "spy_returns"].values,
                    )
                else:
                    spy_beta = 1.0
                merged["excess_returns"] = merged["returns"] - spy_beta * merged[
                    "spy_returns"
                ]
        else:
            merged["spy_returns"] = np.nan
            merged["excess_returns"] = merged["returns"]

        merged["forward_1d_return"] = merged["returns"].shift(-1)
        if mm == "spy_beta_residual" and spy_beta is not None:
            fwd_spy = merged["spy_returns"].shift(-1)
            merged["forward_excess_return"] = merged["forward_1d_return"] - spy_beta * fwd_spy
        else:
            merged["forward_excess_return"] = merged["forward_1d_return"]

        merged.attrs["align_mode"] = am
        merged.attrs["market_adjustment"] = mm
        merged.attrs["spy_beta"] = spy_beta
        merged.attrs["min_mentions_per_day"] = mnp
        merged.attrs["data_source"] = data_source

        return merged

    def calculate_correlation(
        self,
        ticker: str,
        period: str = "90d",
        sentiment_metric: str = "net_sentiment",
        price_metric: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trailing_days: int = 1,
        data_source: Optional[str] = None,
        min_mentions_per_day: int = 1,
        align_mode: str = "same_day",
        market_adjustment: str = "none",
    ) -> Dict:
        """
        Calculate Pearson and Spearman correlations between sentiment and price.

        Use ``price_metric='auto'`` or omit (via API) to pick a column from
        ``align_mode`` and ``market_adjustment`` (e.g. next-day return, SPY residual).
        """
        w = _trailing_window_days(trailing_days)
        am = _norm_align(align_mode)
        mm = _norm_market(market_adjustment)
        eff_price = _resolve_price_metric(price_metric, am, mm)

        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=am,
            market_adjustment=mm,
        )

        if merged.empty or len(merged) < 5:
            return {
                "ticker": ticker,
                "data_points": len(merged) if not merged.empty else 0,
                "error": "Insufficient data for correlation analysis (need at least 5 overlapping days)",
                "pearson": None,
                "spearman": None,
                "trailing_days": w,
                "align_mode": am,
                "market_adjustment": mm,
                "effective_price_metric": eff_price,
                "spy_beta": (merged.attrs.get("spy_beta") if not merged.empty else None),
                "data_source": data_source,
                "min_mentions_per_day": max(1, int(min_mentions_per_day)),
            }

        sent_col = _sentiment_column_for_correlation(sentiment_metric)
        work = merged.dropna(subset=[sent_col, eff_price]).copy()

        sentiment_values = work[sent_col].values
        price_values = work[eff_price].values

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
                "align_mode": am,
                "market_adjustment": mm,
                "effective_price_metric": eff_price,
                "spy_beta": merged.attrs.get("spy_beta"),
                "data_source": data_source,
                "min_mentions_per_day": max(1, int(min_mentions_per_day)),
            }

        if np.std(sentiment_clean) == 0 or np.std(price_clean) == 0:
            return {
                "ticker": ticker,
                "data_points": int(len(sentiment_clean)),
                "error": "One or both variables are constant (no variance)",
                "pearson": None,
                "spearman": None,
                "trailing_days": w,
                "align_mode": am,
                "market_adjustment": mm,
                "effective_price_metric": eff_price,
                "spy_beta": merged.attrs.get("spy_beta"),
                "data_source": data_source,
                "min_mentions_per_day": max(1, int(min_mentions_per_day)),
            }

        pearson_r, pearson_p = stats.pearsonr(sentiment_clean, price_clean)
        spearman_r, spearman_p = stats.spearmanr(sentiment_clean, price_clean)

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
            "price_metric": (price_metric or "auto"),
            "effective_price_metric": eff_price,
            "align_mode": am,
            "market_adjustment": mm,
            "spy_beta": merged.attrs.get("spy_beta"),
            "data_source": data_source,
            "min_mentions_per_day": max(1, int(min_mentions_per_day)),
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
        data_source: Optional[str] = None,
        min_mentions_per_day: int = 1,
        align_mode: str = "same_day",
        market_adjustment: str = "none",
    ) -> Dict:
        """
        Analyze correlation at different lag periods.

        Tests whether sentiment at time t predicts price movement at time t+lag.
        Positive lag = sentiment leads price. Negative lag = price leads sentiment.

        Returns:
            Dictionary with lag correlation results.
        """
        w = _trailing_window_days(trailing_days)
        am = _norm_align(align_mode)
        mm = _norm_market(market_adjustment)
        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=am,
            market_adjustment=mm,
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
        # Same-day raw returns for lag sweep (interpretable lead/lag structure).
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
        data_source: Optional[str] = None,
        min_mentions_per_day: int = 1,
        align_mode: str = "same_day",
        market_adjustment: str = "none",
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
        am = _norm_align(align_mode)
        mm = _norm_market(market_adjustment)
        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=am,
            market_adjustment=mm,
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

        # Augmented Dickey-Fuller pre-check on both input series. Granger tests
        # assume (weakly) stationary inputs; a unit root in either series can
        # inflate the F statistic and produce spurious rejections. If either
        # series fails the ADF test at alpha = 0.05, apply first-differencing
        # to the offending series and re-test, then run Granger on the
        # differenced series. The raw ADF verdicts are returned in the response
        # so that users can see which series was transformed.
        adf_sentiment = _adf_stationarity(sentiment_clean)
        adf_price = _adf_stationarity(price_clean)
        stationarity_note = "both series stationary at alpha=0.05"
        transforms: List[str] = []

        if adf_sentiment.get("available") and not adf_sentiment.get("stationary", True):
            sentiment_clean = np.diff(sentiment_clean)
            transforms.append("sentiment_first_difference")
        if adf_price.get("available") and not adf_price.get("stationary", True):
            price_clean = np.diff(price_clean)
            transforms.append("price_first_difference")

        # Realign lengths after any differencing (np.diff drops one observation).
        if transforms:
            n = min(len(sentiment_clean), len(price_clean))
            sentiment_clean = sentiment_clean[-n:]
            price_clean = price_clean[-n:]
            stationarity_note = (
                f"non-stationary input detected; applied {', '.join(transforms)} "
                "before Granger F-test"
            )
            if n < max_lag + 10:
                return {
                    "ticker": ticker,
                    "error": (
                        "Insufficient data after stationarity differencing "
                        f"(need {max_lag + 10}, have {n})"
                    ),
                    "trailing_days": w,
                    "stationarity": {
                        "sentiment": adf_sentiment,
                        "price": adf_price,
                        "transforms_applied": transforms,
                    },
                }

        results: Dict = {
            "ticker": ticker,
            "max_lag": max_lag,
            "data_points": int(len(sentiment_clean)),
            "trailing_days": w,
            "sentiment_to_price": [],
            "price_to_sentiment": [],
            "summary": {},
            "stationarity": {
                "sentiment": adf_sentiment,
                "price": adf_price,
                "transforms_applied": transforms,
                "note": stationarity_note,
            },
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
        price_metric: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trailing_days: int = 1,
        data_source: Optional[str] = None,
        min_mentions_per_day: int = 1,
        align_mode: str = "same_day",
        market_adjustment: str = "none",
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
        am = _norm_align(align_mode)
        mm = _norm_market(market_adjustment)
        eff_price = _resolve_price_metric(price_metric, am, mm)
        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=am,
            market_adjustment=mm,
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
        work = merged.dropna(subset=[sent_col, eff_price])
        if len(work) < window + 2:
            return {
                "ticker": ticker,
                "window": window,
                "data_points": 0,
                "error": f"Insufficient data after NaN drop on {eff_price}",
                "series": [],
                "trailing_days": w,
            }

        sent_series = work[sent_col]
        price_series = work[eff_price]

        rolling_corr = sent_series.rolling(window=window).corr(price_series)

        period_label = (
            "custom"
            if start_date is not None and end_date is not None
            else period
        )

        series = []
        for i, (_, row) in enumerate(work.iterrows()):
            corr_val = rolling_corr.iloc[i]
            if pd.notna(corr_val):
                series.append(
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "correlation": round(float(corr_val), 4),
                        "window_start": work.iloc[max(0, i - window + 1)][
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
            "effective_price_metric": eff_price,
        }

    def get_correlation_overview(
        self,
        min_mentions: int = 3,
        period: str = "90d",
        data_source: Optional[str] = None,
        min_mentions_per_day: int = 1,
        align_mode: str = "same_day",
        market_adjustment: str = "none",
    ) -> Dict:
        """
        Calculate correlation for all tracked stocks with sufficient data.

        Returns a dict with ``items`` (sorted by |Pearson r|) and Bonferroni
        metadata for multiple testing across tickers.
        """
        all_sentiments = self.storage.get_all_sentiments()

        ticker_counts: Dict[str, int] = defaultdict(int)
        for record in all_sentiments:
            sym = record.get("ticker", "")
            if not sym:
                continue
            if data_source and record.get("data_source") != data_source:
                continue
            ticker_counts[sym] += 1

        qualified_tickers = [t for t, c in ticker_counts.items() if c >= min_mentions]

        results: List[Dict] = []
        for sym in qualified_tickers:
            try:
                corr = self.calculate_correlation(
                    sym,
                    period=period,
                    sentiment_metric="net_sentiment",
                    data_source=data_source,
                    min_mentions_per_day=min_mentions_per_day,
                    align_mode=align_mode,
                    market_adjustment=market_adjustment,
                )
                if corr.get("pearson") is not None:
                    results.append(
                        {
                            "ticker": sym,
                            "mentions": ticker_counts[sym],
                            "data_points": corr["data_points"],
                            "pearson_r": corr["pearson"]["coefficient"],
                            "pearson_p": corr["pearson"]["p_value"],
                            "significant": corr["pearson"]["significant"],
                            "interpretation": corr["pearson"]["interpretation"],
                            "spearman_r": corr["spearman"]["coefficient"],
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed correlation for {sym}: {e}")
                continue

        results.sort(key=lambda x: abs(x.get("pearson_r", 0)), reverse=True)
        n = len(results)
        alpha_b = (0.05 / n) if n else None
        for row in results:
            row["significant_bonferroni"] = bool(
                alpha_b is not None and row["pearson_p"] < alpha_b
            )

        return {
            "n_tickers_tested": n,
            "alpha_individual": 0.05,
            "alpha_bonferroni": round(float(alpha_b), 8) if alpha_b is not None else None,
            "align_mode": _norm_align(align_mode),
            "market_adjustment": _norm_market(market_adjustment),
            "data_source": data_source,
            "items": results,
        }

    def out_of_sample_correlation(
        self,
        ticker: str,
        period: str = "90d",
        sentiment_metric: str = "net_sentiment",
        price_metric: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trailing_days: int = 1,
        train_ratio: float = 0.7,
        data_source: Optional[str] = None,
        min_mentions_per_day: int = 1,
        align_mode: str = "same_day",
        market_adjustment: str = "none",
    ) -> Dict:
        """
        Split the merged window into early (train) and late (holdout) segments
        and report Pearson r / p on each using the same effective price column.
        """
        w = _trailing_window_days(trailing_days)
        am = _norm_align(align_mode)
        mm = _norm_market(market_adjustment)
        eff_price = _resolve_price_metric(price_metric, am, mm)
        tr = float(train_ratio)
        if tr <= 0.05 or tr >= 0.95:
            return {
                "ticker": ticker,
                "error": "train_ratio must be between 0.05 and 0.95 (exclusive).",
            }

        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=am,
            market_adjustment=mm,
        )
        if merged.empty:
            return {"ticker": ticker, "error": "No merged data for out-of-sample split."}

        sent_col = _sentiment_column_for_correlation(sentiment_metric)
        work = merged.dropna(subset=[sent_col, eff_price]).reset_index(drop=True)
        n = len(work)
        split_i = int(n * tr)
        if split_i < 5 or n - split_i < 5:
            return {
                "ticker": ticker,
                "error": f"Need at least 5 points in train and test; have n={n}, split={split_i}.",
                "data_points": n,
            }

        train = work.iloc[:split_i]
        test = work.iloc[split_i:]

        def _pearson_block(
            label: str, a: pd.DataFrame
        ) -> Dict[str, object]:
            xs = a[sent_col].values
            ys = a[eff_price].values
            m = ~(np.isnan(xs) | np.isnan(ys))
            xs, ys = xs[m], ys[m]
            if len(xs) < 5 or np.std(xs) == 0 or np.std(ys) == 0:
                return {
                    "label": label,
                    "n": int(len(xs)),
                    "pearson_r": None,
                    "pearson_p": None,
                    "significant": False,
                }
            r, p = stats.pearsonr(xs, ys)
            return {
                "label": label,
                "n": int(len(xs)),
                "pearson_r": round(float(r), 4),
                "pearson_p": round(float(p), 6),
                "significant": bool(p < 0.05),
            }

        split_date = test.iloc[0]["date"]
        split_s = split_date.strftime("%Y-%m-%d") if hasattr(split_date, "strftime") else str(split_date)

        return {
            "ticker": ticker,
            "train_ratio": round(tr, 4),
            "split_date": split_s,
            "effective_price_metric": eff_price,
            "sentiment_metric": sentiment_metric,
            "align_mode": am,
            "market_adjustment": mm,
            "spy_beta": merged.attrs.get("spy_beta"),
            "train": _pearson_block("train", train),
            "test": _pearson_block("test", test),
            "trailing_days": w,
        }

    def get_timeseries_response(
        self,
        ticker: str,
        period: str = "90d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trailing_days: int = 1,
        data_source: Optional[str] = None,
        min_mentions_per_day: int = 1,
        align_mode: str = "same_day",
        market_adjustment: str = "none",
    ) -> Dict:
        """
        Get formatted time-series data for API response.

        Returns dictionary suitable for JSON serialization and frontend charting.
        """
        w = _trailing_window_days(trailing_days)
        am = _norm_align(align_mode)
        mm = _norm_market(market_adjustment)
        merged = self.get_merged_timeseries(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
            trailing_days=w,
            data_source=data_source,
            min_mentions_per_day=min_mentions_per_day,
            align_mode=am,
            market_adjustment=mm,
        )

        if merged.empty:
            return {
                "ticker": ticker,
                "data_points": 0,
                "series": [],
                "trailing_days": w,
                "spy_beta": None,
                "align_mode": am,
                "market_adjustment": mm,
            }

        def _fcell(v: object) -> Optional[float]:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return round(float(v), 6)

        series = []
        for _, row in merged.iterrows():
            series.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "close": round(float(row["close"]), 2),
                    "returns": _fcell(row["returns"]),
                    "spy_returns": _fcell(row.get("spy_returns")),
                    "excess_returns": _fcell(row.get("excess_returns")),
                    "forward_1d_return": _fcell(row.get("forward_1d_return")),
                    "forward_excess_return": _fcell(row.get("forward_excess_return")),
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
            "spy_beta": merged.attrs.get("spy_beta"),
            "align_mode": am,
            "market_adjustment": mm,
        }
