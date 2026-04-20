"""
Run the correlation engine across a representative panel of tickers and
emit a summary suitable for the evaluation-chapter LaTeX table.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/run_correlation_panel.py

Env: DATABASE_URL must be set (loaded from .env automatically by the backend).

Outputs:
    - scripts/correlation_panel_results.csv (machine-readable)
    - stdout: LaTeX-table-ready rows
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make backend importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

# Load .env from repo root
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.analysis.correlation import CorrelationAnalyzer  # noqa: E402

TICKERS = [
    "NVDA",
    "AAPL",
    "TSLA",
    "AMZN",
    "GOOGL",
    "META",
    "BTC",
    "ETH",
    "SPY",
]
PERIOD = "90d"
ROLLING_WINDOW = 14
MAX_LAG = 2


def fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_r(r: float | None) -> str:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "--"
    return f"{r:+.2f}"


def run_ticker(analyzer: CorrelationAnalyzer, ticker: str) -> dict:
    out = {"ticker": ticker}

    # Main correlation + merged series length
    try:
        corr = analyzer.calculate_correlation(ticker=ticker, period=PERIOD)
    except Exception as e:
        print(f"[{ticker}] correlation error: {e}", file=sys.stderr)
        corr = {}

    pearson = (corr or {}).get("pearson") or {}
    spearman = (corr or {}).get("spearman") or {}
    out["pearson_r"] = pearson.get("coefficient")
    out["pearson_p"] = pearson.get("p_value")
    out["spearman_r"] = spearman.get("coefficient")
    out["spearman_p"] = spearman.get("p_value")
    out["n"] = (corr or {}).get("data_points") or (corr or {}).get("n")

    # Granger (min p over lags 1..MAX_LAG, sentiment -> price direction)
    try:
        gc = analyzer.granger_causality(ticker=ticker, period=PERIOD, max_lag=MAX_LAG)
    except Exception as e:
        print(f"[{ticker}] granger error: {e}", file=sys.stderr)
        gc = {}

    sp = (gc or {}).get("sentiment_to_price") or []
    granger_p_values = [
        entry.get("p_value")
        for entry in sp
        if isinstance(entry, dict) and entry.get("p_value") is not None
    ]
    out["granger_p_min"] = min(granger_p_values) if granger_p_values else None

    # Rolling correlation
    try:
        rc = analyzer.rolling_correlation(
            ticker=ticker, period=PERIOD, window=ROLLING_WINDOW
        )
    except Exception as e:
        print(f"[{ticker}] rolling error: {e}", file=sys.stderr)
        rc = {}

    stats_block = (rc or {}).get("statistics") or {}
    out["rolling_mean"] = stats_block.get("mean_correlation")
    out["rolling_std"] = stats_block.get("std_correlation")

    return out


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 2

    analyzer = CorrelationAnalyzer()

    rows = []
    for ticker in TICKERS:
        print(f"Running {ticker}...", file=sys.stderr)
        rows.append(run_ticker(analyzer, ticker))

    df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "scripts" / "correlation_panel_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}\n", file=sys.stderr)

    # Emit LaTeX-ready rows
    print("% LaTeX table rows (paste into tab:correlation_panel):")
    for r in rows:
        rolling_cell = "--"
        if r.get("rolling_mean") is not None:
            rolling_cell = f"{r['rolling_mean']:+.2f} $\\pm$ {r['rolling_std']:.2f}"
        pearson_cell = f"{fmt_r(r.get('pearson_r'))} ({fmt_p(r.get('pearson_p'))})"
        spearman_cell = f"{fmt_r(r.get('spearman_r'))} ({fmt_p(r.get('spearman_p'))})"
        granger_cell = fmt_p(r.get("granger_p_min"))
        n_cell = r.get("n") if r.get("n") is not None else "--"
        print(
            f"    {r['ticker']:<7s} & {pearson_cell} & {spearman_cell} "
            f"& {granger_cell} & {rolling_cell} & {n_cell} \\\\"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
