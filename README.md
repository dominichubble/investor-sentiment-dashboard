# Investor Sentiment Dashboard

### BSc Computer Science — Final Year Project

**Author:** Dominic Hubble (F319895)
**Supervisor:** Professor Stephen Lynch
**Department:** Computer Science, Loughborough University
**Academic Year:** 2025–2026

---

## Project Overview

The **Investor Sentiment Dashboard** is an end-to-end analytics system that uses NLP and machine learning to analyse public sentiment around financial assets (stocks, ETFs, cryptocurrencies). It ingests text from Reddit, X/Twitter, and financial news, classifies sentiment using **FinBERT**, provides **explainable AI** (LIME) for transparency, and visualises insights through an interactive React dashboard backed by a FastAPI API.

Key capabilities:

- **Sentiment classification** — FinBERT achieves 82% accuracy / 80.3% macro F1 on a 200-sample labeled benchmark (vs 68.5% / 67.9% for a keyword baseline).
- **Sentiment–price correlation** — Pearson/Spearman correlation, lag analysis, Granger causality, and rolling correlation for any tracked stock with custom date range filtering.
- **Explainable AI** — LIME token-level explanations integrated into the dashboard and saved as evaluation artifacts.
- **150k+ sentiment records** from Reddit, NewsAPI, and a historical tweet dataset (Sep 2021 – Sep 2022).

---

## Quick Start

### Prerequisites

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- A **Neon** (or other PostgreSQL) database and `DATABASE_URL` in repo-root `.env` (see `.env.example`)
- A terminal (PowerShell, bash, etc.)

### 1. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Check health: `GET http://localhost:8000/health`.

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The dashboard will be available at `http://localhost:5173`.

---

## Project Structure

```
investor-sentiment-dashboard/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point
│   │   ├── api/v1/                  # API routers (data, sentiment, correlation)
│   │   ├── analysis/                # CorrelationAnalyzer, PriceService
│   │   ├── evaluation/              # Benchmark, metrics, labeled dataset
│   │   ├── explainability/          # LIME & SHAP explainers, visualizations
│   │   ├── models/                  # FinBERT model wrapper, keyword baseline
│   │   ├── pipelines/               # Reddit, Twitter, News ingestion
│   │   ├── services/                # Import service, statistics service
│   │   └── storage/                 # PostgreSQL (Neon) via SQLAlchemy
│   ├── api/routers/                 # Correlation router (legacy path)
│   ├── scripts/                     # import_tweets.py, generate_lime_examples.py
│   ├── data/evaluation/             # Benchmark results (JSON)
│   ├── tests/                       # Pytest test suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/                   # Homepage, StockAnalysis (correlation)
│   │   ├── components/              # Charts, Navbar, MetricCard, etc.
│   │   ├── context/                 # DashboardContext (data fetching)
│   │   ├── services/api.ts          # API client
│   │   └── types/                   # TypeScript interfaces
│   └── package.json
├── data/
│   ├── db/                          # Local DB dir (optional; Neon is primary store)
│   └── processed/explanations/      # LIME example outputs (PNG + HTML)
├── notebooks/                       # Jupyter notebooks for exploration
└── CONTEXT.MD                       # Detailed project context and progress log
```

---

## Evaluation Artifacts

### Quantitative — FinBERT vs Keyword Benchmark

Results in `backend/data/evaluation/benchmark_results.json`, generated from a curated 200-sample labeled dataset (70 positive, 70 negative, 60 neutral).

| Metric          | Keyword Baseline | FinBERT  | Delta    |
|-----------------|-----------------|----------|----------|
| Accuracy        | 68.5%           | **82.0%**| +13.5%   |
| Macro F1        | 67.9%           | **80.3%**| +12.4%   |
| Macro Precision | 70.5%           | **85.1%**| +14.6%   |
| Macro Recall    | 69.1%           | **80.6%**| +11.5%   |

To regenerate: `cd backend && python -m app.evaluation.benchmark --all --output data/evaluation/benchmark_results.json`

### Qualitative — LIME Explanation Examples

10 LIME explanation visualizations in `data/processed/explanations/lime_examples/` (PNG bar charts + interactive HTML files), covering positive, negative, and neutral predictions with token-level feature weights.

To regenerate: `python backend/scripts/generate_lime_examples.py`

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

---

## Tech Stack

| Layer     | Technology                                      |
|-----------|-------------------------------------------------|
| Frontend  | React 18, TypeScript, Vite, Recharts            |
| Backend   | FastAPI, Uvicorn, SQLAlchemy, PostgreSQL (Neon)  |
| ML Model  | FinBERT (ProsusAI/finbert via HuggingFace)       |
| XAI       | LIME (lime-text), Matplotlib                     |
| Data      | Reddit (PRAW), NewsAPI, yfinance, historical CSV |

---

## Objectives

- Aggregate sentiment data from **Reddit**, **X (Twitter)**, and **financial news APIs**.
- Apply **FinBERT** for financial-domain sentiment classification (positive / negative / neutral).
- Integrate **Explainable AI** tools such as **LIME** to interpret model predictions.
- Build an interactive, web-based dashboard using **React** (frontend) and **FastAPI** (backend).
- Evaluate accuracy, interpretability, and usability of the system.
