# Investor Sentiment Dashboard

### BSc Computer Science — Final Year Project (COC251)

**Author:** Dominic Hubble  
**Student ID:** F319859  
**Supervisor:** Professor Stephen Lynch  
**Department:** Computer Science, Loughborough University  
**Academic year:** 2025–2026  
**Dissertation sources:** `Final_Year_Project_Report/` (LaTeX, April 2026)

---

## Project overview

The **Investor Sentiment Dashboard** is an end-to-end research artefact that combines **natural language processing (NLP)**, **explainable AI (XAI)**, and **market data** to study public investor sentiment and its relationship to asset prices. Text is aggregated from **Reddit** (PRAW), **X/Twitter** (official X API v2 via Tweepy, with an optional **historical CSV** path for reproducible evaluation), and **financial news** (NewsAPI). The core classifier is **FinBERT** (ProsusAI/finbert via Hugging Face), with a **keyword baseline** for controlled comparison.

The system is **not** a regulated financial product: correlation and Granger-style outputs are **exploratory statistics**, not trading advice or claims of causation. Limitations and scope are stated in the SPA (**`/methodology`**, **`/legal`**) and in full in the dissertation.

### Main capabilities (aligned with the dissertation)

| Area | What the codebase delivers |
|------|----------------------------|
| **Classification** | Three-way sentiment (positive / negative / neutral) on the live multi-source corpus. |
| **Novel emotion layer** | A **finance-specific emotion taxonomy** (fear, optimism, uncertainty, confidence, scepticism, mixed) layered over FinBERT using normalised Shannon entropy, domain lexicon cues, and financial-aspect priors (`backend/app/analysis/finance_emotion.py`). |
| **Traceable rationales** | A lightweight **aspect-based enrichment** path that surfaces evidence-style justification for headline sentiment (`backend/app/analysis/financial_sentiment_enrichment.py`). |
| **Explainability** | **LIME** exposed as a REST surface and rendered in the React app (token-level explanations); **SHAP** utilities exist in the backend for offline / API experimentation (`backend/app/explainability/`). |
| **Entity resolution** | **spaCy** NER plus fuzzy matching to link mentions to tickers (`backend/app/entities/`). |
| **Sentiment vs price** | **Pearson / Spearman**, **lag** views, **Granger** tests, **rolling correlation**, and related dashboard analytics (`backend/app/analysis/`, API v1 routes). |
| **Scale** | **150k+** stored sentiment records across the ingested channels (see evaluation chapter for corpus balance caveats, especially Twitter). |

### Quantitative headline results (200-sample benchmark)

On a purpose-built benchmark reflecting **Reddit, news, and social-style** text (not Financial PhraseBank alone), **FinBERT** beats the **keyword baseline** by **13.5** points accuracy and **12.4** points macro **F1**; the gap is **statistically significant** under a paired **McNemar** test (continuity-corrected χ² = 9.52, *p* = 0.002) — see the dissertation evaluation chapter. The largest single-class gain is **negative recall** (**47.1% → 94.3%**), illustrating how lexicon methods miss implicit financial phrasing.

**Labelling quality (thesis):** labels are author-annotated with an **LLM cross-pass** (reported Cohen’s κ; interpret as **internal consistency**, not classical inter-rater reliability) and a **supplementary human** pass on a **100-sample** stratified subset (Cohen’s κ = 0.895, author vs volunteer). Full detail, confusion matrices, and limitations are in `Final_Year_Project_Report/chapters/06_Evaluation.tex`.

### Correlation study (thesis, illustrative panel)

A nine-ticker demonstrative run reports same-day Pearson *r* in **[-0.10, +0.24]** with **NVDA**, **AAPL**, and **AMZN** individually significant at α = 0.05 (uncorrected) and **SPY** near zero as a sanity check; **Granger** tests on the short lags examined do **not** reach significance at α = 0.05 — reported transparently in the evaluation chapter.

---

## For examiners (dissertation / viva)

- **Written report:** `Final_Year_Project_Report/main.tex` (figures under `Final_Year_Project_Report/figures/`).
- **Methodology in the artefact:** With the frontend running, open **`/methodology`** for a concise map from thesis claims (pipeline, evaluation, limitations) to dashboard screens and repository paths.
- **Reproducibility:** `.env.example`, the **Quick Start** below, backend **`pytest`**, and the benchmark / LIME commands under **Evaluation artifacts**.

---

## Quick start

### Prerequisites

- **Python 3.11+** and pip  
- **Node.js 18+** and npm  
- A **PostgreSQL** URL (e.g. **Neon**) in a repo-root **`.env`** — see **`.env.example`**  
- Optional: API keys for Reddit, X, NewsAPI, Groq (see `.env.example` comments)

### Backend

```bash
cd backend

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

python -m uvicorn app.main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/health`.

**Lean API (small cloud instances, no on-box heavy ML):**  
`pip install -r requirements-lean.txt`, set **`LEAN_API=1`**, then run Uvicorn as above. You retain statistics, correlation, stock-quality, and **Groq**-backed ticker narrative endpoints; on-box **FinBERT / LIME** paths respond **503** unless you use the full `requirements.txt`. See **`backend/Dockerfile.lean`** and **`render.yaml`** for deployment hints.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Dev server: `http://localhost:5173`.

**Monorepo helper (repo root):** after `npm install` at the root, `npm run dev` can run backend and frontend together via `concurrently` — see root `package.json`. On Windows, **`./start-dev.ps1`** is documented in `QUICKSTART.md`.

---

## Repository layout

```
investor-sentiment-dashboard/
├── Final_Year_Project_Report/   # Dissertation LaTeX, figures, references.bib
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entry
│   │   ├── api/v1/                 # Versioned HTTP API (data, sentiment, correlation, …)
│   │   ├── analysis/               # Correlation, price service, emotion taxonomy, enrichment
│   │   ├── entities/               # spaCy + fuzzy ticker resolution
│   │   ├── evaluation/             # Benchmark runner, metrics, labelled dataset
│   │   ├── explainability/         # LIME, SHAP helpers, plots
│   │   ├── models/                 # FinBERT wrapper, keyword baseline
│   │   ├── pipelines/              # Reddit, Twitter/X, news ingestion & preprocessing
│   │   ├── preprocessing/        # FinBERT tokenization, text utilities
│   │   ├── services/               # Import, statistics, narrative, windows
│   │   └── storage/                # SQLAlchemy / PostgreSQL access
│   ├── scripts/                    # e.g. generate_lime_examples.py, import helpers
│   ├── tests/                      # Pytest suite
│   ├── requirements.txt            # Full stack (torch, transformers, spaCy, LIME, …)
│   └── requirements-lean.txt     # API-only / constrained deployments
├── frontend/
│   ├── src/
│   │   ├── pages/                  # Market overview, stock analysis, methodology, legal, LIME
│   │   ├── components/             # Charts, navbar, footer, brand, error boundaries
│   │   ├── context/                # Dashboard data context
│   │   ├── services/               # Axios API client
│   │   └── types/
│   └── package.json
├── data/                           # Gitignored raw/processed data; see .gitignore
├── notebooks/                      # Exploratory notebooks
├── docker-compose.yml
├── README-DEVELOPMENT.md
├── QUICKSTART.md
└── CONTEXT.MD                      # Long-form project log / context
```

---

## Evaluation artifacts

### FinBERT vs keyword (regenerate)

```bash
cd backend
python -m app.evaluation.benchmark --all --output data/evaluation/benchmark_results.json
```

Committed reference output: `backend/data/evaluation/benchmark_results.json` (if present in your clone).

| Metric          | Keyword baseline | FinBERT (thesis headline) |
|-----------------|------------------|---------------------------|
| Accuracy        | 68.5%            | **82.0%**                 |
| Macro F1        | 67.9%            | **80.3%**                 |
| Macro precision | 70.5%            | **85.1%**                 |
| Macro recall    | 69.1%            | **80.6%**                 |

### LIME examples (regenerate)

```bash
python backend/scripts/generate_lime_examples.py
```

Default gallery paths are described in the dissertation and `.gitignore` allows committed examples under `data/processed/explanations/lime_examples/` where applicable.

---

## Running tests

```bash
cd backend
python -m pytest tests/ -v
```

**Note:** the full suite is large and some cases assume optional models / spaCy resources / data fixtures are available in the environment. If you see failures in entity or tokenizer-heavy tests, compare with a clean venv created from `requirements.txt` and the dissertation’s **Evaluation** / **Implementation** notes. For a fast API smoke pass:

```bash
python -m pytest tests/api/test_bootstrap_app.py tests/api/test_v1_sentiment.py -v
```

Frontend **`npm run build`** is the primary compile check. **`npm run test:run`** / Vitest scripts require **`vitest`** (and related config) to be present in `devDependencies` if you want CI-style unit tests for the SPA.

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, TypeScript, Vite, Recharts, React Router |
| Backend | FastAPI, Uvicorn, SQLAlchemy, PostgreSQL (Neon) |
| NLP / ML | FinBERT (Hugging Face Transformers), PyTorch, optional SHAP |
| XAI | LIME (`lime`), custom visualisations |
| NLP utilities | spaCy (NER), domain lexicons / heuristics in analysis modules |
| Data | PRAW, Tweepy (X API v2), NewsAPI, yfinance, historical CSV path for tweets |
| Tooling | Black, isort, pytest (see `backend/pyproject.toml`) |

---

## Objectives (summary)

Derived from the dissertation introduction: aggregate heterogeneous sentiment; classify with **FinBERT** against a reproducible baseline; deploy the **emotion taxonomy** and **enrichment** layers at scale on the corpus; integrate **LIME** into the product surface; resolve **tickers** from free text; expose **sentiment–price** analytics (Pearson, Spearman, lag, Granger, rolling); ship a **Vite/React** SPA with **full** and **lean** backend deployment modes.

---

## Submission zip (additional material)

For a **clean source archive** (tracked files only, no `.git/`, no `node_modules`, no `.venv`, no gitignored secrets or data), from the repository root:

```bash
git archive --format=zip --prefix=F319859-COC251-2-Sourcezip/ -o F319859-COC251-2-Sourcezip.zip HEAD
```

Unpacking creates a single top-level folder **`F319859-COC251-2-Sourcezip/`**. Install dependencies as in **Quick start**, then run **`pytest`** / **`npm run build`** as above.
