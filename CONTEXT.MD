# Investor Sentiment Dashboard — Project Context (FYP)

## One-liner
Build an end-to-end Investor Sentiment Dashboard that ingests finance text data (Reddit, X/Twitter, News), runs financial sentiment classification using FinBERT, provides explainable AI (SHAP/LIME) for transparency, and visualises insights in a React dashboard backed by a FastAPI API.

## What “Done” Means (MVP Definition)
The MVP is complete when ALL are true:

1) Backend API exists and can be run locally:
- `uvicorn ...` starts successfully.
- Health endpoint returns OK.
- Core endpoints return correctly shaped JSON matching the frontend contract.

2) Frontend is wired to real API data:
- Homepage loads without mock data.
- Statistics cards show real counts/distributions.
- Correlation overview panel renders from real endpoint (even if limited scope).

3) Novelty delivered:
- At least ONE explainability method is integrated end-to-end:
  - An API endpoint generates an explanation for a given text (LIME or SHAP).
  - Frontend displays it (token highlights or ranked tokens).

4) Evaluation artifacts exist:
- Quantitative: at least one metrics table (accuracy/F1 or similar) on a labeled dataset or a justified proxy evaluation.
- Qualitative: at least 3 explanation examples (screenshots or saved outputs) demonstrating interpretability.

## Current State (Dev Branch)
### Implemented / Partially Implemented
- Ingestion pipelines exist:
  - `backend/app/pipelines/ingest_reddit.py`, `ingest_twitter.py`, `ingest_news.py`
- FinBERT model wrapper + inference utilities exist:
  - `backend/app/models/finbert_model.py`
  - `backend/app/models/sentiment_inference.py`
  - Tokenizer helper: `backend/app/preprocessing/finbert_tokenizer.py`
- Frontend exists (React + TS + Vite):
  - Dashboard components and Homepage exist.
  - Context/provider exists and calls API endpoints.
  - API contract is defined in `frontend/src/services/api.ts`.

### Missing / Major Gaps
- FastAPI bootstrap is implemented at `backend/app/main.py` with `/health`, `/api/v1`, and CORS for Vite.
- No runtime persistence layer (DB) or aggregation services powering stats/trending/correlation endpoints.
- No correlation analytics module (price data fetching, correlation, lag analysis, rolling windows, etc).
- Explainability (SHAP/LIME) is listed as dependencies but not implemented in code.
- Evaluation scripts/notebooks for metrics + explainability examples are not finalized.

## Key Constraints
- Timeline: ~2 months to submission.
- Prioritise end-to-end demonstrable slices over perfect architecture.
- Keep API stable to match frontend contract first, then improve internals.

## API Contract (Reference)
The backend should align with frontend expectations while allowing documented evolution.
Primary contract file:
- `frontend/src/services/api.ts`

Expected base URL:
- Default: `http://localhost:8000/api/v1`

Key endpoints used by frontend today include:
- `GET /data/statistics`
- `GET /correlation/overview/all`
…and several others already defined in `api.ts` (correlation detail, lag analysis, rolling correlation, stock info, etc).

## Repo Map (Important Paths)
### Frontend
- `frontend/src/services/api.ts` — API client + expected routes
- `frontend/src/context/DashboardContext.tsx` — data fetching orchestration
- `frontend/src/pages/Homepage/Homepage.tsx` — main dashboard UI

### Backend
- `backend/app/pipelines/*` — ingestion + preprocessing scripts
- `backend/app/models/*` — FinBERT model wrapper + inference
- `backend/app/preprocessing/*` — tokenizer/text processing utilities
- `backend/app/storage/prediction_storage.py` — file-based storage helpers (legacy/utility)

## Planned MVP Architecture
- FastAPI app:
  - `backend/app/main.py` (canonical)
- Routers:
  - `/api/v1/data/*`
  - `/api/v1/sentiment/*`
  - `/api/v1/stocks/*`
  - `/api/v1/correlation/*`
  - `/health` (non-versioned)

- Persistence:
  - Start with SQLite (fast to implement, enough for MVP)
  - Use SQLAlchemy (already in requirements)
  - Optionally migrate to Postgres later.

## Novelty (Pick & Ship)
Primary novelty (must ship):
1) Explainability integrated into the product:
- LIME or SHAP explanation endpoint + UI display on a prediction

Optional secondary novelty if time:
2) Sentiment–price correlation + lag analysis:
- Correlation overview + per-ticker analysis page

## “How to Work With Me” (for LLM assistants)
- Use `frontend/src/services/api.ts` as a strong reference, not an immutable contract.
- Implement the smallest slice that enables a demo, then iterate.
- Prefer file-by-file diffs and explicit commands.
- Include at least minimal tests for critical functions/endpoints.
- When uncertain: ask for the current API response types or paste the file.

## Confirmed Project Decisions (2026-02-23)
- Backend entrypoint choice: use `backend/app/main.py` as the canonical app entrypoint (simple and conventional for FastAPI projects).
- `frontend/src/services/api.ts` is a strong reference, but NOT strict source-of-truth. Backend can evolve where needed with documented alignment changes.
- Persistence strategy: go straight to SQLite (no temporary mock/seed-only architecture).
- Timestamp/date consistency: use UTC ISO-8601 strings consistently across API responses.
- Correlation scope: real implementations required (not stubs), as this is a final-year project with real outputs.
- Sentiment endpoint response detail: return the most practical shape (top label + score by default, extensible options for extra scores when useful).
- Explainability novelty scope: focus on LIME first; SHAP is optional stretch work.
- Evaluation artifacts: generate as part of code tasks (scripts + saved outputs in-repo), not ad-hoc manual-only outputs.
- Testing level: pytest minimum focused on critical API happy-path and key validation failures.
- Directional architecture preference: move toward DB-first storage with clear separation between ingestion, processing, storage, and frontend; avoid long-term reliance on local `data/` files for runtime behavior.

## Architecture Direction (DB-First Separation)
- Ingestion layer: fetches external data (Reddit, X/Twitter, News) and does source-level cleaning only.
- Processing layer: runs sentiment/classification logic independently of transport/storage concerns.
- Storage layer: persists canonical records in SQLite via service/storage adapters (source of truth).
- API/frontend layer: reads aggregated/queryable state from DB; does not depend on local raw files.
- Local `data/` exports are now optional artifacts for debugging/audit, not required runtime dependencies.

## Progress Log
- 2026-02-23: Initial project context and top-5 ticket plan documented in-repo.
- 2026-02-23: Requirements clarified with project owner:
  - confirmed SQLite-first implementation,
  - confirmed UTC date/time consistency,
  - confirmed real correlation implementation (no stubs),
  - confirmed LIME-first explainability focus,
  - confirmed evaluation artifacts should be generated by code tasks,
  - confirmed pytest minimum strategy.
- 2026-02-23: Ticket 1 implemented:
  - canonical app created at `backend/app/main.py`,
  - `/health` returns `{ "status": "ok" }`,
  - `/api/v1` mount established via `backend/app/api/v1/router.py`,
  - CORS allows `http://localhost:5173` and `http://127.0.0.1:5173`,
  - uvicorn startup validated and bootstrap tests pass.
- 2026-02-23: Ticket 4 implemented in canonical app path:
  - SQLite DB initialization added on startup via `backend/app/main.py`,
  - `/api/v1/data/predictions` implemented in `backend/app/api/v1/data.py`,
  - import service added at `backend/app/services/import_service.py`,
  - CLI import job added at `backend/scripts/import_to_db.py`,
  - tests added for canonical predictions endpoint and import service wiring.
- 2026-02-23: DB-first hardening applied to ingestion pipelines:
  - `ingest_reddit.py`, `ingest_news.py`, and `ingest_twitter.py` now support DB storage as default path,
  - file output is optional via CLI flag (`--write-files`) and no longer required for successful ingestion runs,
  - pipeline behavior now aligns with separation between ingestion, processing, storage, and frontend access.
- 2026-02-23: GitHub Actions daily collection workflow updated for DB-first operation:
  - `collect-data.yml` now runs ingestion with `--store-db --no-write-files`,
  - removed scheduled dependency on local raw-data artifacts and file-count reporting,
  - summary now reports DB mode and `sentiment_records` row count.
- 2026-02-23: One-time local data cleanup performed:
  - verified import script path over `data/raw` loads existing records and inserts no new rows,
  - removed legacy local raw JSON/meta artifacts after migration verification.
- 2026-02-23: Additional legacy local-storage cleanup:
  - removed legacy `data/predictions/*` artifacts,
  - removed legacy `data/stock_sentiments/stock_sentiments.json`,
  - updated ignore rules/docs to reflect DB-first source-of-truth.
- 2026-02-23: Legacy raw-file stock processing removed from runtime path:
  - `process_stock_sentiments.py` converted to DB-native enrichment (reads document rows from SQLite and writes stock rows back to SQLite),
  - removed tracked raw placeholder files (`data/raw/*/.gitkeep`) and removed now-unused legacy data directories.
- 2026-02-23: Ticket 2 completed with DB-backed statistics endpoint:
  - `GET /api/v1/data/statistics` implemented in canonical router,
  - aggregation logic moved into `backend/app/services/statistics_service.py`,
  - payload aligned with frontend `StatisticsResponse` shape.
- 2026-02-23: Ticket 3 completed with canonical sentiment API endpoints:
  - `/api/v1/sentiment/analyze` and `/api/v1/sentiment/batch` implemented in `backend/app/api/v1/sentiment.py`,
  - endpoints use existing FinBERT inference utilities and preserve batch input order,
  - empty input handling returns HTTP 400 with clear validation message.
- 2026-02-23: Ticket 5 completed with LIME explainability integration:
  - `POST /api/v1/sentiment/explain` added in canonical sentiment router,
  - response returns token-level importance list (no HTML payload),
  - frontend Homepage now includes inline modal flow for explainability output.
- 2026-02-23: Local developer startup commands aligned to canonical backend app path:
  - `start-dev.ps1` and root `package.json` now start `app.main:app` rather than legacy `api.main:app`.
- 2026-02-24: Frontend debugging audit completed and converted into actionable backlog:
  - confirmed canonical backend currently lacks mounted `/api/v1/correlation/*` routes (frontend receives 404 on correlation overview),
  - confirmed Homepage metric mini-charts use synthetic random data causing non-deterministic visuals,
  - confirmed LIME modal typing triggers homepage re-render side-effects (charts appear to change while typing),
  - observed explain endpoint responsiveness issue requiring dedicated backend/frontend stabilization work.
- 2026-02-24: FE-1 fix completed (canonical correlation routing):
  - mounted correlation router into canonical API v1 assembly (`backend/app/api/v1/router.py`),
  - added lazy analyzer initialization in correlation router to avoid heavy startup-time initialization side effects,
  - added bootstrap test asserting correlation route presence in OpenAPI for canonical app.
- 2026-02-24: Historical dataset integration prep (stock_tweets.csv, ~80k tweets, 25 tickers, Sep 2021 – Sep 2022):
  - created `backend/scripts/import_tweets.py` with FinBERT batch processing, progress/resume support, and dual document+stock record creation,
  - backend "recent activity" and "trending" calculations now anchor to latest DB record instead of `datetime.utcnow()`, so historical data shows meaningful numbers,
  - `GET /api/v1/data/statistics` now accepts optional `?days=N` query param to filter to the last N days relative to newest record,
  - frontend `DashboardContext` passes selected timeframe to statistics API and re-fetches on timeframe change,
  - timeframe dropdown expanded: "All Time" (default), 7d, 14d, 30d, 90d, 6 Months, 1 Year,
  - legacy `api/routers/data.py` statistics recent_activity also anchored to latest data timestamp.
- 2026-02-24: Frontend design unification (both pages consistent):
  - added Navbar to StockAnalysis page for consistent navigation,
  - standardized all card/component borders to 1px solid #e0e0e0, 12px radius, 0.04 opacity shadows,
  - fixed dark-theme remnants in Granger causality cards, rolling stats, RollingCorrelation tooltip, ErrorBoundary, and NotFound page,
  - standardized page padding (40px) across Homepage and StockAnalysis,
  - extracted CorrelationHeatmap from inline styles to CSS file,
  - set color-scheme to light-only in index.css.
- 2026-02-24: FE-2 + FE-3 fix (deterministic mini-charts):
  - added `daily_trend` field to statistics API response (per-day count + net_sentiment via SQLAlchemy date grouping + case expressions),
  - added `DailyTrendPoint` interface to frontend API types,
  - removed `generateTrendData()` (Math.random) from Homepage,
  - Net Sentiment and Total Records mini-charts now use real daily data via `useMemo`,
  - chart data is deterministic across re-renders — typing in LIME modal no longer causes chart changes.
- 2026-02-25: FE-4 fix (LIME endpoint responsiveness):
  - root cause: LIME calls `_predict_proba` ~num_samples times, each running FinBERT inference on CPU — very slow at 1000 samples,
  - reduced default `num_samples` from 1000 to 500 in `sentiment.py` explain endpoint,
  - wrapped `run_lime_explain()` in `asyncio.to_thread()` to avoid blocking the async event loop,
  - frontend `explainSentiment()` timeout extended from 30 s to 120 s with `AbortController` support,
  - LIME modal now displays elapsed-time counter during processing with contextual helper text after 10 s,
  - added Cancel button alongside Run Explanation during LIME processing.
- 2026-02-25: FE-5 fix (navbar filters):
  - timeframe filter confirmed already wired: DashboardContext passes `days` to statistics API and `period` to correlation overview, re-fetches on change,
  - asset filter disabled with "Coming Soon" badge — no backend filtering support yet,
  - added `disabled`/`disabledMessage` props to `DropdownButton`, piped through `Navbar` via `assetDisabled`/`assetDisabledMessage`.
- 2026-02-25: FE-6 fix (placeholder copy cleanup):
  - removed misleading `SummaryCard` default props (`+0.38`, `Mildly Positive`, `SENTIMENT DISTRIBUTION`) — made embedded card props required,
  - improved defensive fallback copy in Homepage (`'Awaiting data'`, `'Fetching sentiment data from the backend…'`),
  - audited all N/A and empty-state messages — existing fallbacks are appropriate.