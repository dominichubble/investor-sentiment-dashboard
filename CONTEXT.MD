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