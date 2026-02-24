# TODO Tickets (Top 5)

## Confirmed Direction (2026-02-23)
- Backend app entrypoint: `backend/app/main.py`.
- API contract handling: `frontend/src/services/api.ts` is a reference, not strict source-of-truth.
- Data layer: implement SQLite directly.
- Date/timestamp policy: UTC ISO-8601 consistently.
- Correlation endpoints: implement real analytics (no placeholder stubs for MVP outcomes).
- Sentiment response shape: practical default output; extend via options when needed.
- Explainability scope: LIME first.
- Evaluation outputs: generate through code tasks/scripts and save artifacts.
- Testing bar: pytest minimum for critical endpoints/paths.
- Architecture direction: DB-first source-of-truth with separation across ingestion, processing, storage, and frontend/API. Local `data/` files are optional artifacts, not runtime dependency.

## Progress Log
- 2026-02-23: Added and confirmed project-wide implementation decisions from stakeholder Q&A.
- 2026-02-23: Ready to begin Ticket 1 with canonical structure and constraints locked.
- 2026-02-23: Ticket 1 completed and validated (health endpoint, v1 mount, CORS, uvicorn startup check, pytest checks).
- 2026-02-23: Ticket 4 implemented in canonical app path with SQLite startup initialization, `/api/v1/data/predictions`, and import job scaffolding for DB ingestion.
- 2026-02-23: DB-first hardening completed for ingestion pipelines (`reddit`, `news`, `twitter`):
  - default flow stores directly to DB,
  - optional local artifacts are gated behind `--write-files`,
  - pipeline success no longer depends on writing to `data/`.
- 2026-02-23: Daily GitHub Actions collection moved to DB-first mode:
  - workflow now ingests directly to DB with `--store-db --no-write-files`,
  - database summary includes `sentiment_records` row count.
- 2026-02-23: Local raw data artifacts removed after migration verification.
- 2026-02-23: Removed additional legacy local data artifacts (`data/predictions/*`, `data/stock_sentiments/*`) and aligned docs/ignore rules to DB-first operation.
- 2026-02-23: Converted `process_stock_sentiments.py` to DB-native enrichment and removed tracked raw placeholder files/directories.
- 2026-02-23: Ticket 2 implemented in canonical app path:
  - added DB-backed statistics aggregation service (`backend/app/services/statistics_service.py`),
  - added `GET /api/v1/data/statistics` in `backend/app/api/v1/data.py`,
  - added canonical endpoint tests (`backend/tests/api/test_v1_statistics.py`).
- 2026-02-23: Ticket 3 implemented in canonical app path:
  - added real `/api/v1/sentiment/analyze` and `/api/v1/sentiment/batch` endpoints in `backend/app/api/v1/sentiment.py`,
  - wired endpoints to existing FinBERT inference utilities,
  - added canonical endpoint tests (`backend/tests/api/test_v1_sentiment.py`) including required empty-input 400 behavior.
- 2026-02-23: Ticket 5 implemented (LIME-first explainability):
  - added `POST /api/v1/sentiment/explain` returning token-level importance list only,
  - integrated inline explainability modal on Homepage,
  - added canonical backend tests for explainability endpoint behavior.
- 2026-02-23: Dev-runner alignment fix:
  - updated local startup commands to run canonical backend app (`app.main:app`) instead of legacy app path.

## Ticket 1 — Bootstrapping FastAPI + Health + Versioning
**Status**
Completed (2026-02-23)

**Goal**
Create a runnable FastAPI backend with `/health` and base API versioning at `/api/v1`.

**Acceptance Criteria**
- `uvicorn ...` runs without error.
- `GET /health` returns `{ "status": "ok" }`.
- API is mounted under `/api/v1` (even if endpoints are stubbed initially).
- CORS configured so `http://localhost:5173` (Vite) can call it.

**Suggested Files**
- `backend/app/api/main.py` (or `backend/app/main.py`)
- `backend/app/api/router.py` (optional)
- `backend/app/api/v1/__init__.py`
- `backend/app/api/v1/data.py` (stub)
- `backend/app/api/v1/sentiment.py` (stub)

**Notes**
- Keep structure consistent from day 1; don’t scatter endpoints.
- Implementation decision: canonical app entrypoint is `backend/app/main.py`.
- Implemented files:
  - `backend/app/main.py`
  - `backend/app/api/__init__.py`
  - `backend/app/api/v1/__init__.py`
  - `backend/app/api/v1/router.py`
  - `backend/app/api/v1/data.py` (stub)
  - `backend/app/api/v1/sentiment.py` (stub)
  - `backend/tests/api/test_bootstrap_app.py`


## Ticket 2 — Implement /data/statistics to Match Frontend
**Status**
Completed (2026-02-23)

**Goal**
Implement `GET /api/v1/data/statistics` returning the exact shape expected by `StatisticsResponse` in `frontend/src/services/api.ts`.

**Acceptance Criteria**
- Endpoint returns a valid `StatisticsResponse` JSON with:
  - `total_predictions`
  - `total_stocks_analyzed`
  - `sentiment_distribution` (counts + percentages)
  - `top_stocks` list
  - `recent_activity` {last_24h, last_7d, last_30d}
  - `date_range` {earliest, latest}
- Frontend Homepage loads and renders cards using real API response.
- If DB not ready yet: it can return computed stats from a local JSON/CSV “seed” file to unblock UI.

**Suggested Files**
- `frontend/src/services/api.ts` (reference types)
- `backend/app/api/v1/data.py` (endpoint)
- `backend/app/services/statistics_service.py` (aggregation logic)
- `backend/app/storage/*` (temporary adapter)

**Notes**
- This is the fastest “end-to-end demo unlock” for your dashboard.
- Data source decision: back with SQLite-backed aggregation (no mock-only stopgap).
- Implemented with DB-first aggregation and frontend-compatible response shape.


## Ticket 3 — Implement /sentiment/analyze + /sentiment/batch (FinBERT wiring)
**Status**
Completed (2026-02-23)

**Goal**
Expose FinBERT inference through the API.

**Acceptance Criteria**
- `POST /api/v1/sentiment/analyze` accepts `{ text, options? }` and returns:
  - predicted label + score (and optionally all scores)
- `POST /api/v1/sentiment/batch` accepts `{ texts, options? }` and returns list of results aligned with input order.
- Errors are clean:
  - empty text => 400 with clear message
- Minimal tests added (at least happy path + empty input).

**Suggested Files**
- `backend/app/api/v1/sentiment.py`
- `backend/app/models/sentiment_inference.py` (already exists)
- `backend/tests/test_sentiment_endpoints.py`

**Notes**
- Keep latency acceptable: reuse singleton model (already in `get_model()`).
- Response-shape decision: keep default concise and add optional extended score payload when requested.
- Empty-input validation is implemented as HTTP 400 with clear message (per project decision for this ticket).


## Ticket 4 — Persistence + Prediction Ingest into DB (SQLite first)
**Status**
Completed (2026-02-23)

**Goal**
Store predictions in SQLite and make them queryable for stats/trending.

**Acceptance Criteria**
- SQLite DB created automatically on startup (or via init command).
- A `Prediction` table exists with:
  - id, source, timestamp, text, label, score, ticker(optional), metadata(optional)
- A simple “import job” exists:
  - reads from `data/raw` or `data/processed` outputs
  - runs sentiment
  - inserts into DB
- `GET /api/v1/data/predictions` returns paginated results matching `PredictionsResponse` in frontend client.

**Suggested Files**
- `backend/app/db.py` (engine/session)
- `backend/app/models/db_models.py` (SQLAlchemy models)
- `backend/app/services/import_service.py`
- `backend/app/api/v1/data.py` (predictions endpoint)

**Notes**
- Start with SQLite to ship MVP; you can mention scalability and future migration in dissertation.
- Confirmed: SQLite is required now (not deferred).
- Implementation decision used: reuse/adapt existing SQLAlchemy + SQLite storage layer (`sentiment_records`) rather than creating a second parallel DB stack.
- Added canonical endpoint implementation at `backend/app/api/v1/data.py` for paginated predictions.
- Added import job service at `backend/app/services/import_service.py` and CLI entry script `backend/scripts/import_to_db.py`.
- Ingestion scripts now use DB-first persistence through `ImportService.import_from_records(...)` with optional file export.


## Ticket 5 — Novelty Slice: Explainability (LIME first) + UI Display
**Status**
Completed (2026-02-23)

**Goal**
Deliver your novelty: explainability integrated into the product.

**Acceptance Criteria**
- `POST /api/v1/sentiment/explain` (or `/explainability/lime`) accepts `{ text }` and returns:
  - token-level importance (list of {token, weight})
  - optionally an HTML snippet
- Frontend can display explanation:
  - highlight tokens in the input text OR show ranked list
- Include 3–5 saved examples for dissertation (screenshots or JSON outputs).
- Document limitations (approximation, runtime, stability).

**Suggested Files**
- `backend/app/services/explainability/lime_explainer.py`
- `backend/app/api/v1/sentiment.py` (new endpoint)
- `frontend/src/components/...` (new Explanation component)
- `frontend/src/pages/...` (where it’s shown, even as a modal)

**Notes**
- LIME is usually quicker for text explanations than SHAP on transformers.
- SHAP can be a stretch goal; implement later if time.
- Confirmed: prioritize LIME for MVP novelty delivery.
- Implemented on canonical sentiment API path (`/api/v1/sentiment/explain`) with token list response and inline modal UI.


## Frontend Stabilization Backlog (Post-MVP Findings)
These are evidence-based issues found during a targeted frontend debugging pass.

### FE-1 — Correlation API is not wired in canonical backend app
**Priority**
Critical

**Status**
Completed (2026-02-24)

**Problem**
- Frontend calls `/api/v1/correlation/*`, but canonical app router (`backend/app/api/v1/router.py`) currently mounts only `data` and `sentiment`.
- Live check returns `404` on `/api/v1/correlation/overview/all`.

**Impact**
- Correlation overview and correlation analysis page cannot load real data on canonical startup path.

**Acceptance Criteria**
- Add canonical correlation router under `backend/app/api/v1` and include it in `api_router`.
- `GET /api/v1/correlation/overview/all` returns `200` in canonical app.
- Stock analysis page renders correlation sections without endpoint-not-found failures.
- Implemented by mounting correlation routes in canonical v1 router and lazy-loading analyzer initialization to avoid heavy startup-time blocking.

### FE-2 — Homepage mini-trend charts use random synthetic data
**Priority**
High

**Problem**
- `Homepage.tsx` uses `generateTrendData()` with `Math.random()` for metric-card charts.
- This data is not from DB/API time-series.

**Impact**
- Charts do not represent true time-series behavior.
- Dashboard appears inconsistent and non-deterministic.

**Acceptance Criteria**
- Remove random chart generation from Homepage.
- Replace with real backend-derived trend series (or hide trend charts until real series endpoint is available).
- Chart values are stable across re-renders for unchanged data.

### FE-3 — Typing in LIME modal causes charts to change
**Priority**
High

**Problem**
- Every keystroke updates component state (`explainText`) which re-renders Homepage.
- Re-render re-executes random chart data generation, so unrelated graphs visually change.

**Impact**
- Severe UX trust issue ("dashboard moves while typing").

**Acceptance Criteria**
- Ensure typing in explainability modal does not mutate metric chart data.
- Use memoized/derived deterministic chart data only.
- Manual test: typing 20+ chars in modal leaves metric charts unchanged.

### FE-4 — LIME explainability endpoint is not reliably responsive
**Priority**
High

**Problem**
- Live endpoint check for `POST /api/v1/sentiment/explain` timed out in local verification.
- Existing tests mock explainer path; no integration check validates end-to-end latency/responsiveness.

**Impact**
- Explainability feature appears broken to end users.

**Acceptance Criteria**
- Identify root cause of timeout (model load, LIME runtime, dependency/runtime issue) and fix.
- Add backend integration smoke check for explain endpoint response under a defined timeout budget.
- Frontend shows clear progress and actionable error states for long-running explain requests.

### FE-5 — Navbar filters are mostly cosmetic today
**Priority**
Medium

**Problem**
- Asset/timeframe selections update context state, but data fetch does not use these selections in API requests.
- Dashboard context currently fetches fixed params for correlation overview and unparameterized statistics.

**Impact**
- Controls appear as placeholders/non-functional filters.

**Acceptance Criteria**
- Wire selected timeframe/asset into backend query parameters where supported.
- For unsupported filters, disable with "coming soon" state or remove until implemented.
- Filter changes produce observable deterministic changes in fetched data.

### FE-6 — Placeholder/default copy and fallback values need cleanup
**Priority**
Medium

**Problem**
- Some components still carry template defaults (e.g., `SummaryCard` fallback values) and generic instructional text.
- Copy does not always reflect current live data or feature state.

**Impact**
- Product polish and credibility are reduced.

**Acceptance Criteria**
- Remove or replace static fallback values that can leak into UI.
- Audit and revise copy to reflect real system behavior and known limitations.
- Add empty-state messaging that is specific, actionable, and endpoint-aware.