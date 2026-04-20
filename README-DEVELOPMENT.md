# Development guide

This document complements the root [README.md](README.md) with local setup, configuration, and troubleshooting.

## Run options

### PowerShell (recommended on Windows)

```powershell
.\start-dev.ps1
```

Starts `uvicorn` on `app.main:app` and the Vite dev server in separate windows.

### npm (one terminal)

```bash
npm install          # root: installs concurrently
npm run dev          # backend + frontend
```

Individual scripts (see root `package.json`):

- `npm run dev:backend` — `cd backend && python -m uvicorn app.main:app --reload --host localhost --port 8000`
- `npm run dev:frontend` — `cd frontend && npm run dev`
- `npm run install:all` — installs frontend packages

### Docker

```bash
docker-compose up --build
```

### Manual (two terminals)

**Terminal 1 — backend**

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host localhost --port 8000
```

**Terminal 2 — frontend**

```bash
cd frontend
npm install
npm run dev
```

---

## Prerequisites

- Python 3.11+
- Node.js 18+ and npm 9+
- Optional: Docker Desktop for Compose workflows

---

## Configuration

### Backend

Prefer a repo-root `.env` (see `.env.example`). Typical variables include `DATABASE_URL` and optional keys for Reddit, X, NewsAPI, Groq.

**Without `DATABASE_URL`:** the API uses `backend/data/demo/sentiment_demo.sqlite` and seeds synthetic rows (see root README, **Local demo**). Readiness: `GET /health/ready` includes `"demo": true` in that mode.

Lean deployments: `pip install -r requirements-lean.txt` and set `LEAN_API=1` (see root README).

### Frontend

Optional `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000/api/v1
VITE_ENV=development
```

The dev server uses port **3000** by default (`frontend/vite.config.ts`) and proxies `/api` to port 8000.

---

## Troubleshooting

### Port already in use (8000)

PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
Stop-Process -Id <PID> -Force
```

### Backend import errors

```bash
cd backend
python -c "from app.main import app; print('OK')"
```

If this fails, reinstall dependencies and ensure the working directory is `backend/` when running Uvicorn.

### Frontend dependency issues

```bash
cd frontend
Remove-Item -Recurse -Force node_modules, .vite -ErrorAction SilentlyContinue
npm install
```

On synced folders (e.g. OneDrive), `npm ci` can hit file locks; prefer `npm install` or clone to a local non-synced path for heavy installs.

### Production-style frontend check

```bash
cd frontend
npm run build
npm run preview
```

---

## Useful URLs

| Service   | URL                     |
|-----------|-------------------------|
| Dashboard | http://localhost:3000 |
| API docs  | http://localhost:8000/docs |
| Health    | http://localhost:8000/health |
