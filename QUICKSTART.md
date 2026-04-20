# Quick start

Three ways to run the backend and frontend together. Use whichever fits your workflow.

## 1. PowerShell (Windows)

From the repository root:

```powershell
.\start-dev.ps1
```

Installs frontend dependencies if `frontend/node_modules` is missing, then opens two windows: FastAPI (port 8000) and Vite (port 3000).

## 2. npm (single terminal)

First time at repo root:

```bash
npm install
cd frontend && npm install && cd ..
```

Then:

```bash
npm run dev
```

Uses `concurrently` to run `dev:backend` and `dev:frontend` with coloured logs. `Ctrl+C` stops both.

## 3. Docker Compose

```bash
docker-compose up --build
```

Use `docker-compose down` to stop.

---

## URLs

| Service   | URL                          |
|-----------|------------------------------|
| Dashboard | http://localhost:3000      |
| API       | http://localhost:8000      |
| OpenAPI   | http://localhost:8000/docs |

Vite is configured with `port: 3000` and proxies `/api` to the backend (`frontend/vite.config.ts`).

---

## First-time checklist

1. Copy `.env.example` to `.env` at the repo root and set `DATABASE_URL` (and optional API keys).
2. Backend: `cd backend`, create a venv, `pip install -r requirements.txt`.
3. Frontend: `cd frontend`, `npm install`.

Health check:

```bash
curl http://localhost:8000/health
```

More detail: [README-DEVELOPMENT.md](README-DEVELOPMENT.md) and the root [README.md](README.md).
