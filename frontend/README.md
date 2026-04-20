# Frontend — Investor Sentiment Dashboard

React 18 + TypeScript SPA built with **Vite**. It consumes the FastAPI backend under `/api/v1` (see `src/services/api.ts`).

## Commands

```bash
npm install
npm run dev      # http://localhost:3000 (see vite.config.ts)
npm run build
npm run preview
npm run lint
```

`npm run test` / `test:run` are defined in `package.json`; add **Vitest** (and any test deps) to `devDependencies` if you want those scripts to run in a fresh install.

## Layout

- `src/pages/` — routed views (market overview, stock analysis, methodology, legal, LIME, etc.)
- `src/components/` — charts, layout, navbar, footer, error boundaries
- `src/context/` — shared dashboard data
- `src/services/api.ts` — HTTP client for the backend

Project overview, evaluation commands, and submission notes: [../README.md](../README.md).
