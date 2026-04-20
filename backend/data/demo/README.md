# Local demo database (SQLite)

If you run the backend **without** setting `DATABASE_URL`, it automatically uses:

`backend/data/demo/sentiment_demo.sqlite`

On first startup the file is created (if missing) and filled with **synthetic** sentiment rows (`app/storage/demo_seed.py`) so the dashboard and `/api/v1/data/*` routes work without PostgreSQL or any `.env` file.

- The data is **not** real market or social text; it exists only for UI and API smoke testing.
- Set `DATABASE_URL` to your Neon (or other Postgres) connection string to use your real corpus instead.
- The `.sqlite` file is gitignored; each clone generates its own copy on first run.
