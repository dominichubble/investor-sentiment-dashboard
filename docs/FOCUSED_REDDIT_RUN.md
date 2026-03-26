# Focused Reddit pipeline (collect → ML → Neon)

Use this when you want **a narrow set of tickers** and **in-depth Reddit coverage** for later evaluation.

## One-time setup on your PC

1. **Python env** — from repo root, install backend deps (includes `torch`, `praw`, `psycopg2-binary`, etc.):

   ```powershell
   cd backend
   pip install -r requirements.txt
   ```

2. **CUDA (optional, recommended for RTX 2070 Super)** — install a **CUDA-enabled** PyTorch build from [pytorch.org](https://pytorch.org) so FinBERT uses the GPU. If you skip this, inference still runs on CPU (slower).

3. **`.env` at repo root** must include:

   - `DATABASE_URL` — Neon connection string  
   - `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`

4. **Ticker / subreddit lists (optional)** — copy examples and edit:

   ```text
   data/config/focus_tickers.example.txt   →   data/config/focus_tickers.txt
   data/config/focus_subreddits.example.txt →   data/config/focus_subreddits.txt
   ```

   If those files are missing, the script uses a small built-in default ticker and subreddit set.

## Run the full pipeline

From **repository root** (adjust path if your folder name differs):

```powershell
cd "C:\path\to\investor-sentiment-dashboard"
python backend\scripts\focused_reddit_pipeline.py
```

Or use the helper (also from repo root):

```powershell
.\backend\scripts\run_focused_reddit.ps1
.\backend\scripts\run_focused_reddit.ps1 -Quick
```

**Quick test** (smaller limits, faster):

```powershell
python backend\scripts\focused_reddit_pipeline.py --quick
```

**Override tickers without editing files:**

```powershell
python backend\scripts\focused_reddit_pipeline.py --tickers NVDA TSLA PLTR
```

**Keep a JSON backup** of raw posts (in addition to Neon):

```powershell
python backend\scripts\focused_reddit_pipeline.py --write-files
```

---

## Copy-paste prompt (for you or an assistant)

Use this verbatim when you sit down at your PC and want the same workflow clarified or automated further:

> I’m running the investor-sentiment-dashboard on my Windows PC. I want to execute the **focused Reddit pipeline**: fetch Reddit posts for a **small list of tickers**, run **FinBERT sentiment** locally (I have an RTX 2070 Super — use GPU if PyTorch CUDA is installed), and **store results in Neon** via `DATABASE_URL` in the repo-root `.env`.
>
> Run from the repo root:
> `python backend\scripts\focused_reddit_pipeline.py`
> For a faster check, add `--quick`. To pin tickers inline: `--tickers NVDA TSLA AAPL`.
> Optional: copy `data/config/focus_tickers.example.txt` to `data/config/focus_tickers.txt` and edit my watchlist; same for `focus_subreddits.txt`.
>
> Confirm `REDDIT_*` and `DATABASE_URL` are set, then run the script and report `records_inserted` and any errors.

---

## What the script does

| Stage | What happens |
|--------|----------------|
| **Collect** | For each configured subreddit: new/hot/top/rising + **per-ticker searches** (`new`, `relevance`, `hot`; `year` window when not `--quick`). |
| **ML** | `ImportService` runs FinBERT on post text (+ aspect snippets). GPU is used automatically if PyTorch sees CUDA. |
| **Storage** | Rows go to **Neon** (`published_at` = post time, `ingested_at` = insert time). Duplicate `(id)` keys are skipped. |

## If Reddit rate-limits you

Edit the script’s `sleep_s` values upward, or call `reddit_bulk_ingest` directly with `--sleep-seconds 1.5` after checking `python -m app.pipelines.reddit_bulk_ingest --help`.
