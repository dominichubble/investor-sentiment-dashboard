# Data Pipelines

Scripts for collecting data from social media and news sources.

## 📁 Structure

```
pipelines/
├── __init__.py
├── ingest_reddit.py      # Reddit data collection
└── utils.py              # Shared utilities (future)
```

## 🚀 Reddit Ingestion Pipeline

### Usage

**Basic usage (uses defaults):**
```bash
cd backend/app/pipelines
python ingest_reddit.py
```

**Custom time filter and limit:**
```bash
python ingest_reddit.py --time-filter day --limit 100
```

**Specific subreddits:**
```bash
python ingest_reddit.py --subreddits stocks investing finance
```

**Custom keywords:**
```bash
python ingest_reddit.py --keywords "stock market" "earnings" "fed"
```

**Custom output directory:**
```bash
python ingest_reddit.py --output ../../../data/processed/reddit --run-id 2025-10-28
```

**Combine multiple options:**
```bash
python ingest_reddit.py \
  --subreddits wallstreetbets stocks \
  --time-filter week \
  --limit 500 \
  --output ../../../data/raw/reddit \
  --log-level DEBUG
```

### Configuration

Set these environment variables (or use `.env` file at project root):

```bash
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=investor-sentiment-dashboard/0.1 by your_username
```

### Output

The script generates two files:

1. **`reddit_finance_{date}.json`** - Array of normalized posts
2. **`reddit_finance_{date}_meta.json`** - Metadata about the run (timestamp, parameters, stats)

### Features

- ✅ **Robust error handling** - Continues if one subreddit fails
- ✅ **Logging** - Detailed logs with timestamps
- ✅ **Deduplication** - Removes duplicate posts across subreddits
- ✅ **Text cleaning** - Removes URLs, normalizes whitespace
- ✅ **Metadata tracking** - Records run parameters and results
- ✅ **CLI arguments** - Fully configurable via command line
- ✅ **Read-only mode** - Safe, doesn't post or modify Reddit

### Scheduling

**Linux/Mac (cron):**
```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/backend/app/pipelines && python ingest_reddit.py
```

**Windows (Task Scheduler):**
```powershell
# Create a scheduled task
schtasks /create /tn "Reddit Ingestion" /tr "python C:\path\to\ingest_reddit.py" /sc daily /st 02:00
```

## 🔧 Development

### Adding a new pipeline

1. Create new script (e.g., `ingest_twitter.py`)
2. Follow the same structure:
   - CLI arguments with `argparse`
   - Proper logging
   - Error handling
   - Metadata generation
3. Update this README

### Testing

```bash
# Run with a small limit for testing
python ingest_reddit.py --limit 10 --time-filter day
```
