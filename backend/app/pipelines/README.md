# Data Pipelines

Scripts for collecting data from social media and news sources.

## 📁 Structure

```
pipelines/
├── __init__.py
├── ingest_reddit.py      # Reddit data collection
├── ingest_twitter.py     # Twitter/X data collection
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
# Test Reddit pipeline with small limit
python ingest_reddit.py --limit 10 --time-filter day

# Test Twitter pipeline with minimal tweets
python ingest_twitter.py --max-tweets 10
```

---

## 🐦 Twitter/X Ingestion Pipeline

### Prerequisites

1. **Get Twitter API credentials:**
   - Apply for developer account at https://developer.twitter.com/
   - Create an app and get your **Bearer Token**
   - Free tier: 1,500 tweets/month (50 tweets/day budget)

2. **Add credentials to `.env`:**
   ```bash
   TWITTER_BEARER_TOKEN=your_bearer_token_here
   ```

### Usage

**Basic usage (30 tweets with default keywords):**
```bash
cd backend/app/pipelines
python ingest_twitter.py
```

**Custom keywords:**
```bash
python ingest_twitter.py --keywords "NVDA earnings" "fed meeting" "market crash"
```

**Adjust tweet count:**
```bash
python ingest_twitter.py --max-tweets 50
```

**Higher engagement threshold (filter low-quality tweets):**
```bash
python ingest_twitter.py --min-engagement 10
```

**Custom output directory:**
```bash
python ingest_twitter.py --output ../../../data/processed/twitter --run-id 2025-10-28
```

**Combine multiple options:**
```bash
python ingest_twitter.py \
  --keywords "earnings beat" "fed rate" "market crash" \
  --max-tweets 50 \
  --min-engagement 10 \
  --output ../../../data/processed/twitter
```

### Default Configuration

```python
KEYWORDS = [
    'stock market', 'stocks', 'earnings', 'fed rate', 'inflation',
    'NVDA', 'TSLA', 'AAPL', 'wall street', 'bull market', 'bear market'
]
MAX_TWEETS = 30  # Conservative for free tier
MIN_ENGAGEMENT = 5  # Minimum likes + retweets + replies
LANGUAGE = 'en'
```

### Quality Filtering

The Twitter pipeline automatically filters:
- ✅ **Spam tweets** (promotional content, bot patterns)
- ✅ **Bot networks** (uniform engagement patterns like 9,9,9)
- ✅ **Low engagement** (tweets below minimum threshold)
- ✅ **Short tweets** (< 20 characters)
- ✅ **Excessive emojis** (> 5 emojis)
- ✅ **Excessive hashtags** (> 8 hashtags)
- ✅ **Stock promoters** ("This blogger recommends stocks...")

### Output Format

**CSV file structure:**
```csv
id,text,raw_text,author_id,created_at,retweet_count,reply_count,like_count,quote_count,lang
123456789,"Market analysis...","#Market analysis...","user123","2025-10-28T10:00:00",5,2,10,1,en
```

**Metadata file (txt):**
```
Run ID: 2025-10-28
Timestamp: 2025-10-28T10:15:30
Total tweets: 28
```

### API Limits (Free Tier)

- **Monthly limit:** 1,500 tweets
- **Daily budget:** ~50 tweets (conservative)
- **Per request:** Max 100 tweets
- **Rate limit:** 450 requests per 15 minutes
- **Search window:** Last 7 days only

**Recommendation:** Run 1-2 times per day with 25-30 tweets per run

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--keywords` | Keywords to search for | stock market, stocks, earnings, etc. |
| `--max-tweets` | Maximum tweets to collect | 30 |
| `--language` | Language code | en |
| `--min-engagement` | Minimum engagement threshold | 5 |
| `--output` | Output directory | data/processed/twitter |
| `--run-id` | Run identifier | Current date (YYYY-MM-DD) |
| `--log-level` | Logging level | INFO |

---

## 🔄 Adding New Pipelines

1. Create new script (e.g., `ingest_news.py`)
2. Follow the same structure:
   - CLI arguments with `argparse`
   - Proper logging
   - Error handling
   - Metadata generation
   - Quality filtering
3. Add tests in `backend/tests/`
4. Update this README

