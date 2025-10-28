# Notebooks

This directory contains Jupyter notebooks for data exploration, analysis, and experimentation.

> **⚠️ Note:** For production data collection, use the Python scripts in `backend/app/pipelines/` instead of notebooks. Notebooks here are for exploration and documentation only.

## 📓 Notebooks

### `01-reddit-ingest.ipynb`
**Purpose:** Documentation and prototyping for Reddit data collection.

**Status:** ✅ Production script available at `backend/app/pipelines/ingest_reddit.py`

**What this notebook demonstrates:**
- How to connect to Reddit API using PRAW
- Text cleaning and normalization techniques
- Post deduplication logic
- Data structure and output format

**For production use:**
Use the production script instead:
```bash
cd backend/app/pipelines
python ingest_reddit.py --help
```

See [backend/app/pipelines/README.md](../backend/app/pipelines/README.md) for full documentation.

---

### `02-twitter-ingest.ipynb`
**Purpose:** Documentation and prototyping for Twitter/X data collection.

**Status:** ✅ Production script available at `backend/app/pipelines/ingest_twitter.py`

**What this notebook demonstrates:**
- How to connect to Twitter API using Tweepy
- Text cleaning (URLs, mentions, hashtags)
- Spam and bot detection techniques
- Engagement-based quality filtering
- CSV export format

**For production use:**
Use the production script instead:
```bash
cd backend/app/pipelines
python ingest_twitter.py --help
```

**Quality Filters:**
- Removes spam patterns (promotional content, bot networks)
- Filters low engagement tweets (< 5 likes+retweets+replies)
- Detects uniform engagement patterns (bot networks)
- Removes excessive emojis, hashtags, and cashtags

See [backend/app/pipelines/README.md](../backend/app/pipelines/README.md) for full documentation.

---

## 🔧 Setup

**Requirements:**
```bash
pip install -r requirements.txt
```

**Environment Variables:**
Create a `.env` file in the project root with:
```bash
# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=investor-sentiment-dashboard/0.1 by your_username

# Twitter API
TWITTER_BEARER_TOKEN=your_bearer_token
```

### Getting API Credentials

**Reddit:**
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" as the app type
4. Copy the client ID and secret

**Twitter/X:**
1. Apply for developer account at https://developer.twitter.com/
2. Create an app
3. Get your Bearer Token (for read-only access)
4. Free tier: 1,500 tweets/month

**Usage:**
1. Open the notebook in Jupyter or VS Code
2. Adjust parameters in cell 2 if needed (subreddits, keywords, time filter)
3. Run all cells

**Output:**
- JSON file with posts containing: id, title, selftext, author, subreddit, timestamps, engagement metrics

---

## 🔧 Setup

Install dependencies for all notebooks:
```bash
cd notebooks
pip install -r requirements.txt
```

## 📁 Data Output

All notebooks write output to `../data/` (project-level data directory), not within the notebooks folder. This keeps the repository clean and separates code from data.
