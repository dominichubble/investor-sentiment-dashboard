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

**Requirements:**
```bash
pip install -r requirements.txt
```

**Environment Variables:**
Create a `.env` file in the project root with:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=investor-sentiment-dashboard/0.1 by your_username
```

**To get Reddit API credentials:**
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" as the app type
4. Copy the client ID and secret

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
