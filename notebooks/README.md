# Notebooks

This directory contains Jupyter notebooks for data collection, exploration, and model experimentation.

## 📓 Notebooks

### `01-reddit-ingest.ipynb`
**Purpose:** Collect finance-related posts from Reddit using the PRAW API.

**What it does:**
- Searches specified subreddits for posts containing finance keywords
- Cleans text (removes URLs, normalizes whitespace)
- Deduplicates posts and exports to JSON
- Outputs to `../data/processed/reddit/{date}/reddit_finance_{date}.json`

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
