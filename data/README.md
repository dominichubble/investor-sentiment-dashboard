# Data Directory

This directory contains sample cleaned and annotated datasets for the Investor Sentiment Dashboard project.

## Current Storage Model (DB-First)

The project runs in **DB-first mode** for daily operations:

- The canonical store is **PostgreSQL (Neon)** via `DATABASE_URL` in `.env` (and the same secret in GitHub Actions for scheduled ingest).
- Local `data/raw/`, `data/predictions/`, and `data/stock_sentiments/` are legacy paths and are not required for runtime.
- Local file artifacts (for debugging or audit) are optional and disabled by default in ingestion scripts.

If any older sections below reference raw/processed folder workflows, treat them as historical notes rather than the primary execution path.

## Overview

The datasets in this repository represent **cleaned and preprocessed** financial sentiment data from multiple sources:

- **Reddit**: Posts from financial subreddits
- **News**: Articles from major financial news sources

All data has been preprocessed using the FinBERT-optimized configuration to prepare it for sentiment analysis.

## Directory Structure

```
data/
├── README.md                          # This file
├── raw/                               # Raw data (gitignored, not committed)
│   ├── reddit/YYYY-MM-DD/            # Raw Reddit posts
│   ├── twitter/YYYY-MM-DD/           # Raw tweets
│   └── news/YYYY-MM-DD/              # Raw news articles
└── processed/                         # Cleaned and annotated data (committed)
    ├── reddit/                       # Preprocessed Reddit posts
    ├── twitter/                      # Preprocessed tweets
    └── news/                         # Preprocessed news articles
```

## Dataset Schemas

### Reddit Data (`processed/reddit/`)

**Source**: Reddit financial subreddits  
**Format**: JSON with nested structure

**Schema**:
**Data Fields:**

- `id`: Unique post identifier
- `title`: Post title
- `text`: Original post text
- `text_cleaned`: Preprocessed text (FinBERT-optimized)
- `text_tokens`: Tokenized version of cleaned text
- `author`: Reddit username
- `subreddit`: Subreddit name
- `created_utc`: Unix timestamp
- `score`: Upvotes minus downvotes
- `num_comments`: Number of comments
- `upvote_ratio`: Percentage of upvotes (0-1)
- `url`: Direct link to post
- `permalink`: Reddit permalink

**Metadata Fields:**

- `preprocessing.processed_at`: Timestamp of preprocessing
- `preprocessing.config`: Preprocessing configuration used
- `preprocessing.stats`: Token counts and statistics

### News Data (`processed/news/`)

**Source**: Financial news APIs  
**Format**: JSON with nested structure

**Data Fields**:

- `source_id`: News source identifier
- `source_name`: Human-readable source name
- `author`: Article author
- `title`: Article headline
- `description`: Article summary
- `content`: Full article text
- `content_cleaned`: Preprocessed content (FinBERT-optimized)
- `content_tokens`: Tokenized version of cleaned content
- `url`: Article URL
- `url_to_image`: Featured image URL
- `published_at`: ISO 8601 publication timestamp

**Metadata Fields:**

- `preprocessing.processed_at`: Timestamp of preprocessing
- `preprocessing.config`: Preprocessing configuration used
- `preprocessing.stats`: Token counts and statistics

## Preprocessing Configuration

All sample data uses the **FinBERT** preprocessing configuration, optimized for transformer-based sentiment models:

### Key Features:

- ✅ **Case Preservation**: Maintains original casing (important for entities like "NVDA", "Fed")
- ✅ **Financial Terms**: Preserves domain-specific vocabulary (bullish, bearish, rally, crash)
- ✅ **Numeric Context**: Keeps financial punctuation (%, $, decimals)
- ✅ **Negation Handling**: Marks negations to prevent sentiment reversal (e.g., "not good" → "not_good")
- ✅ **URL/Mention Removal**: Cleans social media artifacts
- ❌ **No Stopword Removal**: Transformers use full context
- ❌ **No Lemmatization**: Transformers understand word forms

### Example Transformations:

**Before**: "Stock isn't doing well. Down 5% today. 😔"  
**After**: "Stock is not_doing well Down 5% today"

**Before**: "NVDA crushed earnings! Up 12% after hours 🚀 https://t.co/xyz"  
**After**: "NVDA crushed earnings Up 12% after hours"

## Data Collection Methodology

### Reddit

- **Subreddits**: wallstreetbets, stocks, investing, finance
- **Search Keywords**: stock market, earnings, fed rate, inflation, major tickers
- **Time Filter**: Last 7 days
- **Sorting**: New posts
- **Quality Filters**: Minimum engagement threshold

### Twitter/X

- **Keywords**: stock market, earnings, inflation, major tickers ($NVDA, $TSLA, etc.)
- **Language**: English only
- **Quality Filters**:
  - Minimum 5 total engagements (likes + retweets + replies)
  - Spam/bot detection
  - Excessive emoji filtering

### News

- **Sources**: Bloomberg, Reuters, WSJ, Financial Times, CNBC, Business Insider
- **Keywords**: stock market, earnings, interest rates, inflation, market crash
- **Time Range**: Last 7 days
- **Language**: English
- **Quality Filters**:
  - Minimum content length (100+ characters)
  - Removes paywalled/[Removed] articles
  - URL/title deduplication

## Usage

### Loading Datasets

```python
import json
from pathlib import Path

# Load preprocessed Reddit data
reddit_files = list(Path("data/processed/reddit").glob("*.json"))
for file in reddit_files:
    with open(file, "r") as f:
        data = json.load(f)
        records = data.get("data", data)  # Handle nested structure

        # Access cleaned text for sentiment analysis
        for post in records[:5]:  # First 5 posts
            print(f"Title: {post.get('title', 'N/A')}")
            print(f"Cleaned: {post.get('text_cleaned', 'N/A')}")
            print(f"Subreddit: {post.get('subreddit', 'N/A')}")
            print()

# Load preprocessed News data
news_files = list(Path("data/processed/news").glob("*.json"))
for file in news_files:
    with open(file, "r") as f:
        data = json.load(f)
        records = data.get("data", data)

        for article in records[:3]:  # First 3 articles
            print(f"Title: {article.get('title', 'N/A')}")
            print(f"Source: {article.get('source_name', 'N/A')}")
            print(f"Cleaned: {article.get('content_cleaned', 'N/A')[:100]}...")
            print()
```

### Preprocessing New Data

To preprocess new data collected from the pipelines:

```bash
cd backend/app/pipelines

# Process specific source (automatically outputs to data/processed/{source}/)
python preprocess_data.py --input ../../../data/raw/reddit/YYYY-MM-DD --config finbert --source reddit
python preprocess_data.py --input ../../../data/raw/news/YYYY-MM-DD --config finbert --source news

# Or specify custom output directory
python preprocess_data.py --input ../../../data/raw/reddit/YYYY-MM-DD --output ../../../data/processed/reddit --config finbert
```

See [`backend/app/pipelines/README.md`](../backend/app/pipelines/README.md) for detailed pipeline documentation.

## Data Privacy & Ethics

- ✅ All data is from **public sources**
- ✅ No personal identifiable information (PII) collected beyond public usernames
- ✅ Data used for **academic research purposes only** (Final Year Project, Loughborough University)
- ✅ Complies with platform Terms of Service
- ⚠️ Data collection requires valid API credentials (see `.env.example`)

## Related Documentation

- [Data Pipeline Overview](../docs/data-pipeline.md)
- [Preprocessing Guide](../docs/preprocessing-guide.md)
- [Backend Pipelines](../backend/app/pipelines/README.md)
- [Notebooks](../notebooks/README.md)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Preprocessing Config**: FinBERT-optimized  
**Project**: Final Year Project - Investor Sentiment Dashboard  
**Institution**: Loughborough University
