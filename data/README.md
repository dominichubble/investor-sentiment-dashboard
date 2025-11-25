# Data Directory

This directory contains sample cleaned and annotated datasets for the Investor Sentiment Dashboard project.

## Overview

The datasets in this repository represent **cleaned and preprocessed** financial sentiment data collected on **November 25, 2025** from:
- **Reddit**: 292 posts from financial subreddits (r/wallstreetbets, r/stocks, r/investing, r/finance)
- **News**: 100 articles from major financial news sources (Bloomberg, Reuters, CNBC, etc.)

All data has been preprocessed using the FinBERT-optimized configuration to prepare it for sentiment analysis.

## Directory Structure

```
data/
├── README.md                                    # This file
├── raw/                                         # Raw data (gitignored, not committed)
│   ├── reddit/2025-11-25/                      # Raw Reddit posts
│   └── news/2025-11-25/                        # Raw news articles
└── processed/                                   # Cleaned and annotated data (committed)
    ├── reddit/
    │   └── reddit_finance_2025-11-25.json      # 292 cleaned Reddit posts
    └── news/
        └── news_finance_2025-11-25.json        # 100 cleaned news articles
```

## Datasets

### `processed/reddit/reddit_finance_2025-11-25.json`

**Source**: Reddit financial subreddits (r/wallstreetbets, r/stocks, r/investing, r/finance)  
**Records**: 292 posts collected on November 25, 2025  
**Format**: JSON array of objects  
**Time Range**: Past week (November 18-25, 2025)

**Schema**:
- `id` (string): Unique Reddit post ID
- `title` (string): Post title
- `text` (string): Original post text
- `text_cleaned` (string): Preprocessed text ready for sentiment analysis
- `author` (string): Reddit username
- `subreddit` (string): Subreddit name
- `created_utc` (integer): Unix timestamp
- `score` (integer): Upvotes minus downvotes
- `num_comments` (integer): Number of comments
- `upvote_ratio` (float): Percentage of upvotes (0-1)
- `url` (string): Direct link to post
- `permalink` (string): Reddit permalink
- `processed_at` (string): ISO 8601 timestamp of preprocessing
- `preprocessing_config` (string): Configuration used ("finbert")

### `processed/news/news_finance_2025-11-25.json`

**Source**: Financial news APIs (Bloomberg, Reuters, CNBC, Financial Times, WSJ, Business Insider, Fortune)  
**Records**: 100 articles collected on November 25, 2025  
**Format**: JSON array of objects  
**Time Range**: Past 7 days (November 18-25, 2025)

**Schema**:
- `source_id` (string): News source identifier
- `source_name` (string): Human-readable source name
- `author` (string): Article author
- `title` (string): Article headline
- `description` (string): Article summary
- `content` (string): Full article text
- `content_cleaned` (string): Preprocessed content
- `url` (string): Article URL
- `url_to_image` (string): Featured image URL
- `published_at` (string): ISO 8601 timestamp
- `processed_at` (string): ISO 8601 timestamp of preprocessing
- `preprocessing_config` (string): Configuration used ("finbert")

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

# Load Reddit dataset (292 posts)
with open("data/processed/reddit/reddit_finance_2025-11-25.json", "r") as f:
    reddit_data = json.load(f)

# Load News dataset (100 articles)
with open("data/processed/news/news_finance_2025-11-25.json", "r") as f:
    news_data = json.load(f)

# Access cleaned text for sentiment analysis
for post in reddit_data[:5]:  # First 5 posts
    print(f"Title: {post['title']}")
    print(f"Cleaned: {post['text_cleaned']}")
    print(f"Subreddit: {post['subreddit']}")
    print()
```

### Preprocessing New Data

To preprocess new data collected from the pipelines:

```bash
cd backend/app/pipelines

# Process all sources
python preprocess_data.py --source all --config finbert

# Process specific source
python preprocess_data.py --source reddit --config finbert

# Process specific date directory
python preprocess_data.py --input ../../../data/raw/reddit/2025-11-25 --config finbert
```

See [`backend/app/pipelines/README.md`](../backend/app/pipelines/README.md) for detailed pipeline documentation.

## Data Privacy & Ethics

- ✅ All data is from **public sources**
- ✅ No personal identifiable information (PII) collected beyond public usernames
- ✅ Data used for **academic research purposes only** (Final Year Project, Loughborough University)
- ✅ Complies with platform Terms of Service
- ✅ Real data collected from Reddit and News APIs on November 25, 2025
- ⚠️ Data collection requires valid API credentials (see `.env.example`)

## Related Documentation

- [Data Pipeline Overview](../docs/data-pipeline.md)
- [Preprocessing Guide](../docs/preprocessing-guide.md)
- [Backend Pipelines](../backend/app/pipelines/README.md)
- [Notebooks](../notebooks/README.md)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Last Updated**: November 25, 2025  
**Datasets Version**: 1.0.0  
**Preprocessing Config**: FinBERT-optimized
