# Data Pipeline Documentation

## Overview

The Investor Sentiment Dashboard uses a multi-stage data pipeline to collect, preprocess, and analyze financial sentiment from various sources.

## Pipeline Stages

### 1. Data Ingestion

Data is collected from three primary sources:

#### Reddit (`backend/app/ingestion/reddit_ingest.py`)
- Collects posts and comments from financial subreddits
- Filters by keywords: stocks, investing, trading, market, bull, bear, etc.
- Stores raw data in `data/raw/reddit/YYYY-MM-DD/`

#### Twitter (`backend/app/ingestion/twitter_ingest.py`)
- Collects tweets about financial markets and specific stocks
- Uses keywords and hashtags for filtering
- Stores raw data in `data/raw/twitter/YYYY-MM-DD/`

#### News (`backend/app/ingestion/news_ingest.py`)
- Collects financial news articles from News API
- Searches for market-related news and company-specific articles
- Stores raw data in `data/raw/news/YYYY-MM-DD/`

### 2. Text Preprocessing

Text preprocessing prepares raw text for sentiment analysis using FinBERT.

#### Pipeline Script (`backend/app/pipelines/preprocess_data.py`)

**Configurations:**
- `minimal`: Basic cleaning only (lowercase, remove URLs)
- `standard`: Add stopword removal
- `full`: Add lemmatization (traditional NLP)
- `finbert`: **Recommended** - Optimized for transformer models
  - Preserves case (important for entities)
  - No stopword removal (transformers use context)
  - No lemmatization (transformers understand word forms)
  - Preserves financial punctuation (%, $, decimals)
  - Handles negations (marks "not good" as "not_good")

**Usage:**
```bash
# Process all sources with FinBERT config (recommended)
python backend/app/pipelines/preprocess_data.py --source all --config finbert

# Process specific source
python backend/app/pipelines/preprocess_data.py --source reddit --config finbert

# Process specific directory
python backend/app/pipelines/preprocess_data.py --input data/raw/reddit/2025-11-04 --config finbert
```

**Output:** Preprocessed data in `data/processed/{source}/YYYY-MM-DD/`

#### Preprocessing Module (`backend/app/preprocessing/text_processor.py`)

**Key Features:**
- **Financial Term Preservation**: 50+ domain-specific terms (bullish, bearish, rally, crash, etc.)
- **Intensity Modifiers**: Preserves sentiment strength words (very, extremely, highly, etc.)
- **Negation Handling**: Marks negations to prevent sentiment flip (critical for accuracy)
- **Numeric Context**: Preserves %, $, and decimal points in financial context
- **Configurable Pipeline**: Easily customize for different models and use cases

### 3. Sentiment Analysis

Sentiment analysis uses the **ProsusAI/FinBERT** model to classify financial text into positive, negative, or neutral sentiment.

#### FinBERT Module (`backend/app/models/finbert.py`)

**Features:**
- Pre-trained financial sentiment model (438MB cached locally)
- Automatic GPU/CPU detection and fallback
- Batch processing for efficient analysis
- Single and batch prediction methods

**Usage:**
```bash
# Initialize model (first time downloads ~438MB)
cd backend
python -m app.models.init_finbert

# Process datasets
python -c "from app.models.finbert import FinBERTSentiment; finbert = FinBERTSentiment()"
```

**See:** [FinBERT Model Documentation](finbert-model.md) for detailed API reference and examples.

### 4. Dashboard Visualization

*(Coming soon - React dashboard for exploring sentiment trends)*

## Data Flow Diagram

```
┌─────────────────┐
│  Data Sources   │
│  - Reddit       │
│  - Twitter      │
│  - News APIs    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Raw Ingestion  │
│  (JSON files)   │
│    data/raw/    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Processing │
│  - Normalize    │
│  - Tokenize     │
│  - Preserve $%  │
│  - Handle NOT   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessed   │
│  (JSON files)   │
│ data/processed/ │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sentiment Model │
│   (FinBERT)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Dashboard     │
│  (React + D3)   │
└─────────────────┘
```

## Directory Structure

```
data/
├── raw/                 # Raw ingested data
│   ├── reddit/
│   │   └── YYYY-MM-DD/
│   ├── twitter/
│   │   └── YYYY-MM-DD/
│   └── news/
│       └── YYYY-MM-DD/
└── processed/           # Preprocessed data ready for ML
    ├── reddit/
    ├── twitter/
    ├── news/
    └── samples/         # Test samples
```

## Best Practices

1. **Always use the `finbert` config** for sentiment analysis preprocessing
2. **Run preprocessing daily** after ingestion to keep data pipeline current
3. **Monitor output sizes** - preprocessed files should be similar size to input
4. **Check logs** for any errors during processing
5. **Validate samples** using notebooks before full processing

## Related Documentation

- [Preprocessing Guide](preprocessing-guide.md) - Detailed preprocessing configuration
- [FinBERT Model](finbert-model.md) - Sentiment analysis API reference
- [FinBERT Implementation](finbert-implementation.md) - Implementation details
- [Notebooks](../notebooks/README.md) - Exploration and analysis
