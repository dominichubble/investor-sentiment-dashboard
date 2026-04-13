# Backend

Python backend for the Investor Sentiment Dashboard. Handles data ingestion, preprocessing, sentiment analysis, and API services.

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── pipelines/              # Data ingestion and preprocessing scripts
│   │   ├── __init__.py
│   │   ├── ingest_reddit.py    # Reddit data collection
│   │   ├── ingest_twitter.py   # Twitter/X data collection
│   │   ├── ingest_news.py      # News API data collection
│   │   ├── preprocess_data.py  # Text preprocessing pipeline
│   │   └── README.md           # Detailed pipeline documentation
│   └── preprocessing/          # Text processing modules
│       ├── __init__.py
│       └── text_processor.py   # Text cleaning and normalization
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── test_ingest_reddit.py
│   │   ├── test_ingest_twitter.py
│   │   └── test_ingest_news.py
│   └── preprocessing/
│       ├── __init__.py
│       └── test_preprocessing.py
├── pyproject.toml             # Tool configurations (black, pytest, mypy)
├── requirements.txt           # Python dependencies
├── .flake8                    # Linting configuration
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11 or higher
- Virtual environment (recommended)
- API credentials (Reddit, Twitter/X, NewsAPI)

### 2. Installation

```bash
# Navigate to project root
cd investor-sentiment-dashboard

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

**PyTorch and weight loading:** `requirements.txt` requires `torch>=2.6.0` (see [CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434) for the `torch.load` hardening rationale). FinBERT is loaded with `use_safetensors=True` so Hugging Face prefers `model.safetensors` instead of legacy pickle checkpoints. For GPU, install a CUDA build of PyTorch that satisfies `>=2.6` from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=investor-sentiment-dashboard/0.1 by your_username

# Twitter/X API
TWITTER_BEARER_TOKEN=your_bearer_token

# News API
NEWS_API_KEY=your_newsapi_key
```

### 4. Download NLTK Data (for preprocessing)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 📊 Data Pipelines

### Reddit Ingestion

```bash
cd backend/app/pipelines

# Basic usage (default settings)
python ingest_reddit.py

# Custom parameters
python ingest_reddit.py \
  --subreddits wallstreetbets stocks \
  --time-filter week \
  --limit 500
```

### Twitter/X Ingestion

```bash
cd backend/app/pipelines

# Basic usage
python ingest_twitter.py

# Custom search query and date range
python ingest_twitter.py \
  --query "stock market OR crypto" \
  --max-results 200
```

### News Ingestion

```bash
cd backend/app/pipelines

# Basic usage
python ingest_news.py

# Custom keywords and sources
python ingest_news.py \
  --keywords "stock market" "federal reserve" \
  --sources bloomberg reuters \
  --days 7
```

### Text Preprocessing

```bash
cd backend/app/pipelines

# Preprocess Reddit data
python preprocess_data.py \
  --source reddit \
  --input ../../../data/raw/reddit/2025-11-04 \
  --output ../../../data/processed/reddit/2025-11-04

# Preprocess with custom settings
python preprocess_data.py \
  --source news \
  --input ../../../data/raw/news/2025-11-04 \
  --output ../../../data/processed/news/2025-11-04 \
  --remove-stopwords \
  --lemmatize
```

See [app/pipelines/README.md](app/pipelines/README.md) for detailed documentation.

## 🧪 Testing

### Run All Tests

```bash
cd backend
pytest
```

### Run Specific Test Files

```bash
# Test Reddit ingestion
pytest tests/pipelines/test_ingest_reddit.py

# Test preprocessing
pytest tests/preprocessing/test_preprocessing.py
```

### Run with Coverage

```bash
pytest --cov=app --cov-report=html
```

### Test Configuration

Tests are configured in `pyproject.toml`:
- Test discovery: `tests/` directory
- Naming convention: `test_*.py` files
- Markers: `slow`, `integration`

## 🛠️ Development

### Code Quality Tools

**Black (code formatting):**
```bash
black app/ tests/
```

**isort (import sorting):**
```bash
isort app/ tests/
```

**flake8 (linting):**
```bash
flake8 app/ tests/
```

**mypy (type checking):**
```bash
mypy app/
```

### Configuration Files

- **pyproject.toml** - Black, isort, pytest, coverage, mypy settings
- **.flake8** - Linting rules and exclusions
- **requirements.txt** - Python package dependencies

## 📦 Modules

### `app/pipelines/`

Data ingestion and preprocessing scripts that can be run as standalone CLI tools or imported as modules.

**Key Features:**
- Robust error handling and logging
- CLI argument parsing
- Metadata tracking
- Data validation and cleaning
- Rate limiting and API quota management

### `app/preprocessing/`

Reusable text preprocessing components for financial sentiment analysis.

**TextProcessor Class:**
- URL removal
- Mention/hashtag handling
- Punctuation normalization
- Tokenization (NLTK)
- Stopword removal (with financial term preservation)
- Lemmatization

## 🔐 API Credentials

### Reddit API
1. Visit https://www.reddit.com/prefs/apps
2. Create a "script" type application
3. Copy client ID and secret

### Twitter/X API
1. Apply at https://developer.twitter.com/
2. Create an app
3. Generate Bearer Token
4. Free tier: 1,500 tweets/month

### NewsAPI
1. Register at https://newsapi.org/register
2. Choose "Developer" plan (free)
3. Copy API key from dashboard
4. Free tier: 100 requests/day, 1-month history

## 📚 Documentation

- [Data Pipeline Documentation](../docs/data-pipeline.md)
- [Preprocessing Guide](../docs/preprocessing-guide.md)
- [Pipeline README](app/pipelines/README.md)

## 🐛 Troubleshooting

**Import Errors:**
```bash
# Ensure you're in the project root and virtual environment is activated
cd investor-sentiment-dashboard
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac

# Install in editable mode
pip install -e backend/
```

**NLTK Data Missing:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

**API Rate Limits:**
- Check API quota in your developer dashboard
- Reduce `--limit` or `--max-results` parameters
- Add delays between requests (already implemented)

## 📝 License

See [LICENSE](../LICENSE) in project root.
