# Analysis Notebooks

Interactive Jupyter notebooks for exploring the investor sentiment analysis system.

## Available Notebooks

### 01. Reddit Ingestion (`01-reddit-ingest.ipynb`)
Demonstrates Reddit data collection using PRAW API.

### 02. Twitter Ingestion (`02-twitter-ingest.ipynb`)
Shows Twitter/X data collection using Tweepy.

### 03. News Ingestion (`03-news-ingest.ipynb`)
Explores news article collection using NewsAPI.

### 04. Text Preprocessing (`04-text-preprocessing.ipynb`)
Covers text cleaning, normalization, and preprocessing techniques.

### 05. Sentiment Analysis (`05-sentiment-analysis.ipynb`)
Demonstrates FinBERT sentiment analysis for financial text.

### 06. SHAP Explainability (`06-shap-explainability.ipynb`)
Shows SHAP-based model explainability for sentiment predictions.

### 07. LIME Explainability (`07-lime-explainability.ipynb`)
Demonstrates LIME local interpretable explanations for FinBERT.

### 08. Stock Entity Sentiment (`08-stock-entity-sentiment.ipynb`) ⭐ NEW
**Complete stock entity extraction and sentiment pairing system (FYP-203).**

Features:
- Extract stocks by ticker symbols ($AAPL) and company names (Apple)
- Analyze sentiment for each stock mention
- Store and query stock sentiment data
- Support for 13,000+ US-listed stocks
- Visualizations and performance analysis

## Quick Start

### Setup Environment

```bash
# Install dependencies
pip install -r ../backend/requirements.txt

# Download spaCy model (for notebook 08)
python -m spacy download en_core_web_sm

# Launch Jupyter
jupyter notebook
```

### Recommended Order

For a complete walkthrough of the system:

1. **Data Collection**: Notebooks 01-03
2. **Preprocessing**: Notebook 04
3. **Sentiment Analysis**: Notebook 05
4. **Explainability**: Notebooks 06-07
5. **Stock Entity Pairing**: Notebook 08 ⭐

## Notebook 08 Highlights

The stock entity sentiment notebook (`08-stock-entity-sentiment.ipynb`) is the most comprehensive and includes:

### 1. Stock Database
- Access 13,000+ US stocks from SEC EDGAR
- Lookup by ticker or company name
- Search functionality

### 2. Entity Extraction
- spaCy NER for company names
- Financial keyword detection
- Context extraction

### 3. Entity Resolution
- Exact and fuzzy matching
- Handle name variations
- Blacklist for false positives

### 4. Sentiment Analysis
- Single stock analysis
- Multiple stocks with different sentiments
- Real-world examples (Reddit, Twitter, News)

### 5. Storage & Querying
- Save stock sentiment data
- Aggregate by ticker
- Trending stocks analysis

### 6. Visualization
- Sentiment comparison charts
- Distribution plots
- Performance benchmarks

### 7. Integration
- Process existing data files
- Batch processing examples
- API integration examples

## Example Usage

```python
from app.stocks import analyze_stock_sentiment

# Analyze text
text = "Apple reported strong earnings. Stock surged 15%."
result = analyze_stock_sentiment(text)

# Access results
for stock in result['stocks']:
    print(f"{stock['ticker']}: {stock['sentiment']['label']}")
```

## Output Examples

**Single Stock:**
```
AAPL: positive (0.92)
Context: Apple reported strong earnings
```

**Multiple Stocks:**
```
AAPL: positive (0.89)
TSLA: negative (0.78)
MSFT: positive (0.85)
```

## Data Sources

Notebooks use data from:
- `data/processed/reddit/` - Reddit posts
- `data/processed/news/` - News articles
- `data/processed/twitter/` - Tweets (when available)

## Tips

- **Restart Kernel**: If you modify backend code, restart the notebook kernel to reload modules
- **Module Reload**: Notebook 08 includes automatic module reloading
- **Performance**: Use `extract_context=False` for faster processing
- **Memory**: For large datasets, process in batches

## Documentation

- **Quick Start**: `../docs/FYP-203-Quick-Start.md`
- **Implementation Guide**: `../docs/FYP-203-Implementation-Guide.md`
- **API Documentation**: `../backend/api/README.md`

## Troubleshooting

**Import Errors:**
```python
import sys
from pathlib import Path
backend_path = str(Path.cwd().parent / "backend")
sys.path.insert(0, backend_path)
```

**spaCy Model Missing:**
```bash
python -m spacy download en_core_web_sm
```

**FinBERT Model Download:**
The model downloads automatically on first use (~450MB).

## Contributing

When adding new notebooks:
1. Follow the naming convention: `##-descriptive-name.ipynb`
2. Include clear markdown explanations
3. Add example outputs
4. Update this README
5. Test with fresh kernel

## Resources

- **Project Repository**: Main investor sentiment dashboard
- **Backend Code**: `../backend/app/`
- **Tests**: `../backend/tests/`
- **Scripts**: `../backend/scripts/`
