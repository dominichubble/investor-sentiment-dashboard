# Investor Sentiment Dashboard API

FastAPI application providing REST API endpoints for stock entity extraction and sentiment analysis.

## Features

- **Stock Entity Extraction**: Detect stocks by ticker symbols and company names
- **Sentiment Analysis**: FinBERT-powered sentiment analysis for each stock mention
- **Storage**: Save and query historical stock sentiment data
- **Aggregation**: Trending stocks, sentiment distribution, time-series queries

## API Endpoints

### Stock Analysis

#### `POST /api/v1/stocks/analyze`
Analyze text and extract stock-sentiment pairs.

**Request:**
```json
{
  "text": "Apple reported strong earnings while Tesla faced delays",
  "extract_context": true,
  "include_movements": true,
  "source": "news"
}
```

**Response:**
```json
{
  "text": "Apple reported strong earnings...",
  "overall_sentiment": {
    "label": "positive",
    "score": 0.89,
    "scores": {"positive": 0.89, "negative": 0.05, "neutral": 0.06}
  },
  "stocks": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "mentioned_as": "Apple",
      "sentiment": {"label": "positive", "score": 0.92},
      "context": "Apple reported strong earnings",
      "position": {"start": 0, "end": 5}
    }
  ],
  "metadata": {
    "entities_found": 2,
    "tickers_extracted": ["AAPL", "TSLA"],
    "processing_time_ms": 45.2
  }
}
```

### Query Stock Sentiment

#### `GET /api/v1/stocks/{ticker}/sentiment`
Get aggregated sentiment for a specific stock.

**Parameters:**
- `start_date` (optional): Filter by start date (YYYY-MM-DD)
- `end_date` (optional): Filter by end date (YYYY-MM-DD)
- `source` (optional): Filter by source (reddit/twitter/news)
- `include_records` (optional): Include individual records

**Response:**
```json
{
  "ticker": "AAPL",
  "total_mentions": 150,
  "average_score": 0.72,
  "sentiment_distribution": {
    "positive": 95,
    "negative": 25,
    "neutral": 30
  }
}
```

#### `GET /api/v1/stocks/{ticker}/mentions`
Get recent mentions of a stock with pagination.

**Parameters:**
- `limit`: Maximum results (default: 50, max: 500)
- `offset`: Pagination offset (default: 0)
- `source`: Filter by source

#### `GET /api/v1/stocks/trending`
Get trending stocks by mention count.

**Parameters:**
- `period`: Time period (24h, 7d, 30d)
- `min_mentions`: Minimum mentions required
- `limit`: Maximum results

**Response:**
```json
{
  "trending": [
    {"ticker": "AAPL", "mentions": 245},
    {"ticker": "TSLA", "mentions": 189}
  ],
  "period_hours": 24,
  "total_stocks": 15
}
```

#### `POST /api/v1/stocks/compare`
Compare sentiment across multiple stocks.

**Query Parameters:**
- `tickers`: List of ticker symbols (repeatable)
- `start_date`: Optional start date
- `end_date`: Optional end date

**Example:**
```
POST /api/v1/stocks/compare?tickers=AAPL&tickers=TSLA&tickers=MSFT
```

#### `GET /api/v1/stocks/statistics`
Get overall statistics.

## Running the API

### Development

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing

```bash
pytest tests/ -v
```

## Dependencies

- **FastAPI**: Modern web framework
- **Pydantic**: Data validation
- **FinBERT**: Sentiment analysis
- **spaCy**: Entity extraction
- **FuzzyWuzzy**: Fuzzy string matching

## Architecture

```
api/
├── main.py           # FastAPI app setup
├── routers/
│   └── stocks.py     # Stock endpoints
└── README.md

app/
├── entities/         # Entity extraction
├── stocks/           # Stock sentiment analysis
├── storage/          # Data persistence
└── models/           # FinBERT model
```

## Rate Limiting

Consider implementing rate limiting for production:
- Per-IP limits
- API key authentication
- Request throttling

## CORS Configuration

Update CORS settings in `main.py` for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Monitoring

Consider adding:
- Request logging
- Error tracking (Sentry)
- Performance monitoring
- Health checks

## Security

- Implement authentication for sensitive endpoints
- Use HTTPS in production
- Validate and sanitize all inputs
- Set appropriate CORS policies
- Rate limiting per user/IP
