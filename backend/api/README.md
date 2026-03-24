# Investor Sentiment Dashboard API

REST API for financial sentiment analysis with explainability and stock entity extraction.

## Features

- **Sentiment Analysis**: Real-time and batch sentiment classification using FinBERT
- **Explainability**: LIME and SHAP explanations for model predictions
- **Stock Entity Extraction**: Identify and track sentiment for specific stocks
- **Historical Data**: Access past predictions and sentiment trends
- **Statistics**: Aggregate metrics and insights

## Quick Start

### Installation

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Install additional API dependencies
pip install fastapi uvicorn python-multipart
```

### Running the Server

```bash
# Development mode (with auto-reload)
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access Documentation

Once the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### Core Sentiment (`/api/v1/sentiment`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Single text sentiment analysis |
| `/batch` | POST | Batch sentiment analysis (up to 100 texts) |

### Explainability (`/api/v1/explainability`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/explain` | POST | LIME explanation for prediction |
| `/batch` | POST | Batch LIME explanations (up to 20 texts) |
| `/shap` | POST | SHAP explanation for prediction |
| `/examples` | GET | Pre-computed example explanations |

### Data Retrieval (`/api/v1/data`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predictions` | GET | Historical predictions with filtering |
| `/predictions/{id}` | GET | Get specific prediction by ID |
| `/stocks/{ticker}/sentiment` | GET | Sentiment history for a stock |
| `/statistics` | GET | Aggregate statistics and metrics |

### Stock Analysis (`/api/v1/stocks`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze text for stock entities and sentiment |
| `/batch` | POST | Batch stock analysis |
| `/ticker/{ticker}` | GET | Get stock information |
| `/search` | GET | Search stocks by name |
| `/trending` | GET | Get trending stocks by sentiment |

### Health & Info

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/info` | GET | API version and capabilities |
| `/` | GET | Root endpoint with navigation |

## Usage Examples

### 1. Analyze Sentiment

```bash
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Tesla stock surged 15% on strong Q4 earnings",
    "options": {"include_scores": true}
  }'
```

**Response:**
```json
{
  "text": "Tesla stock surged 15% on strong Q4 earnings",
  "sentiment": {
    "label": "positive",
    "score": 0.95,
    "scores": {
      "positive": 0.95,
      "negative": 0.02,
      "neutral": 0.03
    }
  },
  "metadata": {
    "model": "finbert",
    "processing_time_ms": 127.5,
    "timestamp": "2026-01-23T10:30:00Z"
  }
}
```

### 2. Get LIME Explanation

```bash
curl -X POST "http://localhost:8000/api/v1/explainability/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Markets crashed on disappointing jobs report",
    "num_features": 5
  }'
```

**Response:**
```json
{
  "text": "Markets crashed on disappointing jobs report",
  "prediction": {
    "label": "negative",
    "score": 0.93,
    "all_scores": {
      "positive": 0.02,
      "negative": 0.93,
      "neutral": 0.05
    }
  },
  "features": [
    {"feature": "crashed", "weight": 0.42},
    {"feature": "disappointing", "weight": 0.31},
    {"feature": "Markets", "weight": 0.15},
    {"feature": "jobs", "weight": 0.08},
    {"feature": "report", "weight": 0.05}
  ],
  "metadata": {
    "method": "LIME",
    "num_features": 5,
    "num_samples": 1000,
    "processing_time_ms": 2341.2,
    "timestamp": "2026-01-23T10:35:00Z"
  }
}
```

### 3. Analyze Stock Mentions

```bash
curl -X POST "http://localhost:8000/api/v1/stocks/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple reports record iPhone sales while Microsoft announces AI breakthrough"
  }'
```

**Response:**
```json
{
  "text": "Apple reports record iPhone sales while Microsoft announces AI breakthrough",
  "overall_sentiment": {
    "label": "positive",
    "score": 0.91
  },
  "stocks": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "sentiment": {
        "label": "positive",
        "score": 0.89
      },
      "context": "Apple reports record iPhone sales",
      "confidence": 0.95
    },
    {
      "ticker": "MSFT",
      "company_name": "Microsoft Corp",
      "sentiment": {
        "label": "positive",
        "score": 0.93
      },
      "context": "Microsoft announces AI breakthrough",
      "confidence": 0.92
    }
  ],
  "metadata": {
    "num_stocks_found": 2,
    "processing_time_ms": 345.7,
    "timestamp": "2026-01-23T10:40:00Z"
  }
}
```

### 4. Get Statistics

```bash
curl "http://localhost:8000/api/v1/data/statistics"
```

**Response:**
```json
{
  "total_predictions": 1523,
  "total_stocks_analyzed": 247,
  "sentiment_distribution": {
    "positive": 612,
    "negative": 534,
    "neutral": 377,
    "positive_percentage": 40.18,
    "negative_percentage": 35.06,
    "neutral_percentage": 24.76
  },
  "top_stocks": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "count": 89,
      "positive": 56,
      "negative": 21,
      "neutral": 12
    }
  ],
  "recent_activity": {
    "last_24h": 45,
    "last_7d": 312,
    "last_30d": 1523
  },
  "date_range": {
    "earliest": "2025-12-23T08:15:00Z",
    "latest": "2026-01-23T10:45:00Z"
  }
}
```

### 5. Filter Historical Predictions

```bash
curl "http://localhost:8000/api/v1/data/predictions?sentiment=positive&page=1&page_size=10"
```

## Rate Limits

- **Default endpoints**: 100 requests/minute
- **Batch endpoints**: 10 requests/minute
- **Explanation endpoints**: 5 requests/minute (computationally expensive)

## Error Responses

All endpoints return consistent error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Status Codes:**
- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `422`: Validation error (invalid input format)
- `429`: Rate limit exceeded
- `500`: Internal server error

## Authentication

Currently, the API is open for development. For production:

```python
# TODO: Implement API key authentication
# Add to headers: {"Authorization": "Bearer YOUR_API_KEY"}
```

## Testing

```bash
# Run all API tests
cd backend
pytest tests/api/ -v

# Run specific test file
pytest tests/api/test_sentiment.py -v

# Run with coverage
pytest tests/api/ --cov=api --cov-report=html
```

## Performance

**Typical Response Times:**
- Sentiment analysis: 50-150ms
- LIME explanation: 1-3 seconds
- SHAP explanation: 2-5 seconds
- Stock entity extraction: 100-300ms
- Data queries: 10-50ms

**Optimization Tips:**
1. Use batch endpoints for multiple texts
2. Cache frequent queries
3. Use pagination for large result sets
4. Consider async processing for explanations

## Development

### Project Structure

```
backend/api/
├── main.py              # FastAPI app initialization
├── routers/             # API route handlers
│   ├── sentiment.py     # Sentiment endpoints
│   ├── explainability.py # Explainability endpoints
│   ├── data.py          # Data retrieval endpoints
│   └── stocks.py        # Stock analysis endpoints
└── README.md            # This file

backend/tests/api/
├── test_sentiment.py
├── test_explainability.py
├── test_data.py
└── test_stocks.py
```

### Adding New Endpoints

1. Create/modify router in `backend/api/routers/`
2. Define Pydantic models for request/response
3. Implement endpoint logic
4. Add tests in `backend/tests/api/`
5. Update this README

### CORS Configuration

For production, update CORS settings in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

COPY backend/ .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

### Cloud Platforms

**AWS Lambda:**
```bash
pip install mangum
# Use Mangum adapter in main.py
```

**Google Cloud Run:**
```bash
gcloud run deploy sentiment-api --source .
```

**Azure Container Instances:**
```bash
az container create --name sentiment-api --image sentiment-api:latest
```

## Monitoring

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check service status
curl http://localhost:8000/info
```

### Logging

Logs are output to stdout/stderr. Configure logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Contributing

1. Follow existing code patterns
2. Add tests for new endpoints
3. Update this README
4. Run linter: `flake8 api/`
5. Format code: `black api/`

## License

See main project LICENSE file.

## Support

- **Documentation**: http://localhost:8000/docs
- **Issues**: See project repository
- **API Plan**: See `docs/api-endpoint-plan.md`
