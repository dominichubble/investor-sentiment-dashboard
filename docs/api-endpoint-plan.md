# API Endpoint Plan

**Version**: 1.0  
**Date**: January 2026  
**Status**: Planning

## Overview

This document outlines the planned REST API endpoints for the Investor Sentiment Dashboard. The API will provide access to sentiment analysis, explainability features, and data retrieval capabilities.

## Base URL

```
Development: http://localhost:8000/api/v1
Production: https://api.investor-sentiment.example.com/v1
```

## Authentication

- **Type**: API Key (Bearer token)
- **Header**: `Authorization: Bearer <api_key>`
- **Rate Limiting**: 100 requests/minute per key

---

## 1. Sentiment Analysis Endpoints

### 1.1 Analyze Text Sentiment

**Endpoint**: `POST /sentiment/analyze`

**Description**: Analyze sentiment of provided text using FinBERT model.

**Request Body**:
```json
{
  "text": "Tesla stock surged 15% after strong Q4 earnings",
  "options": {
    "include_scores": true,
    "include_explanation": false
  }
}
```

**Response**:
```json
{
  "text": "Tesla stock surged 15% after strong Q4 earnings",
  "sentiment": {
    "label": "positive",
    "score": 0.94,
    "scores": {
      "positive": 0.94,
      "negative": 0.03,
      "neutral": 0.03
    }
  },
  "metadata": {
    "model": "finbert",
    "processing_time_ms": 45,
    "timestamp": "2026-01-23T10:30:00Z"
  }
}
```

**Use Cases**:
- Real-time sentiment analysis of user-provided text
- Batch processing of social media posts
- News article sentiment classification

---

### 1.2 Batch Sentiment Analysis

**Endpoint**: `POST /sentiment/batch`

**Description**: Analyze sentiment for multiple texts in a single request.

**Request Body**:
```json
{
  "texts": [
    "Markets hit record highs on positive economic data",
    "Tech stocks tumble amid regulatory concerns",
    "Oil prices steady as demand remains stable"
  ],
  "options": {
    "include_scores": true
  }
}
```

**Response**:
```json
{
  "results": [
    {
      "text": "Markets hit record highs on positive economic data",
      "sentiment": {
        "label": "positive",
        "score": 0.96
      }
    },
    // ... more results
  ],
  "metadata": {
    "total_texts": 3,
    "processing_time_ms": 120
  }
}
```

**Rate Limit**: Maximum 100 texts per request

---

### 1.3 Stock Sentiment Analysis

**Endpoint**: `POST /sentiment/stocks`

**Description**: Analyze sentiment with stock entity extraction and pairing.

**Request Body**:
```json
{
  "text": "$AAPL surged 15% while $TSLA dropped 8%",
  "options": {
    "extract_context": true,
    "include_movements": true
  }
}
```

**Response**:
```json
{
  "text": "$AAPL surged 15% while $TSLA dropped 8%",
  "overall_sentiment": {
    "label": "neutral",
    "score": 0.55
  },
  "stocks": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "sentiment": {
        "label": "positive",
        "score": 0.92
      },
      "context": "$AAPL surged 15%",
      "movement": {
        "direction": "up",
        "percentage": 15.0
      }
    },
    {
      "ticker": "TSLA",
      "company_name": "Tesla, Inc.",
      "sentiment": {
        "label": "negative",
        "score": 0.89
      },
      "context": "$TSLA dropped 8%",
      "movement": {
        "direction": "down",
        "percentage": 8.0
      }
    }
  ]
}
```

---

### 1.4 Historical Sentiment

**Endpoint**: `GET /sentiment/history`

**Description**: Retrieve historical sentiment data for a specific stock.

**Query Parameters**:
- `ticker` (required): Stock ticker symbol
- `start_date` (optional): Start date (ISO 8601)
- `end_date` (optional): End date (ISO 8601)
- `source` (optional): Filter by source (twitter, reddit, news)
- `aggregate` (optional): Aggregation level (hourly, daily, weekly)

**Response**:
```json
{
  "ticker": "AAPL",
  "period": {
    "start": "2026-01-01T00:00:00Z",
    "end": "2026-01-23T00:00:00Z"
  },
  "data": [
    {
      "timestamp": "2026-01-01T00:00:00Z",
      "sentiment": {
        "positive": 0.65,
        "negative": 0.20,
        "neutral": 0.15
      },
      "mention_count": 1234,
      "sources": {
        "twitter": 800,
        "reddit": 300,
        "news": 134
      }
    }
    // ... more data points
  ],
  "statistics": {
    "avg_sentiment": 0.62,
    "total_mentions": 45678,
    "sentiment_trend": "positive"
  }
}
```

---

### 1.5 Trending Stocks

**Endpoint**: `GET /sentiment/trending`

**Description**: Get stocks with most mentions and their sentiment.

**Query Parameters**:
- `hours` (optional, default: 24): Time period in hours
- `min_mentions` (optional, default: 10): Minimum mentions threshold
- `limit` (optional, default: 20): Number of results

**Response**:
```json
{
  "trending": [
    {
      "ticker": "NVDA",
      "company_name": "NVIDIA Corporation",
      "mentions": 5678,
      "sentiment": {
        "average": 0.78,
        "distribution": {
          "positive": 0.65,
          "negative": 0.15,
          "neutral": 0.20
        }
      },
      "change_24h": "+245%"
    }
    // ... more stocks
  ],
  "period_hours": 24,
  "total_stocks": 150
}
```

---

## 2. Explainability Endpoints

### 2.1 LIME Explanation

**Endpoint**: `POST /explainability/lime`

**Description**: Generate LIME explanation for sentiment prediction.

**Request Body**:
```json
{
  "text": "Markets crashed following disappointing employment figures",
  "options": {
    "num_features": 10,
    "num_samples": 5000,
    "format": "json"
  }
}
```

**Response**:
```json
{
  "text": "Markets crashed following disappointing employment figures",
  "prediction": {
    "label": "negative",
    "score": 0.94,
    "predicted_class": 1
  },
  "explanation": {
    "features": [
      {
        "word": "crashed",
        "weight": 0.3245,
        "direction": "negative"
      },
      {
        "word": "disappointing",
        "weight": 0.2187,
        "direction": "negative"
      },
      {
        "word": "markets",
        "weight": -0.0523,
        "direction": "neutral"
      }
      // ... more features
    ],
    "local_prediction": 0.89,
    "model_confidence": 0.94
  },
  "metadata": {
    "num_features": 10,
    "num_samples": 5000,
    "processing_time_ms": 3200
  }
}
```

---

### 2.2 SHAP Explanation

**Endpoint**: `POST /explainability/shap`

**Description**: Generate SHAP explanation for sentiment prediction.

**Request Body**:
```json
{
  "text": "Tech stocks rally on strong earnings reports",
  "options": {
    "visualization": false,
    "background_samples": 100
  }
}
```

**Response**:
```json
{
  "text": "Tech stocks rally on strong earnings reports",
  "prediction": {
    "label": "positive",
    "score": 0.91
  },
  "shap_values": {
    "tokens": [
      {"token": "tech", "value": 0.05},
      {"token": "stocks", "value": 0.12},
      {"token": "rally", "value": 0.34},
      {"token": "strong", "value": 0.28},
      {"token": "earnings", "value": 0.15}
    ],
    "base_value": 0.33,
    "output_value": 0.91
  },
  "feature_importance": {
    "most_positive": ["rally", "strong", "earnings"],
    "most_negative": []
  }
}
```

---

### 2.3 Compare Explanations

**Endpoint**: `POST /explainability/compare`

**Description**: Compare LIME and SHAP explanations side-by-side.

**Request Body**:
```json
{
  "text": "Stock prices surged to record highs",
  "methods": ["lime", "shap"]
}
```

**Response**:
```json
{
  "text": "Stock prices surged to record highs",
  "prediction": {
    "label": "positive",
    "score": 0.89
  },
  "explanations": {
    "lime": {
      "top_features": [
        {"word": "surged", "weight": 0.42},
        {"word": "record", "weight": 0.31}
      ]
    },
    "shap": {
      "top_features": [
        {"token": "surged", "value": 0.38},
        {"token": "record", "value": 0.29}
      ]
    }
  },
  "agreement": {
    "correlation": 0.87,
    "top_features_overlap": ["surged", "record"]
  }
}
```

---

### 2.4 Export Explanation

**Endpoint**: `GET /explainability/export/{explanation_id}`

**Description**: Export explanation as HTML or JSON.

**Query Parameters**:
- `format` (optional, default: html): Output format (html, json, pdf)
- `include_visualization` (optional, default: true): Include charts

**Response**:
- `Content-Type: text/html` or `application/json`
- HTML/JSON file download

---

## 3. Data Retrieval Endpoints

### 3.1 Get News Data

**Endpoint**: `GET /data/news`

**Description**: Retrieve processed news articles with sentiment.

**Query Parameters**:
- `start_date` (optional): Start date
- `end_date` (optional): End date
- `ticker` (optional): Filter by stock ticker
- `source` (optional): News source filter
- `sentiment` (optional): Filter by sentiment (positive, negative, neutral)
- `limit` (optional, default: 50): Results per page
- `offset` (optional, default: 0): Pagination offset

**Response**:
```json
{
  "articles": [
    {
      "id": "news_123456",
      "title": "Apple Reports Record Q4 Earnings",
      "source": "Reuters",
      "url": "https://...",
      "published_at": "2026-01-23T09:00:00Z",
      "sentiment": {
        "label": "positive",
        "score": 0.92
      },
      "stocks_mentioned": ["AAPL"],
      "snippet": "Apple Inc. reported record fourth-quarter earnings..."
    }
    // ... more articles
  ],
  "pagination": {
    "total": 1234,
    "limit": 50,
    "offset": 0,
    "has_more": true
  }
}
```

---

### 3.2 Get Reddit Data

**Endpoint**: `GET /data/reddit`

**Description**: Retrieve Reddit posts with sentiment analysis.

**Query Parameters**:
- `subreddit` (optional): Filter by subreddit
- `start_date` (optional): Start date
- `end_date` (optional): End date
- `ticker` (optional): Filter by stock ticker
- `min_score` (optional): Minimum post score
- `limit` (optional, default: 50): Results per page

**Response**:
```json
{
  "posts": [
    {
      "id": "reddit_abc123",
      "subreddit": "wallstreetbets",
      "title": "$TSLA to the moon! 🚀",
      "author": "user123",
      "score": 2345,
      "num_comments": 567,
      "created_at": "2026-01-23T10:15:00Z",
      "sentiment": {
        "label": "positive",
        "score": 0.88
      },
      "stocks_mentioned": ["TSLA"],
      "url": "https://reddit.com/..."
    }
    // ... more posts
  ],
  "pagination": {
    "total": 890,
    "limit": 50,
    "has_more": true
  }
}
```

---

### 3.3 Get Twitter Data

**Endpoint**: `GET /data/twitter`

**Description**: Retrieve tweets with sentiment analysis.

**Query Parameters**:
- `start_date` (optional): Start date
- `end_date` (optional): End date
- `ticker` (optional): Filter by stock ticker
- `verified_only` (optional, default: false): Only verified accounts
- `min_likes` (optional): Minimum like count
- `limit` (optional, default: 50): Results per page

**Response**:
```json
{
  "tweets": [
    {
      "id": "tweet_xyz789",
      "text": "Breaking: $AAPL announces new product line",
      "author": {
        "username": "TechNews",
        "verified": true,
        "followers": 1000000
      },
      "metrics": {
        "likes": 5432,
        "retweets": 1234,
        "replies": 567
      },
      "created_at": "2026-01-23T11:30:00Z",
      "sentiment": {
        "label": "neutral",
        "score": 0.55
      },
      "stocks_mentioned": ["AAPL"]
    }
    // ... more tweets
  ],
  "pagination": {
    "total": 2345,
    "limit": 50,
    "has_more": true
  }
}
```

---

### 3.4 Get Stock Database

**Endpoint**: `GET /data/stocks`

**Description**: Query the stock database.

**Query Parameters**:
- `query` (optional): Search query (ticker or name)
- `exchange` (optional): Filter by exchange
- `limit` (optional, default: 20): Results limit

**Response**:
```json
{
  "stocks": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "exchange": "NASDAQ",
      "sector": "Technology",
      "is_active": true
    }
    // ... more stocks
  ],
  "total_results": 42
}
```

---

### 3.5 Data Statistics

**Endpoint**: `GET /data/statistics`

**Description**: Get overall data statistics and metrics.

**Response**:
```json
{
  "overview": {
    "total_articles": 125678,
    "total_reddit_posts": 234567,
    "total_tweets": 456789,
    "total_stocks_tracked": 10301,
    "last_updated": "2026-01-23T12:00:00Z"
  },
  "date_range": {
    "earliest": "2025-11-25T00:00:00Z",
    "latest": "2026-01-23T12:00:00Z"
  },
  "sentiment_distribution": {
    "positive": 0.45,
    "negative": 0.30,
    "neutral": 0.25
  },
  "top_mentioned_stocks": [
    {"ticker": "AAPL", "mentions": 12345},
    {"ticker": "TSLA", "mentions": 10234}
  ]
}
```

---

## 4. Health & Status Endpoints

### 4.1 Health Check

**Endpoint**: `GET /health`

**Description**: Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-23T12:00:00Z",
  "services": {
    "database": "up",
    "finbert_model": "up",
    "cache": "up"
  }
}
```

---

### 4.2 API Information

**Endpoint**: `GET /info`

**Description**: Get API version and capabilities.

**Response**:
```json
{
  "version": "1.0.0",
  "name": "Investor Sentiment Dashboard API",
  "capabilities": [
    "sentiment_analysis",
    "stock_entity_extraction",
    "lime_explanation",
    "shap_explanation",
    "historical_data"
  ],
  "models": {
    "sentiment": "ProsusAI/finbert",
    "ner": "en_core_web_sm"
  },
  "rate_limits": {
    "default": "100/minute",
    "batch": "10/minute"
  }
}
```

---

## 5. Error Responses

All endpoints follow consistent error format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Text is required",
    "details": {
      "field": "text",
      "issue": "missing_required_field"
    },
    "timestamp": "2026-01-23T12:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Missing or invalid API key |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `MODEL_ERROR` | 500 | ML model processing error |
| `SERVER_ERROR` | 500 | Internal server error |
| `NOT_FOUND` | 404 | Resource not found |

---

## 6. Implementation Priority

### Phase 1: Core Sentiment (✅ Partially Implemented)
- [x] POST /sentiment/stocks (implemented in FYP-203)
- [ ] POST /sentiment/analyze
- [ ] POST /sentiment/batch
- [ ] GET /health

### Phase 2: Explainability
- [ ] POST /explainability/lime
- [ ] POST /explainability/shap
- [ ] POST /explainability/compare

### Phase 3: Data Retrieval
- [ ] GET /data/news
- [ ] GET /data/reddit
- [ ] GET /data/twitter
- [ ] GET /data/statistics

### Phase 4: Advanced Features
- [ ] GET /sentiment/history
- [ ] GET /sentiment/trending
- [ ] GET /data/stocks
- [ ] GET /explainability/export

---

## 7. Security Considerations

### Authentication
- API key based authentication
- JWT tokens for user sessions
- OAuth2 for third-party integrations

### Rate Limiting
- Per-key rate limits
- Endpoint-specific limits
- Burst allowance for authenticated users

### Data Privacy
- No PII storage
- Anonymized analytics
- GDPR compliance for EU users

### Input Validation
- Text length limits (max 5000 characters)
- SQL injection prevention
- XSS protection
- Request size limits

---

## 8. Performance Targets

| Metric | Target |
|--------|--------|
| Response time (p95) | < 500ms |
| Response time (p99) | < 1000ms |
| Throughput | 1000 req/sec |
| Uptime | 99.9% |
| Model inference | < 100ms |

---

## 9. Monitoring & Logging

### Metrics to Track
- Request count per endpoint
- Response times
- Error rates
- Model inference times
- Cache hit rates

### Logging
- Request/response logging
- Error stack traces
- Model prediction logs
- Audit trail for data access

---

## 10. Documentation

### API Documentation Tools
- **Swagger/OpenAPI**: Interactive API documentation
- **Postman Collection**: Pre-configured requests
- **Code Examples**: Python, JavaScript, cURL

### Endpoints to Document
- Request/response schemas
- Authentication flows
- Error handling
- Rate limiting
- Example use cases

---

## Next Steps

1. Review and approve API plan
2. Create OpenAPI specification
3. Set up FastAPI project structure
4. Implement Phase 1 endpoints
5. Add authentication middleware
6. Set up monitoring and logging
7. Write API documentation
8. Create client libraries

---

## References

- FastAPI Documentation: https://fastapi.tiangolo.com/
- OpenAPI Specification: https://swagger.io/specification/
- REST API Best Practices: https://restfulapi.net/
