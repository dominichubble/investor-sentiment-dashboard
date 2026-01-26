"""
Main FastAPI application for Investor Sentiment Dashboard.

Provides API endpoints for sentiment analysis and stock entity extraction.
"""

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import sentiment, stocks

# Create FastAPI app
app = FastAPI(
    title="Investor Sentiment Dashboard API",
    description="API for analyzing investor sentiment on stocks from social media and news sources",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sentiment.router, prefix="/api/v1")
app.include_router(stocks.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Investor Sentiment Dashboard API",
        "version": "1.0.0",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
        },
        "endpoints": {
            "sentiment": "/api/v1/sentiment",
            "stocks": "/api/v1/stocks",
            "health": "/health",
            "info": "/info",
        },
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns the health status of the API and its dependencies.

    **Response:**
    - `status`: Overall health status (healthy/degraded/unhealthy)
    - `version`: API version
    - `timestamp`: Current server time
    - `services`: Status of individual services
    """
    try:
        # Check if model can be loaded
        from app.models.finbert_model import FinBERTModel

        model_status = "up"
        try:
            model = FinBERTModel()
            model_status = "up"
        except Exception:
            model_status = "down"

        # Check if storage is accessible
        storage_status = "up"
        try:
            from app.storage import StockSentimentStorage

            storage = StockSentimentStorage()
            storage_status = "up"
        except Exception:
            storage_status = "down"

        # Determine overall status
        overall_status = "healthy"
        if model_status == "down" or storage_status == "down":
            overall_status = "degraded"

        return {
            "status": overall_status,
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "services": {
                "finbert_model": model_status,
                "storage": storage_status,
                "api": "up",
            },
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(e),
        }


@app.get("/info")
async def api_info():
    """
    Get API version and capabilities.

    Returns detailed information about the API including available features,
    models, and rate limits.

    **Response:**
    - `version`: API version
    - `name`: API name
    - `capabilities`: List of available features
    - `models`: Information about ML models used
    - `rate_limits`: Rate limiting information
    """
    return {
        "version": "1.0.0",
        "name": "Investor Sentiment Dashboard API",
        "description": "Financial sentiment analysis API with stock entity extraction",
        "capabilities": [
            "sentiment_analysis",
            "batch_sentiment_analysis",
            "stock_entity_extraction",
            "context_aware_sentiment",
            "stock_database_lookup",
            "sentiment_storage",
        ],
        "models": {
            "sentiment": {
                "name": "FinBERT",
                "source": "ProsusAI/finbert",
                "type": "transformer",
                "classes": ["positive", "negative", "neutral"],
            },
            "ner": {
                "name": "spaCy",
                "model": "en_core_web_sm",
                "type": "statistical",
            },
        },
        "data": {
            "stock_database": {
                "source": "SEC EDGAR",
                "stocks": "10,000+",
                "coverage": "US markets",
            }
        },
        "rate_limits": {
            "default": "100 requests/minute",
            "batch": "10 requests/minute",
            "note": "Rate limits are per API key",
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
