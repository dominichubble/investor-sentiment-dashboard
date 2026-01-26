"""
Main FastAPI application for Investor Sentiment Dashboard.

Provides API endpoints for sentiment analysis and stock entity extraction.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import stocks

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
app.include_router(stocks.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Investor Sentiment Dashboard API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "stocks": "/api/v1/stocks",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "investor-sentiment-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
