"""
Core sentiment analysis API endpoints.

Provides REST API for text sentiment analysis using FinBERT model.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.finbert_model import FinBERTModel

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# Initialize FinBERT model
model = FinBERTModel()


# Request/Response Models
class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis."""

    text: str = Field(..., description="Text to analyze", min_length=1, max_length=5000)
    options: Optional[dict] = Field(
        None,
        description="Analysis options",
        json_schema_extra={"example": {"include_scores": True, "include_explanation": False}},
    )


class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment analysis."""

    texts: List[str] = Field(
        ...,
        description="List of texts to analyze",
        min_length=1,
        max_length=100,
    )
    options: Optional[dict] = Field(None, description="Analysis options")


class SentimentScores(BaseModel):
    """Detailed sentiment scores."""

    positive: float = Field(..., description="Positive score (0-1)")
    negative: float = Field(..., description="Negative score (0-1)")
    neutral: float = Field(..., description="Neutral score (0-1)")


class SentimentResult(BaseModel):
    """Sentiment analysis result."""

    label: str = Field(..., description="Sentiment label (positive/negative/neutral)")
    score: float = Field(..., description="Confidence score (0-1)")
    scores: Optional[SentimentScores] = Field(
        None, description="Detailed sentiment scores"
    )


class SentimentMetadata(BaseModel):
    """Metadata for sentiment analysis."""

    model: str = Field(default="finbert", description="Model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Analysis timestamp (ISO 8601)")


class SentimentAnalysisResponse(BaseModel):
    """Response model for sentiment analysis."""

    text: str = Field(..., description="Analyzed text")
    sentiment: SentimentResult = Field(..., description="Sentiment analysis result")
    metadata: SentimentMetadata = Field(..., description="Analysis metadata")


class BatchSentimentResult(BaseModel):
    """Single result in batch analysis."""

    text: str = Field(..., description="Analyzed text")
    sentiment: SentimentResult = Field(..., description="Sentiment analysis result")


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis."""

    results: List[BatchSentimentResult] = Field(..., description="Analysis results")
    metadata: SentimentMetadata = Field(..., description="Analysis metadata")


# Endpoints


@router.post("/analyze")
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
) -> SentimentAnalysisResponse:
    """
    Analyze sentiment of provided text using FinBERT model.

    This endpoint provides real-time sentiment classification with confidence scores.

    **Use Cases:**
    - Real-time sentiment analysis of user-provided text
    - Social media post classification
    - News article sentiment analysis

    **Example:**
    ```json
    {
      "text": "Tesla stock surged 15% after strong Q4 earnings",
      "options": {"include_scores": true}
    }
    ```

    **Response:**
    - `label`: Sentiment classification (positive/negative/neutral)
    - `score`: Model confidence (0-1, higher is more confident)
    - `scores`: Detailed scores for each class (if requested)
    """
    import time

    try:
        start_time = time.time()

        # Get options
        include_scores = (
            request.options.get("include_scores", True) if request.options else True
        )

        # Analyze sentiment with all scores if requested
        result = model.predict(request.text, return_all_scores=include_scores)

        # Build response
        sentiment_scores = None
        if include_scores and "scores" in result:
            scores = result["scores"]
            sentiment_scores = SentimentScores(
                positive=scores.get("positive", 0.0),
                negative=scores.get("negative", 0.0),
                neutral=scores.get("neutral", 0.0),
            )

        processing_time = (time.time() - start_time) * 1000

        return SentimentAnalysisResponse(
            text=request.text,
            sentiment=SentimentResult(
                label=result["label"],
                score=result["score"],
                scores=sentiment_scores,
            ),
            metadata=SentimentMetadata(
                model="finbert",
                processing_time_ms=round(processing_time, 2),
                timestamp=datetime.utcnow().isoformat() + "Z",
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}",
        )


@router.post("/batch")
async def batch_analyze_sentiment(
    request: BatchSentimentRequest,
) -> BatchSentimentResponse:
    """
    Analyze sentiment for multiple texts in a single request.

    Processes up to 100 texts efficiently using batch inference.

    **Use Cases:**
    - Batch processing of social media posts
    - Analyzing multiple news articles
    - Historical data processing

    **Rate Limit:** Maximum 100 texts per request

    **Example:**
    ```json
    {
      "texts": [
        "Markets hit record highs on positive economic data",
        "Tech stocks tumble amid regulatory concerns"
      ]
    }
    ```
    """
    import time

    try:
        start_time = time.time()

        # Validate batch size
        if len(request.texts) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 texts allowed per batch request",
            )

        # Get options
        include_scores = (
            request.options.get("include_scores", True) if request.options else True
        )

        # Analyze all texts
        results = []
        for text in request.texts:
            result = model.predict(text, return_all_scores=include_scores)

            sentiment_scores = None
            if include_scores and "scores" in result:
                scores = result["scores"]
                sentiment_scores = SentimentScores(
                    positive=scores.get("positive", 0.0),
                    negative=scores.get("negative", 0.0),
                    neutral=scores.get("neutral", 0.0),
                )

            results.append(
                BatchSentimentResult(
                    text=text,
                    sentiment=SentimentResult(
                        label=result["label"],
                        score=result["score"],
                        scores=sentiment_scores,
                    ),
                )
            )

        processing_time = (time.time() - start_time) * 1000

        return BatchSentimentResponse(
            results=results,
            metadata=SentimentMetadata(
                model="finbert",
                processing_time_ms=round(processing_time, 2),
                timestamp=datetime.utcnow().isoformat() + "Z",
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch sentiment analysis failed: {str(e)}",
        )
