"""Sentiment endpoints for API v1."""

from datetime import datetime
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.explainability.lime_explainer import get_lime_explainer
from app.models.sentiment_inference import analyze_batch as run_batch_sentiment
from app.models.sentiment_inference import analyze_sentiment as run_sentiment

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


class SentimentScores(BaseModel):
    """Optional full score breakdown."""

    positive: float
    negative: float
    neutral: float


class SentimentResult(BaseModel):
    """Single sentiment result payload."""

    label: str
    score: float
    scores: SentimentScores | None = None


class AnalyzeRequest(BaseModel):
    """Single-text analysis request."""

    text: str | None = Field(default=None)
    options: dict[str, Any] | None = Field(default=None)


class AnalyzeResponse(BaseModel):
    """Single-text analysis response."""

    text: str
    sentiment: SentimentResult
    metadata: dict[str, Any]


class BatchRequest(BaseModel):
    """Batch analysis request."""

    texts: list[str] = Field(default_factory=list)
    options: dict[str, Any] | None = Field(default=None)


class BatchItemResponse(BaseModel):
    """Batch item response."""

    text: str
    sentiment: SentimentResult


class BatchResponse(BaseModel):
    """Batch analysis response."""

    results: list[BatchItemResponse]
    metadata: dict[str, Any]


class ExplainRequest(BaseModel):
    """LIME explainability request payload."""

    text: str | None = Field(default=None)
    options: dict[str, Any] | None = Field(default=None)


class ExplanationToken(BaseModel):
    """Token-level LIME weight."""

    token: str
    weight: float


class ExplainResponse(BaseModel):
    """LIME explainability response payload."""

    text: str
    prediction: SentimentResult
    tokens: list[ExplanationToken]
    metadata: dict[str, Any]


@router.get("/_ping")
async def ping_sentiment() -> dict[str, str]:
    """Temporary v1 sentiment route proving router mount."""
    return {"status": "ok"}


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def run_lime_explain(
    text: str,
    num_features: int = 12,
    num_samples: int = 1000,
) -> dict[str, Any]:
    """Run LIME explanation using shared explainer singleton."""
    explainer = get_lime_explainer()
    return explainer.explain(
        text=text, num_features=num_features, num_samples=num_samples
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_sentiment(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze sentiment for a single text using FinBERT."""
    text = request.text if isinstance(request.text, str) else ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must be a non-empty string")

    include_scores = bool((request.options or {}).get("include_scores", False))
    started = perf_counter()
    try:
        result = run_sentiment(text, return_all_scores=include_scores)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"sentiment analysis failed: {exc}"
        ) from exc

    scores_payload = None
    if include_scores and isinstance(result.get("scores"), dict):
        scores_payload = SentimentScores(
            positive=float(result["scores"].get("positive", 0.0)),
            negative=float(result["scores"].get("negative", 0.0)),
            neutral=float(result["scores"].get("neutral", 0.0)),
        )

    return AnalyzeResponse(
        text=text,
        sentiment=SentimentResult(
            label=str(result.get("label", "neutral")),
            score=float(result.get("score", 0.0)),
            scores=scores_payload,
        ),
        metadata={
            "model": "finbert",
            "processing_time_ms": round((perf_counter() - started) * 1000, 2),
            "timestamp": _utc_now_iso(),
        },
    )


@router.post("/batch", response_model=BatchResponse)
async def batch_analyze_sentiment(request: BatchRequest) -> BatchResponse:
    """Analyze sentiment for multiple texts, preserving input order."""
    if not request.texts:
        raise HTTPException(
            status_code=400, detail="texts must contain at least one item"
        )
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400, detail="maximum 100 texts allowed per request"
        )

    include_scores = bool((request.options or {}).get("include_scores", False))
    started = perf_counter()
    try:
        batch_results = run_batch_sentiment(
            request.texts,
            return_all_scores=include_scores,
            skip_errors=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"batch sentiment analysis failed: {exc}"
        ) from exc

    normalized_results: list[BatchItemResponse] = []
    for text, result in zip(request.texts, batch_results):
        if result is None:
            raise HTTPException(
                status_code=500,
                detail="batch sentiment analysis returned null result",
            )
        scores_payload = None
        if include_scores and isinstance(result.get("scores"), dict):
            scores_payload = SentimentScores(
                positive=float(result["scores"].get("positive", 0.0)),
                negative=float(result["scores"].get("negative", 0.0)),
                neutral=float(result["scores"].get("neutral", 0.0)),
            )
        normalized_results.append(
            BatchItemResponse(
                text=text,
                sentiment=SentimentResult(
                    label=str(result.get("label", "neutral")),
                    score=float(result.get("score", 0.0)),
                    scores=scores_payload,
                ),
            )
        )

    return BatchResponse(
        results=normalized_results,
        metadata={
            "model": "finbert",
            "processing_time_ms": round((perf_counter() - started) * 1000, 2),
            "timestamp": _utc_now_iso(),
        },
    )


@router.post("/explain", response_model=ExplainResponse)
async def explain_sentiment(request: ExplainRequest) -> ExplainResponse:
    """Generate token-level LIME explanation for a single text."""
    text = request.text if isinstance(request.text, str) else ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must be a non-empty string")

    options = request.options or {}
    num_features = int(options.get("num_features", 12))
    num_samples = int(options.get("num_samples", 1000))

    started = perf_counter()
    try:
        explanation = run_lime_explain(
            text=text,
            num_features=num_features,
            num_samples=num_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"explainability failed: {exc}"
        ) from exc

    prediction = explanation.get("prediction", {})
    token_weights = explanation.get("top_features", [])
    tokens = [
        ExplanationToken(token=str(token), weight=float(weight))
        for token, weight in token_weights
    ]

    scores_payload = None
    if isinstance(prediction.get("scores"), dict):
        scores_payload = SentimentScores(
            positive=float(prediction["scores"].get("positive", 0.0)),
            negative=float(prediction["scores"].get("negative", 0.0)),
            neutral=float(prediction["scores"].get("neutral", 0.0)),
        )

    return ExplainResponse(
        text=text,
        prediction=SentimentResult(
            label=str(prediction.get("label", "neutral")),
            score=float(prediction.get("score", 0.0)),
            scores=scores_payload,
        ),
        tokens=tokens,
        metadata={
            "method": "LIME",
            "num_features": len(tokens),
            "num_samples": num_samples,
            "processing_time_ms": round((perf_counter() - started) * 1000, 2),
            "timestamp": _utc_now_iso(),
        },
    )
