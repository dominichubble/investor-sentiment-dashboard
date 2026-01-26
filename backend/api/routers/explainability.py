"""
Explainability API endpoints for sentiment analysis.

Provides LIME and SHAP explanations for model predictions.
"""

from datetime import datetime
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.explainability.lime_explainer import LIMEExplainer
from app.explainability.shap_explainer import SHAPExplainer
from app.models.finbert_model import FinBERTModel

router = APIRouter(prefix="/explainability", tags=["explainability"])

# Initialize explainers (they load the model internally)
lime_explainer = LIMEExplainer()
shap_explainer = SHAPExplainer()


# Request/Response Models
class ExplainRequest(BaseModel):
    """Request model for explanation generation."""

    text: str = Field(..., description="Text to explain", min_length=1, max_length=5000)
    num_features: int = Field(
        10, description="Number of top features to return", ge=1, le=50
    )
    num_samples: int = Field(
        1000, description="Number of samples for LIME", ge=100, le=5000
    )


class BatchExplainRequest(BaseModel):
    """Request model for batch explanations."""

    texts: List[str] = Field(
        ...,
        description="List of texts to explain",
        min_length=1,
        max_length=20,
    )
    num_features: int = Field(10, description="Number of top features", ge=1, le=50)
    num_samples: int = Field(1000, description="Number of samples for LIME", ge=100, le=5000)


class SHAPExplainRequest(BaseModel):
    """Request model for SHAP explanation."""

    text: str = Field(..., description="Text to explain", min_length=1, max_length=5000)
    num_features: int = Field(
        10, description="Number of top features to return", ge=1, le=50
    )


class FeatureWeight(BaseModel):
    """Feature importance weight."""

    feature: str = Field(..., description="Feature/word")
    weight: float = Field(..., description="Importance weight")


class SentimentPrediction(BaseModel):
    """Sentiment prediction result."""

    label: str = Field(..., description="Predicted sentiment label")
    score: float = Field(..., description="Confidence score")
    all_scores: dict = Field(..., description="All class probabilities")


class ExplanationMetadata(BaseModel):
    """Metadata for explanation."""

    method: str = Field(..., description="Explanation method (LIME/SHAP)")
    num_features: int = Field(..., description="Number of features returned")
    num_samples: int = Field(..., description="Number of samples used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Explanation timestamp (ISO 8601)")


class ExplanationResponse(BaseModel):
    """Response model for explanation."""

    text: str = Field(..., description="Analyzed text")
    prediction: SentimentPrediction = Field(..., description="Model prediction")
    features: List[FeatureWeight] = Field(..., description="Feature importance weights")
    metadata: ExplanationMetadata = Field(..., description="Explanation metadata")
    html_visualization: Optional[str] = Field(
        None, description="HTML visualization of explanation"
    )


class BatchExplanationResult(BaseModel):
    """Single result in batch explanation."""

    text: str = Field(..., description="Analyzed text")
    prediction: SentimentPrediction = Field(..., description="Model prediction")
    features: List[FeatureWeight] = Field(..., description="Top features")


class BatchExplanationResponse(BaseModel):
    """Response model for batch explanations."""

    results: List[BatchExplanationResult] = Field(..., description="Explanation results")
    metadata: ExplanationMetadata = Field(..., description="Explanation metadata")


class ExampleExplanation(BaseModel):
    """Pre-computed example explanation."""

    id: str = Field(..., description="Example ID")
    text: str = Field(..., description="Example text")
    category: str = Field(..., description="Example category")
    prediction: SentimentPrediction = Field(..., description="Prediction")
    top_features: List[FeatureWeight] = Field(..., description="Top features")


class ExamplesResponse(BaseModel):
    """Response model for example explanations."""

    examples: List[ExampleExplanation] = Field(..., description="Example explanations")
    total: int = Field(..., description="Total number of examples")


# Endpoints


@router.post("/explain")
async def explain_prediction(request: ExplainRequest) -> ExplanationResponse:
    """
    Generate LIME explanation for sentiment prediction.

    Uses Local Interpretable Model-agnostic Explanations (LIME) to explain
    which words/features contributed most to the model's prediction.

    **Use Cases:**
    - Understanding why model made a specific prediction
    - Identifying influential words in financial text
    - Debugging model behavior
    - Building trust in predictions

    **Parameters:**
    - `text`: Text to analyze and explain
    - `num_features`: Number of top features to return (default: 10)
    - `num_samples`: Number of perturbed samples for LIME (default: 1000, more = slower but more accurate)

    **Example:**
    ```json
    {
      "text": "Stock prices surged on strong earnings",
      "num_features": 10,
      "num_samples": 1000
    }
    ```

    **Response:**
    - `features`: List of words with importance weights (positive = supports prediction, negative = opposes)
    - `prediction`: Model's prediction with confidence scores
    - `html_visualization`: Optional HTML visualization of the explanation
    """
    import time

    try:
        start_time = time.time()

        # Generate LIME explanation
        explanation = lime_explainer.explain(
            text=request.text,
            num_features=request.num_features,
            num_samples=request.num_samples,
        )

        # Build response
        features = [
            FeatureWeight(feature=word, weight=weight)
            for word, weight in explanation["features"]
        ]

        prediction = SentimentPrediction(
            label=explanation["prediction"],
            score=explanation["confidence"],
            all_scores=explanation["all_scores"],
        )

        processing_time = (time.time() - start_time) * 1000

        response = ExplanationResponse(
            text=request.text,
            prediction=prediction,
            features=features,
            metadata=ExplanationMetadata(
                method="LIME",
                num_features=len(features),
                num_samples=request.num_samples,
                processing_time_ms=round(processing_time, 2),
                timestamp=datetime.utcnow().isoformat() + "Z",
            ),
            html_visualization=None,  # Can be added if needed
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}",
        )


@router.post("/batch")
async def batch_explain_predictions(
    request: BatchExplainRequest,
) -> BatchExplanationResponse:
    """
    Generate LIME explanations for multiple texts.

    Batch processing of LIME explanations for efficiency.

    **Rate Limit:** Maximum 20 texts per request (due to computational cost)

    **Use Cases:**
    - Analyzing multiple predictions at once
    - Comparing explanations across texts
    - Batch processing historical data

    **Example:**
    ```json
    {
      "texts": [
        "Markets rallied on positive jobs data",
        "Stocks declined amid inflation concerns"
      ],
      "num_features": 5
    }
    ```
    """
    import time

    try:
        start_time = time.time()

        # Validate batch size
        if len(request.texts) > 20:
            raise HTTPException(
                status_code=400,
                detail="Maximum 20 texts allowed per batch request (LIME is computationally expensive)",
            )

        # Generate explanations for all texts
        results = []
        for text in request.texts:
            explanation = lime_explainer.explain(
                text=text,
                num_features=request.num_features,
                num_samples=request.num_samples,
            )

            features = [
                FeatureWeight(feature=word, weight=weight)
                for word, weight in explanation["features"][:request.num_features]
            ]

            prediction = SentimentPrediction(
                label=explanation["prediction"],
                score=explanation["confidence"],
                all_scores=explanation["all_scores"],
            )

            results.append(
                BatchExplanationResult(
                    text=text,
                    prediction=prediction,
                    features=features,
                )
            )

        processing_time = (time.time() - start_time) * 1000

        return BatchExplanationResponse(
            results=results,
            metadata=ExplanationMetadata(
                method="LIME",
                num_features=request.num_features,
                num_samples=request.num_samples,
                processing_time_ms=round(processing_time, 2),
                timestamp=datetime.utcnow().isoformat() + "Z",
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch explanation failed: {str(e)}",
        )


@router.post("/shap")
async def explain_with_shap(request: SHAPExplainRequest) -> ExplanationResponse:
    """
    Generate SHAP explanation for sentiment prediction.

    Uses SHapley Additive exPlanations (SHAP) to explain model predictions.
    SHAP provides game-theory-based feature attributions.

    **Differences from LIME:**
    - SHAP: Global consistency, based on game theory, slower
    - LIME: Local approximation, faster, good for quick explanations

    **Use Cases:**
    - When you need theoretically-grounded explanations
    - Comparing with LIME explanations
    - Research and academic applications

    **Example:**
    ```json
    {
      "text": "Federal Reserve raises interest rates",
      "num_features": 10
    }
    ```
    """
    import time

    try:
        start_time = time.time()

        # Generate SHAP explanation
        explanation = shap_explainer.explain(text=request.text)

        # Build response (limit to requested number of features)
        features = [
            FeatureWeight(feature=word, weight=float(weight))
            for word, weight in explanation["features"][:request.num_features]
        ]

        prediction = SentimentPrediction(
            label=explanation["prediction"],
            score=explanation["confidence"],
            all_scores=explanation["all_scores"],
        )

        processing_time = (time.time() - start_time) * 1000

        response = ExplanationResponse(
            text=request.text,
            prediction=prediction,
            features=features,
            metadata=ExplanationMetadata(
                method="SHAP",
                num_features=len(features),
                num_samples=0,  # SHAP doesn't use sampling
                processing_time_ms=round(processing_time, 2),
                timestamp=datetime.utcnow().isoformat() + "Z",
            ),
            html_visualization=None,
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {str(e)}",
        )


@router.get("/examples")
async def get_example_explanations(
    category: Optional[str] = Query(
        None, description="Filter by category (positive/negative/neutral)"
    ),
    limit: int = Query(10, description="Number of examples to return", ge=1, le=50),
) -> ExamplesResponse:
    """
    Get pre-computed example explanations.

    Returns a set of example texts with their explanations for demonstration purposes.

    **Use Cases:**
    - Testing the API without generating new explanations
    - UI demos and tutorials
    - Understanding explanation format

    **Query Parameters:**
    - `category`: Filter by sentiment (positive/negative/neutral)
    - `limit`: Number of examples to return (default: 10, max: 50)

    **Example:**
    ```
    GET /api/v1/explainability/examples?category=positive&limit=5
    ```
    """
    try:
        # Pre-computed examples
        all_examples = [
            {
                "id": "ex1",
                "text": "Stock prices surged to record highs on strong earnings",
                "category": "positive",
                "prediction": {
                    "label": "positive",
                    "score": 0.95,
                    "all_scores": {"positive": 0.95, "negative": 0.02, "neutral": 0.03},
                },
                "top_features": [
                    {"feature": "surged", "weight": 0.35},
                    {"feature": "record", "weight": 0.28},
                    {"feature": "highs", "weight": 0.22},
                    {"feature": "strong", "weight": 0.18},
                    {"feature": "earnings", "weight": 0.12},
                ],
            },
            {
                "id": "ex2",
                "text": "Markets crashed following disappointing employment figures",
                "category": "negative",
                "prediction": {
                    "label": "negative",
                    "score": 0.93,
                    "all_scores": {"positive": 0.02, "negative": 0.93, "neutral": 0.05},
                },
                "top_features": [
                    {"feature": "crashed", "weight": 0.42},
                    {"feature": "disappointing", "weight": 0.31},
                    {"feature": "figures", "weight": -0.15},
                    {"feature": "employment", "weight": -0.08},
                    {"feature": "Markets", "weight": 0.05},
                ],
            },
            {
                "id": "ex3",
                "text": "The company announced its quarterly earnings report",
                "category": "neutral",
                "prediction": {
                    "label": "neutral",
                    "score": 0.78,
                    "all_scores": {"positive": 0.12, "negative": 0.10, "neutral": 0.78},
                },
                "top_features": [
                    {"feature": "announced", "weight": 0.08},
                    {"feature": "quarterly", "weight": 0.05},
                    {"feature": "earnings", "weight": 0.04},
                    {"feature": "report", "weight": 0.03},
                    {"feature": "company", "weight": 0.02},
                ],
            },
            {
                "id": "ex4",
                "text": "Tech giants reported robust growth and exceeded analyst expectations",
                "category": "positive",
                "prediction": {
                    "label": "positive",
                    "score": 0.91,
                    "all_scores": {"positive": 0.91, "negative": 0.03, "neutral": 0.06},
                },
                "top_features": [
                    {"feature": "exceeded", "weight": 0.38},
                    {"feature": "robust", "weight": 0.29},
                    {"feature": "growth", "weight": 0.24},
                    {"feature": "expectations", "weight": 0.16},
                    {"feature": "giants", "weight": 0.08},
                ],
            },
            {
                "id": "ex5",
                "text": "Shares plummeted amid concerns over regulatory scrutiny",
                "category": "negative",
                "prediction": {
                    "label": "negative",
                    "score": 0.89,
                    "all_scores": {"positive": 0.04, "negative": 0.89, "neutral": 0.07},
                },
                "top_features": [
                    {"feature": "plummeted", "weight": 0.45},
                    {"feature": "concerns", "weight": 0.32},
                    {"feature": "scrutiny", "weight": 0.18},
                    {"feature": "regulatory", "weight": -0.12},
                    {"feature": "Shares", "weight": 0.06},
                ],
            },
        ]

        # Filter by category if specified
        if category:
            filtered = [ex for ex in all_examples if ex["category"] == category.lower()]
        else:
            filtered = all_examples

        # Apply limit
        filtered = filtered[:limit]

        # Convert to response models
        examples = [
            ExampleExplanation(
                id=ex["id"],
                text=ex["text"],
                category=ex["category"],
                prediction=SentimentPrediction(**ex["prediction"]),
                top_features=[FeatureWeight(**fw) for fw in ex["top_features"]],
            )
            for ex in filtered
        ]

        return ExamplesResponse(
            examples=examples,
            total=len(examples),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch examples: {str(e)}",
        )
