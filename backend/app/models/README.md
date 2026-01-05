# Models

This directory contains machine learning model wrappers for sentiment analysis.

## FinBERT

Financial sentiment analysis using the ProsusAI/finbert model.

### Features

- **Automatic GPU/CPU Detection**: Automatically uses GPU if available, falls back to CPU
- **Model Caching**: Downloads and caches the model locally for faster subsequent loads
- **Batch Processing**: Efficiently process multiple texts at once
- **Singleton Pattern**: Model is loaded once and reused across the application

### Usage

#### Basic Usage

```python
from app.models import FinBERTModel

# Initialize model (loads once, then cached)
model = FinBERTModel()

# Single prediction
text = "The stock market is performing well today"
result = model.predict(text)
print(result)
# Output: {'label': 'positive', 'score': 0.95}

# Get all scores
result = model.predict(text, return_all_scores=True)
print(result)
# Output: {'label': 'positive', 'score': 0.95, 'scores': {'positive': 0.95, 'negative': 0.03, 'neutral': 0.02}}
```

#### Batch Processing

```python
# Multiple texts
texts = [
    "Company reports record earnings",
    "Stock prices fall on bad news",
    "Market remains stable"
]

results = model.predict(texts)
for text, result in zip(texts, results):
    print(f"{text} -> {result['label']}")
```

#### Singleton Pattern (Recommended)

```python
from app.models import get_model

# Get cached model instance
model = get_model()
result = model.predict("Financial text here")
```

### Model Details

- **Model**: `ProsusAI/finbert`
- **Framework**: PyTorch + Transformers
- **Labels**: `positive`, `negative`, `neutral`
- **Max Length**: 512 tokens
- **Device**: Auto-detected (CUDA/CPU)

### Device Information

```python
# Check device configuration
info = model.get_device_info()
print(info)
# Output: {'device': 'cuda', 'device_name': 'NVIDIA GeForce RTX 3080', ...}
```

### Testing

Run the test script to verify setup:

```bash
python backend/tests/models/test_finbert_setup.py
```

Run unit tests:

```bash
pytest backend/tests/models/test_finbert_model.py -v
```

### Performance

- **CPU**: ~200-300ms per text
- **GPU**: ~50-100ms per text (first run slower due to model loading)
- **Batch**: More efficient for multiple texts

### Requirements

```
transformers>=4.35.0
torch>=2.1.0
```

Install with:
```bash
pip install -r backend/requirements.txt
```

### Cache Location

Models are cached by default in:
- Linux/Mac: `~/.cache/huggingface/transformers/`
- Windows: `%USERPROFILE%\.cache\huggingface\transformers\`

First load will download ~400MB. Subsequent loads are instant.

---

## Sentiment Inference API

High-level functions for sentiment analysis with batch processing and metadata tracking.

### Quick Start

```python
from app.models import analyze_sentiment, analyze_batch

# Single text
result = analyze_sentiment("Stock prices surged today")
# {'label': 'positive', 'score': 0.95}

# Batch processing
texts = ["Market up", "Losses reported", "Flat trading"]
results = analyze_batch(texts)
```

### Functions

#### `analyze_sentiment(text, return_all_scores=False)`

Analyze sentiment of a single text input.

```python
result = analyze_sentiment("Market outlook uncertain", return_all_scores=True)
# {
#     'label': 'neutral',
#     'score': 0.78,
#     'scores': {'positive': 0.11, 'negative': 0.11, 'neutral': 0.78}
# }
```

#### `analyze_batch(texts, batch_size=32, return_all_scores=False)`

Efficiently analyze multiple texts.

```python
texts = ["Good news", "Bad news", "Neutral news"]
results = analyze_batch(texts, batch_size=16)
```

#### `analyze_with_metadata(text, metadata=None)`

Analyze sentiment and attach metadata.

```python
result = analyze_with_metadata(
    "Stock volatility increases",
    metadata={'post_id': '123', 'source': 'reddit'}
)
```

#### `get_sentiment_summary(results)`

Generate summary statistics from results.

```python
summary = get_sentiment_summary(results)
# {
#     'total': 100,
#     'counts': {'positive': 40, 'negative': 30, 'neutral': 30},
#     'percentages': {'positive': 40.0, 'negative': 30.0, 'neutral': 30.0},
#     'average_confidence': 0.87
# }
```

### Usage Example: Process Reddit Data

```python
from app.models import analyze_batch, get_sentiment_summary

# Load posts
posts = [
    "TSLA to the moon! 🚀",
    "Market crash coming",
    "Earnings meet expectations"
]

# Analyze
results = analyze_batch(posts)

# Get summary
summary = get_sentiment_summary(results)
print(f"Positive: {summary['percentages']['positive']:.1f}%")
```
