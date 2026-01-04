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
