# FinBERT Sentiment Analysis

This module provides sentiment analysis for financial text using the [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) model.

## Features

- ✅ **Pre-trained FinBERT model** for financial sentiment analysis
- ✅ **GPU/CPU fallback** - Automatically uses GPU if available, falls back to CPU
- ✅ **Model caching** - Downloads once, loads from cache on subsequent uses
- ✅ **Batch processing** - Efficient processing of multiple texts
- ✅ **Error handling** - Graceful handling of empty texts and errors

## Installation

Install required dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Key packages:
- `transformers>=4.35.0` - Hugging Face Transformers
- `torch>=2.1.0` - PyTorch
- `accelerate>=0.25.0` - Model loading optimization

## Quick Start

### Initialize Model

```python
from app.models.finbert import FinBERTSentiment

# Initialize (downloads and caches model on first run)
finbert = FinBERTSentiment()

# Single prediction
result = finbert.predict("Stock prices soared after earnings beat.")
print(result)
# {'label': 'positive', 'score': 0.95}
```

### Batch Prediction

```python
texts = [
    "Market crashed due to recession fears.",
    "Earnings exceeded expectations.",
    "Stock remained stable throughout trading."
]

results = finbert.predict_batch(texts)
for text, result in zip(texts, results):
    print(f"{text}")
    print(f"  → {result['label']} ({result['score']:.2%})")
```

## Pre-caching the Model

To download and cache the model before first use:

```bash
# Auto-detect device (GPU/CPU)
python -m app.models.init_finbert

# Force CPU
python -m app.models.init_finbert --device cpu

# Force GPU
python -m app.models.init_finbert --device cuda

# Custom cache directory
python -m app.models.init_finbert --cache-dir /path/to/cache
```

## API Reference

### FinBERTSentiment Class

```python
FinBERTSentiment(
    model_name: str = "ProsusAI/finbert",
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None
)
```

**Parameters:**
- `model_name`: Hugging Face model identifier (default: `ProsusAI/finbert`)
- `cache_dir`: Model cache directory (default: `~/.cache/finbert`)
- `device`: Device to use (`'cuda'`, `'cpu'`, or `None` for auto-detect)

### Methods

#### `predict(text: str) -> Dict[str, Union[str, float]]`

Predict sentiment for a single text.

**Returns:**
- `label`: Sentiment label (`'positive'`, `'negative'`, or `'neutral'`)
- `score`: Confidence score (0-1)

**Example:**
```python
result = finbert.predict("Stock market rallied today")
# {'label': 'positive', 'score': 0.94}
```

#### `predict_batch(texts: List[str], batch_size: int = 8) -> List[Dict]`

Predict sentiment for multiple texts efficiently.

**Parameters:**
- `texts`: List of texts to analyze
- `batch_size`: Number of texts to process at once (default: 8)

**Returns:**
List of dictionaries with `label` and `score` keys

**Example:**
```python
texts = ["Bull market continues", "Bear market concerns"]
results = finbert.predict_batch(texts)
```

#### `get_model_info() -> Dict`

Get model metadata and configuration.

**Returns:**
```python
{
    'model_name': 'ProsusAI/finbert',
    'device': 'cpu',
    'cuda_available': False,
    'gpu_name': None,
    'cache_dir': '/home/user/.cache/finbert',
    'max_length': 512
}
```

#### `warm_up(sample_text: str = "...") -> None`

Warm up the model by running a sample prediction. Useful for initializing CUDA kernels.

## GPU/CPU Configuration

The module automatically detects and uses the best available device:

1. **Auto-detection** (default):
   - Uses GPU if CUDA is available
   - Falls back to CPU if GPU fails or unavailable

2. **Explicit device selection**:
   ```python
   # Force CPU
   finbert = FinBERTSentiment(device='cpu')
   
   # Force GPU (with automatic fallback)
   finbert = FinBERTSentiment(device='cuda')
   ```

3. **Fallback behavior**:
   - If GPU loading fails, automatically falls back to CPU
   - Logs all device transitions

## Model Caching

### Default Cache Location

```
~/.cache/finbert/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.txt
└── special_tokens_map.json
```

### Custom Cache Directory

```python
finbert = FinBERTSentiment(cache_dir='/custom/path/to/cache')
```

### Cache Size

The FinBERT model requires approximately **438 MB** of disk space.

## Sentiment Labels

FinBERT classifies financial text into three categories:

- **positive**: Optimistic, bullish sentiment
  - Examples: "earnings beat expectations", "stock soared", "strong growth"

- **negative**: Pessimistic, bearish sentiment
  - Examples: "market crashed", "losses exceeded forecasts", "declining sales"

- **neutral**: Balanced or factual statements
  - Examples: "stock remained stable", "company announced merger"

## Error Handling

The module handles errors gracefully:

```python
# Empty text
result = finbert.predict("")
# Returns: {'label': 'neutral', 'score': 0.0}

# Very long text (auto-truncated to 512 tokens)
long_text = "..." * 1000
result = finbert.predict(long_text)
# Automatically truncates and processes

# Network errors during model loading
# Raises RuntimeError with clear error message
```

## Performance Tips

1. **Batch processing**: Use `predict_batch()` for multiple texts
   ```python
   # Much faster than calling predict() in a loop
   results = finbert.predict_batch(texts, batch_size=16)
   ```

2. **GPU acceleration**: Use GPU for significant speedup
   - CPU: ~100 texts/second
   - GPU: ~1000+ texts/second

3. **Pre-cache model**: Download model before production use
   ```bash
   python -m app.models.init_finbert
   ```

4. **Adjust batch size**: Larger batches for GPU, smaller for CPU
   ```python
   # CPU
   results = finbert.predict_batch(texts, batch_size=8)
   
   # GPU
   results = finbert.predict_batch(texts, batch_size=32)
   ```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/models/test_finbert.py -v

# Skip slow tests (that load the actual model)
pytest tests/models/test_finbert.py -v -m "not slow"

# Only slow tests
pytest tests/models/test_finbert.py -v -m "slow"
```

## Example: Processing Dataset

```python
import json
from pathlib import Path
from app.models.finbert import FinBERTSentiment

# Initialize model
finbert = FinBERTSentiment()

# Load preprocessed data
with open("data/processed/reddit/reddit_finance_2025-11-25.json") as f:
    data = json.load(f)
    records = data.get("data", data)

# Extract cleaned texts
texts = [r["text_cleaned"] for r in records if r.get("text_cleaned")]

# Batch predict (efficient!)
results = finbert.predict_batch(texts, batch_size=16)

# Add predictions to records
for record, result in zip(records, results):
    record["sentiment"] = result["label"]
    record["sentiment_score"] = result["score"]

# Save results
output_path = Path("data/analyzed/reddit_with_sentiment.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump({"data": records}, f, indent=2)

print(f"✓ Analyzed {len(results)} texts")
print(f"✓ Results saved to {output_path}")
```

## Troubleshooting

### Model Download Fails

**Problem**: Network timeout or connection error

**Solution**:
```bash
# Download directly with HF CLI
huggingface-cli download ProsusAI/finbert --cache-dir ~/.cache/finbert
```

### CUDA Out of Memory

**Problem**: GPU runs out of memory during batch processing

**Solution**:
```python
# Reduce batch size
results = finbert.predict_batch(texts, batch_size=4)

# Or force CPU
finbert = FinBERTSentiment(device='cpu')
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**:
```bash
pip install transformers torch accelerate
```

## References

- **Model**: [ProsusAI/finbert on Hugging Face](https://huggingface.co/ProsusAI/finbert)
- **Paper**: [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)
- **Transformers Documentation**: [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## License

FinBERT model is licensed under Apache 2.0. See the [model card](https://huggingface.co/ProsusAI/finbert) for details.
