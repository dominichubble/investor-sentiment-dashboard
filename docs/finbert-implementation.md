# FinBERT Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [What is FinBERT?](#what-is-finbert)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Module Breakdown](#module-breakdown)
6. [Usage Guide](#usage-guide)
7. [Error Handling](#error-handling)
8. [Performance Optimization](#performance-optimization)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements **FinBERT** (Financial BERT), a pre-trained NLP model specifically fine-tuned for financial sentiment analysis. Unlike generic sentiment models, FinBERT understands financial terminology, market jargon, and the nuanced language used in financial texts.

### Key Features
- ✅ Financial domain-specific sentiment analysis
- ✅ Three-class classification: Positive, Negative, Neutral
- ✅ Confidence scores for predictions
- ✅ Batch processing for efficiency
- ✅ Comprehensive error handling
- ✅ CPU and GPU support
- ✅ Model caching for faster subsequent runs

---

## What is FinBERT?

### Background

**FinBERT** is a BERT (Bidirectional Encoder Representations from Transformers) model that has been:
1. Pre-trained on general English text (by Google)
2. Further pre-trained on financial text corpus (TRC2-financial)
3. Fine-tuned on Financial PhraseBank dataset for sentiment classification

**Model Details:**
- **Base Model:** BERT-base (110M parameters)
- **Vocabulary Size:** 30,522 tokens
- **Max Sequence Length:** 512 tokens
- **Training Data:** 
  - Reuters TRC2 financial corpus (~1.8M articles)
  - Financial PhraseBank (4,840 sentences with sentiment labels)
- **Output:** 3 classes (positive, negative, neutral)

### Why FinBERT?

Traditional sentiment analyzers fail on financial text because:

```python
# Generic sentiment analyzer would be WRONG:
"The company reported a loss of $1M"  # Actually NEUTRAL (factual reporting)
"Stock prices dropped 5%"              # Could be POSITIVE (buying opportunity)
"Earnings beat expectations"           # Clearly POSITIVE (financial context)
```

FinBERT understands:
- Financial terminology (EPS, EBITDA, market cap, etc.)
- Numeric context (percentages, dollar amounts)
- Industry-specific expressions
- Market sentiment vs factual reporting

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           sentiment_inference.py (API Layer)            │
│  • analyze_sentiment(text) → result                     │
│  • analyze_batch(texts) → results, failures             │
│  • analyze_with_metadata(text, meta) → enriched_result  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│          finbert_model.py (Model Wrapper)               │
│  • get_model() → FinBERTModel                          │
│  • predict(text) → sentiment                           │
│  • predict_batch(texts) → sentiments                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│       finbert_tokenizer.py (Text Processing)            │
│  • get_tokenizer() → AutoTokenizer                     │
│  • tokenize_for_inference(text) → tensors             │
│  • tokenize_batch(texts) → batched_tensors            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           HuggingFace Transformers Library              │
│  • AutoModel.from_pretrained("ProsusAI/finbert")       │
│  • AutoTokenizer.from_pretrained("ProsusAI/finbert")   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  PyTorch Framework                       │
│  • Tensor operations                                    │
│  • Model inference                                      │
│  • GPU/CPU management                                   │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input Text
    │
    ├─► Validation (empty check, type check)
    │
    ├─► Tokenization
    │   ├─► Text → Token IDs [101, 2025, 3445, ..., 102]
    │   ├─► Add special tokens: [CLS], [SEP]
    │   ├─► Padding/Truncation to 512 tokens
    │   └─► Attention masks (1 for real tokens, 0 for padding)
    │
    ├─► Model Inference
    │   ├─► Input tensors → FinBERT model
    │   ├─► BERT encoder (12 layers, 768 hidden units)
    │   ├─► Classification head
    │   └─► Logits → [negative, neutral, positive]
    │
    ├─► Post-processing
    │   ├─► Softmax → probabilities [0.1, 0.2, 0.7]
    │   ├─► Argmax → predicted class (positive)
    │   └─► Confidence score (0.7)
    │
    └─► Output
        {
            "label": "positive",
            "score": 0.7,
            "scores": {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        }
```

---

## Implementation Details

### 1. Model Loading (`finbert_model.py`)

#### Singleton Pattern with Global Cache

```python
_model_cache: Optional[FinBERTModel] = None

def get_model(cache_dir: Optional[str] = None) -> FinBERTModel:
    """
    Lazy loading with caching:
    - First call: Downloads model (~440MB), loads into memory
    - Subsequent calls: Returns cached instance (instant)
    """
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache  # Instant return
    
    # Download and load (only happens once)
    _model_cache = FinBERTModel(cache_dir=cache_dir)
    return _model_cache
```

**Why Singleton?**
- Model is 440MB - don't want multiple copies in memory
- Loading takes 10-30 seconds - cache for performance
- Thread-safe for concurrent requests

#### Model Initialization

```python
class FinBERTModel:
    def __init__(self, model_name: str = "ProsusAI/finbert", cache_dir: Optional[str] = None):
        # Load tokenizer (fast, ~30MB)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True  # Rust-based, 10x faster
        )
        
        # Load model (slow first time, ~440MB)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            num_labels=3  # positive, negative, neutral
        )
        
        # Set to evaluation mode (disable dropout)
        self.model.eval()
        
        # Device management (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
```

**Model Files Downloaded:**
```
~/.cache/huggingface/transformers/
├── config.json          (1 KB)   - Model configuration
├── pytorch_model.bin    (440 MB) - Model weights
├── tokenizer_config.json (1 KB)  - Tokenizer settings
├── vocab.txt            (230 KB) - Vocabulary
└── special_tokens_map.json (1 KB) - Special tokens
```

### 2. Tokenization (`finbert_tokenizer.py`)

#### Token Processing Pipeline

**Step 1: Text to Tokens**
```python
Input:  "Stock prices surged 10% today"

# Tokenization
Tokens: ["[CLS]", "stock", "prices", "surged", "10", "%", "today", "[SEP]"]

# Convert to IDs
Token IDs: [101, 4518, 7597, 14527, 1275, 1003, 2651, 102]

# Attention Mask (1 = real token, 0 = padding)
Attention: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ..., 0]  # Padded to 512
```

**Step 2: Special Tokens**
- `[CLS]` (ID: 101): Classification token - used for final prediction
- `[SEP]` (ID: 102): Separator token - marks end of sequence
- `[PAD]` (ID: 0): Padding token - fills to max_length
- `[UNK]` (ID: 100): Unknown token - for out-of-vocabulary words

**Step 3: Truncation & Padding**
```python
# Text longer than 512 tokens → Truncate
"Very long text..." (1000 tokens) → First 510 tokens + [CLS] + [SEP]

# Text shorter than 512 → Pad
"Short text" (5 tokens) → 5 tokens + 507 [PAD] tokens
```

#### Batch Processing

```python
def tokenize_batch(texts: List[str]) -> Dict[str, torch.Tensor]:
    """
    Efficient batch tokenization:
    - Processes multiple texts at once
    - Pads to longest in batch (not max_length)
    - Reduces memory usage
    """
    inputs = tokenizer(
        texts,
        padding='longest',    # Pad to longest in batch
        truncation=True,
        max_length=512,
        return_tensors='pt'   # Return PyTorch tensors
    )
    return inputs
```

**Memory Optimization:**
```python
# Inefficient: Pad all to 512
Batch of 10 texts → 10 × 512 = 5,120 tokens

# Efficient: Pad to longest (e.g., 128)
Batch of 10 texts → 10 × 128 = 1,280 tokens  # 75% memory savings!
```

### 3. Model Inference (`finbert_model.py`)

#### Single Text Prediction

```python
def predict(self, text: str, return_all_scores: bool = False) -> Dict:
    """
    1. Tokenize text
    2. Move tensors to device (CPU/GPU)
    3. Forward pass through model
    4. Apply softmax to get probabilities
    5. Return prediction
    """
    
    # Tokenize
    inputs = self.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(self.device)
    
    # Inference (no gradient computation)
    with torch.no_grad():
        outputs = self.model(**inputs)
        logits = outputs.logits
    
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get prediction
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    result = {
        "label": label_map[predicted_class],
        "score": confidence
    }
    
    if return_all_scores:
        result["scores"] = {
            "negative": probabilities[0][0].item(),
            "neutral": probabilities[0][1].item(),
            "positive": probabilities[0][2].item()
        }
    
    return result
```

#### Batch Prediction

```python
def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
    """
    Process texts in batches for efficiency:
    - Batch 1: texts[0:32]   → Process → Results 0-31
    - Batch 2: texts[32:64]  → Process → Results 32-63
    - ...
    
    GPU utilization: Higher batch size = better GPU usage
    Memory trade-off: Too large = OOM error
    """
    
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest"  # Memory efficient
        ).to(self.device)
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Process each result in batch
        for probs in probabilities:
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item()
            
            all_results.append({
                "label": label_map[predicted_class],
                "score": confidence
            })
    
    return all_results
```

### 4. High-Level API (`sentiment_inference.py`)

#### Error Handling & Recovery

```python
def analyze_batch(
    texts: List[str],
    batch_size: int = 32,
    skip_errors: bool = True,
    track_failures: bool = True
) -> tuple[List[Optional[Dict]], Optional[FailedItemsTracker]]:
    """
    Multi-layer error handling:
    
    Layer 1: Validation (empty texts, invalid types)
    Layer 2: Batch processing (try entire batch)
    Layer 3: Individual fallback (if batch fails, retry one-by-one)
    Layer 4: Failure tracking (record what failed and why)
    """
    
    failures = FailedItemsTracker() if track_failures else None
    results = []
    
    # Layer 1: Validation
    for i, text in enumerate(texts):
        if not text or not text.strip():
            failures.add_failure(
                item=f"Index {i}",
                error_type="EmptyText",
                error_message="Text is empty or whitespace"
            )
            results.append(None)
            continue
    
    # Layer 2: Batch processing
    try:
        batch_results = model.predict_batch(valid_texts, batch_size=batch_size)
        results.extend(batch_results)
    except Exception as batch_error:
        # Layer 3: Individual fallback
        if skip_errors:
            for idx, text in enumerate(valid_texts):
                try:
                    result = analyze_sentiment(text)
                    results.append(result)
                except Exception as item_error:
                    # Layer 4: Track failure
                    failures.add_failure(
                        item=f"Index {idx}: {text[:200]}",
                        error_type=type(item_error).__name__,
                        error_message=str(item_error),
                        additional_info={"text_length": len(text)}
                    )
                    results.append(None)
        else:
            raise
    
    return results, failures
```

---

## Module Breakdown

### `finbert_model.py` - Core Model Logic

**Responsibilities:**
- Model initialization and caching
- Device management (CPU/GPU)
- Single and batch predictions
- Confidence score calculation

**Key Classes:**
```python
class FinBERTModel:
    """Main model wrapper for FinBERT sentiment analysis."""
    
    def __init__(self, model_name: str, cache_dir: Optional[str])
    def predict(self, text: str, return_all_scores: bool) -> Dict
    def predict_batch(self, texts: List[str], batch_size: int) -> List[Dict]
    def get_device_info(self) -> Dict
```

**Key Functions:**
```python
def get_model(cache_dir: Optional[str] = None) -> FinBERTModel
def clear_model_cache() -> None
```

### `finbert_tokenizer.py` - Text Preprocessing

**Responsibilities:**
- Text tokenization
- Special token handling
- Padding and truncation
- Attention mask creation

**Key Functions:**
```python
def get_tokenizer(cache_dir: Optional[str] = None) -> AutoTokenizer
def tokenize_for_inference(text: str, max_length: int = 512) -> Dict[str, torch.Tensor]
def tokenize_batch(texts: List[str], batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]
def get_tokenizer_info(cache_dir: Optional[str] = None) -> Dict
def clear_tokenizer_cache() -> None
```

### `sentiment_inference.py` - High-Level API

**Responsibilities:**
- User-friendly API
- Input validation
- Error handling and recovery
- Failure tracking
- Batch processing orchestration

**Key Functions:**
```python
def analyze_sentiment(text: str, return_all_scores: bool = False) -> Dict
def analyze_batch(
    texts: List[str],
    batch_size: int = 32,
    return_all_scores: bool = False,
    skip_errors: bool = True,
    track_failures: bool = True
) -> tuple[List[Optional[Dict]], Optional[FailedItemsTracker]]
def analyze_with_metadata(text: str, metadata: Dict = None) -> Dict
def get_sentiment_summary(results: List[Dict]) -> Dict
```

### `logging_config.py` - Logging & Error Tracking

**Responsibilities:**
- Centralized logging configuration
- Failed items tracking
- Exception logging utilities

**Key Classes:**
```python
class FailedItemsTracker:
    """Track and save failed items during processing."""
    
    def add_failure(self, item: str, error_type: str, error_message: str, additional_info: Optional[dict])
    def save(self, output_file: Path) -> int
    def count(self) -> int
    def clear(self) -> None
```

---

## Usage Guide

### Basic Usage

```python
from app.models import analyze_sentiment

# Single text
result = analyze_sentiment("Stock prices surged today")
print(result)
# Output: {'label': 'positive', 'score': 0.92}
```

### Batch Processing

```python
from app.models import analyze_batch

texts = [
    "Market rally continues with strong gains",
    "Company reports massive losses",
    "Trading volumes remain stable"
]

results, failures = analyze_batch(texts, batch_size=32)

for text, result in zip(texts, results):
    if result:
        print(f"{result['label']:8s} ({result['score']:.2f}): {text}")
    else:
        print(f"FAILED: {text}")

# Output:
# positive (0.92): Market rally continues with strong gains
# negative (0.88): Company reports massive losses
# neutral  (0.76): Trading volumes remain stable
```

### With Confidence Scores

```python
result = analyze_sentiment("Earnings beat expectations", return_all_scores=True)

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['score']:.2%}")
print("\nAll scores:")
for label, score in result['scores'].items():
    print(f"  {label:8s}: {score:.2%}")

# Output:
# Prediction: positive
# Confidence: 94.50%
#
# All scores:
#   negative: 2.30%
#   neutral : 3.20%
#   positive: 94.50%
```

### Error Handling

```python
from app.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Process with error tracking
texts = ["Good text", "", "Another text", "   "]  # Contains empty texts

results, failures = analyze_batch(texts, skip_errors=True, track_failures=True)

# Save failed items
if failures and failures.count() > 0:
    failures.save("data/failed_items/failed.json")
    logger.warning(f"Failed to process {failures.count()} items")

# Process successful results
for result in results:
    if result:  # Check for None (failed items)
        print(f"{result['label']}: {result['score']:.2f}")
```

### Production Pipeline

```python
from pathlib import Path
from app.logging_config import setup_logging, get_logger
from app.models import analyze_batch
from app.storage import save_predictions_batch

# Setup
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Load data
texts = load_your_data()  # Your data loading function

# Analyze
logger.info(f"Processing {len(texts)} texts")
results, failures = analyze_batch(texts, batch_size=32, skip_errors=True)

# Save failures
if failures and failures.count() > 0:
    failed_file = Path("data/failed_items/failed.json")
    failures.save(failed_file)
    logger.warning(f"Saved {failures.count()} failed items")

# Prepare predictions
predictions = []
for text, result in zip(texts, results):
    if result:
        predictions.append({
            "text": text[:500],
            "source": "your_source",
            "timestamp": datetime.utcnow().isoformat(),
            "label": result["label"],
            "confidence": result["score"]
        })

# Save predictions
output_file = Path("data/predictions/predictions.csv")
save_predictions_batch(predictions, output_file, format="csv")
logger.info(f"Saved {len(predictions)} predictions")

# Summary statistics
from app.models import get_sentiment_summary
summary = get_sentiment_summary(results)
logger.info(f"Summary: {summary}")
```

---

## Error Handling

### Custom Exceptions

```python
class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis errors."""
    pass

class TokenizationError(SentimentAnalysisError):
    """Exception raised when text tokenization fails."""
    pass

class ModelInferenceError(SentimentAnalysisError):
    """Exception raised when model inference fails."""
    pass
```

### Error Scenarios & Handling

#### 1. Empty Text
```python
try:
    analyze_sentiment("")
except ValueError as e:
    print(f"Validation error: {e}")
    # Output: Validation error: Text input cannot be blank
```

#### 2. Very Long Text
```python
long_text = "a" * 50000

try:
    analyze_sentiment(long_text)
except TokenizationError as e:
    print(f"Tokenization failed: {e}")
    # Logs: WARNING - Text is very long (50000 chars)
```

#### 3. Model Loading Failure
```python
try:
    model = get_model(cache_dir="/invalid/path")
except RuntimeError as e:
    print(f"Model loading failed: {e}")
    # Output: Model loading failed: Failed to load FinBERT tokenizer...
```

#### 4. Batch Processing Failure
```python
texts = ["text1", "text2", "text3"]

# Graceful degradation
results, failures = analyze_batch(texts, skip_errors=True)

# Count failures
failed_count = sum(1 for r in results if r is None)
success_count = sum(1 for r in results if r is not None)

print(f"Success: {success_count}, Failed: {failed_count}")

# Save failures for later analysis
if failures:
    failures.save("data/failed_items/failures.json")
```

### Failure Tracking

```json
{
  "failed_items.json structure": [
    {
      "item": "Text content (truncated to 500 chars)",
      "error_type": "TokenizationError",
      "error_message": "Text is too long (52341 chars)",
      "timestamp": "2026-01-05T14:03:38.957881",
      "additional_info": {
        "text_length": 52341,
        "batch_index": 42
      }
    }
  ]
}
```

---

## Performance Optimization

### Batch Size Tuning

#### CPU Performance
```python
# Small batch (8): Slower overall, less memory
# Good for: Limited RAM, testing

# Medium batch (32): Balanced
# Good for: General use, production

# Large batch (64+): Faster, more memory
# Good for: High-RAM machines, offline processing
```

#### GPU Performance
```python
# GPU memory check
import torch
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {total_memory:.1f} GB")
    
    # Batch size recommendations:
    # 4 GB GPU → batch_size=16
    # 8 GB GPU → batch_size=32
    # 16 GB GPU → batch_size=64
    # 24 GB GPU → batch_size=128
```

### Memory Management

```python
# Clear cache between runs
import torch
import gc

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Use in production
results, failures = analyze_batch(large_dataset, batch_size=32)
clear_gpu_memory()  # Free memory
```

### Processing Speed Estimates

**On CPU (Intel i7)**
- Single text: ~50ms
- 100 texts (batch_size=32): ~5 seconds
- 1,000 texts (batch_size=32): ~45 seconds
- 10,000 texts (batch_size=32): ~8 minutes

**On GPU (NVIDIA RTX 3080)**
- Single text: ~10ms
- 100 texts (batch_size=32): ~1 second
- 1,000 texts (batch_size=64): ~8 seconds
- 10,000 texts (batch_size=64): ~75 seconds

**First Run Overhead**
- Model download: 2-5 minutes (440MB, one-time)
- Model loading: 10-30 seconds (every script start)

### Optimization Tips

```python
# 1. Use batch processing
# BAD: Process one at a time
for text in texts:
    result = analyze_sentiment(text)  # Slow!

# GOOD: Process in batches
results, _ = analyze_batch(texts, batch_size=32)  # Much faster!

# 2. Reuse model instance
# BAD: Reload model every time
def process():
    model = get_model()  # Reloads every call!
    return model.predict(text)

# GOOD: Load once, use many times
model = get_model()  # Load once
for text in texts:
    result = model.predict(text)  # Reuse

# 3. Optimal padding
# BAD: Pad all to max_length
inputs = tokenizer(texts, padding='max_length', max_length=512)

# GOOD: Pad to longest in batch
inputs = tokenizer(texts, padding='longest', max_length=512)

# 4. GPU utilization
# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

---

## Configuration

### Environment Variables

```bash
# .env file
FINBERT_CACHE_DIR=/path/to/cache  # Model cache location
FINBERT_DEVICE=cuda               # Force device (cuda/cpu)
FINBERT_BATCH_SIZE=32             # Default batch size
LOG_LEVEL=INFO                    # Logging level
```

### Model Configuration

```python
# Custom model path or version
model = FinBERTModel(
    model_name="ProsusAI/finbert",  # Default
    cache_dir="/custom/cache/path"
)

# Alternative models
# model_name="yiyanghkust/finbert-tone"  # Alternative FinBERT
# model_name="bert-base-uncased"          # Generic BERT (not recommended)
```

### Logging Configuration

```python
from app.logging_config import setup_logging

# Basic setup
setup_logging()

# Custom configuration
setup_logging(
    log_dir=Path("logs"),
    log_level="DEBUG",      # DEBUG, INFO, WARNING, ERROR
    console_level="INFO",   # Console output level
    file_level="DEBUG"      # File output level
)
```

---

## Troubleshooting

### Common Issues

#### 1. Model Download Fails

**Symptom:**
```
RuntimeError: Failed to load FinBERT tokenizer: Connection timeout
```

**Solution:**
```python
# Option 1: Manual download
# Download from https://huggingface.co/ProsusAI/finbert
# Place in ~/.cache/huggingface/transformers/

# Option 2: Use proxy
import os
os.environ['HTTP_PROXY'] = 'http://proxy:port'
os.environ['HTTPS_PROXY'] = 'http://proxy:port'

# Option 3: Use offline mode (after manual download)
model = FinBERTModel(cache_dir="/path/to/downloaded/model")
```

#### 2. Out of Memory (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Reduce batch size
results, failures = analyze_batch(texts, batch_size=16)  # Was 64

# Clear cache between batches
torch.cuda.empty_cache()

# Use CPU instead
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
```

#### 3. Slow Performance

**Symptom:**
Processing is taking too long

**Solutions:**
```python
# 1. Check device
model = get_model()
print(model.device)  # Should be "cuda" if GPU available

# 2. Increase batch size (if memory allows)
results, failures = analyze_batch(texts, batch_size=64)

# 3. Use fast tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "ProsusAI/finbert",
    use_fast=True  # Should be True
)

# 4. Profile your code
import time
start = time.time()
results, _ = analyze_batch(texts)
print(f"Processed {len(texts)} texts in {time.time()-start:.2f}s")
```

#### 4. Incorrect Predictions

**Symptom:**
Model returns unexpected sentiments

**Diagnostics:**
```python
# Check confidence scores
result = analyze_sentiment(text, return_all_scores=True)
print(result['scores'])
# If scores are close (e.g., 0.33, 0.34, 0.33), prediction is uncertain

# Check text length
print(f"Text length: {len(text)} chars")
# Very short (<10 chars) or very long (>2000 chars) may be problematic

# Check for financial context
# FinBERT works best on financial text
# Generic text may not classify well
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Will show:
# - Tokenization details
# - Model loading steps
# - Batch processing progress
# - Error stack traces

results, failures = analyze_batch(texts)
```

### Performance Profiling

```python
import time
from app.models import analyze_batch

def profile_inference(texts, batch_sizes=[8, 16, 32, 64]):
    """Profile different batch sizes."""
    for bs in batch_sizes:
        start = time.time()
        results, _ = analyze_batch(texts, batch_size=bs)
        duration = time.time() - start
        
        throughput = len(texts) / duration
        print(f"Batch Size {bs:3d}: {duration:.2f}s ({throughput:.1f} texts/sec)")

# Run profiling
texts = load_your_data()
profile_inference(texts)

# Example output:
# Batch Size   8: 45.32s (22.1 texts/sec)
# Batch Size  16: 38.45s (26.0 texts/sec)
# Batch Size  32: 35.12s (28.5 texts/sec)  ← Optimal
# Batch Size  64: 34.89s (28.7 texts/sec)  ← Diminishing returns
```

---

## Appendix

### Model Architecture Details

```
FinBERT Model Architecture:

Input Layer
  └─► BERT Tokenizer (30,522 vocab)
        └─► Input IDs [batch_size, 512]
        └─► Attention Mask [batch_size, 512]

BERT Encoder (12 layers)
  ├─► Layer 1: Multi-Head Self-Attention (12 heads, 768 dim)
  ├─► Layer 2: Multi-Head Self-Attention
  │    ...
  └─► Layer 12: Multi-Head Self-Attention
        └─► Output: [batch_size, 512, 768]

Classification Head
  └─► Take [CLS] token representation [batch_size, 768]
        └─► Linear layer (768 → 3)
              └─► Output logits [batch_size, 3]

Softmax
  └─► Convert logits to probabilities
        └─► Output: [negative, neutral, positive]
```

### References

- **FinBERT Paper:** [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)
- **HuggingFace Model:** [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- **BERT Paper:** [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **Transformers Library:** [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

### Version Information

```python
# Check versions
import torch
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

**Last Updated:** January 5, 2026  
**Project:** Investor Sentiment Dashboard  
**Module:** FinBERT Sentiment Analysis
