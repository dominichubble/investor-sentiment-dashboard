# Preprocessing Module

This module provides text preprocessing utilities for sentiment analysis on financial text data.

## Components

### 1. FinBERT Tokenizer (`finbert_tokenizer.py`)

Specialized tokenization for FinBERT inference with proper padding, truncation, and attention masks.

#### Quick Start

```python
from app.preprocessing import tokenize_for_inference, tokenize_batch

# Single text tokenization
inputs = tokenize_for_inference("Stock prices are rising")
# Returns: {'input_ids': tensor([[...]], 'attention_mask': tensor([[...]]}

# Batch tokenization
texts = ["Market up 5%", "Stock crash", "Neutral trading"]
inputs = tokenize_batch(texts)
# Returns: {'input_ids': tensor([[...], [...], [...]]), 'attention_mask': ...}
```

#### Functions

##### `tokenize_for_inference(text, max_length=512, padding=True, truncation=True)`

Tokenize a single text for FinBERT inference.

**Parameters:**
- `text` (str): Raw text string to tokenize
- `max_length` (int): Maximum sequence length (default: 512)
- `padding` (bool|str): Padding strategy. `True` pads to max_length, `'longest'` pads to longest in batch
- `truncation` (bool): Whether to truncate sequences longer than max_length
- `return_tensors` (str): Format for returned tensors (`'pt'` for PyTorch)
- `add_special_tokens` (bool): Whether to add [CLS] and [SEP] tokens

**Returns:**
- Dict with `input_ids` (Tensor) and `attention_mask` (Tensor)

**Example:**
```python
inputs = tokenize_for_inference(
    "Apple stock surged after earnings",
    max_length=128,
    padding='max_length'
)
print(inputs['input_ids'].shape)  # torch.Size([1, 128])
print(inputs['attention_mask'].sum())  # Number of non-padding tokens
```

##### `tokenize_batch(texts, max_length=512, padding=True, batch_size=None)`

Tokenize multiple texts efficiently with batch processing.

**Parameters:**
- `texts` (List[str]): List of raw text strings to tokenize
- `max_length` (int): Maximum sequence length
- `padding` (bool|str): Padding strategy (`True`, `'longest'`, `'max_length'`)
- `batch_size` (int|None): Process in mini-batches for very large inputs
- Other parameters same as `tokenize_for_inference`

**Returns:**
- Dict with batched `input_ids` (Tensor) and `attention_mask` (Tensor)

**Example:**
```python
texts = [
    "Stock market rallied today",
    "Bearish sentiment dominates",
    "Neutral outlook for Q4"
]

# Pad to longest sequence in batch
inputs = tokenize_batch(texts, padding='longest')
print(inputs['input_ids'].shape)  # torch.Size([3, seq_len])

# Process large batches in chunks
large_texts = ["Text " + str(i) for i in range(1000)]
inputs = tokenize_batch(large_texts, batch_size=32)
```

##### `get_tokenizer(cache_dir=None)`

Load and cache the FinBERT tokenizer.

**Returns:**
- `AutoTokenizer` instance for FinBERT

**Example:**
```python
from app.preprocessing import get_tokenizer

tokenizer = get_tokenizer()
print(tokenizer.vocab_size)  # 30522
print(tokenizer.model_max_length)  # 512
```

##### `get_tokenizer_info(cache_dir=None)`

Get metadata about the FinBERT tokenizer.

**Returns:**
- Dict with tokenizer information (vocab_size, max_length, special tokens)

**Example:**
```python
info = get_tokenizer_info()
print(info['model_name'])  # "ProsusAI/finbert"
print(info['vocab_size'])  # 30522
print(info['pad_token'])  # "[PAD]"
```

##### `clear_tokenizer_cache()`

Clear the cached tokenizer instance (useful for testing).

---

### 2. Text Processor (`text_processor.py`)

General text preprocessing for data cleaning and normalization.

#### Quick Start

```python
from app.preprocessing import preprocess_text, TextProcessor

# Simple preprocessing
clean_text = preprocess_text(
    "Check out $AAPL stock! 🚀 http://example.com",
    remove_urls=True,
    remove_stopwords_flag=True
)
# Returns: "check aapl stock"

# Custom preprocessing pipeline
processor = TextProcessor(
    lowercase=True,
    remove_stopwords=True,
    lemmatize=True
)
tokens = processor.process("Markets are bullish today")
# Returns: ["market", "bullish", "today"]
```

#### Functions

##### `preprocess_text(text, **options)`

Quick preprocessing with common options.

**Parameters:**
- `text` (str): Raw text to preprocess
- `lowercase` (bool): Convert to lowercase
- `remove_urls` (bool): Remove URLs
- `remove_mentions` (bool): Remove @mentions
- `remove_stopwords_flag` (bool): Remove stopwords (preserving financial terms)
- `lemmatize_flag` (bool): Lemmatize words

**Returns:**
- Preprocessed text string

##### `extract_tickers(text)`

Extract stock ticker symbols from text.

**Example:**
```python
text = "Buying $AAPL and $TSLA today"
tickers = extract_tickers(text)
# Returns: {'AAPL', 'TSLA'}
```

##### `detect_stock_movements(text)`

Detect mentions of stock price movements.

**Example:**
```python
text = "Stock rose 5% after earnings"
movements = detect_stock_movements(text)
# Returns: ["rose 5%"]
```

---

## Integration with Inference

The FinBERT tokenizer integrates seamlessly with the sentiment inference module:

```python
from app.preprocessing import tokenize_batch
from app.models import analyze_batch

# Collect raw texts
texts = [
    "Market sentiment is positive today",
    "Stocks plummeted amid crisis fears",
    "Trading volume remains stable"
]

# Method 1: Use high-level inference API (handles tokenization internally)
results = analyze_batch(texts)

# Method 2: Manual tokenization for custom pipelines
inputs = tokenize_batch(texts, padding='longest')
# Pass inputs to model directly if needed
# model_outputs = model(**inputs)
```

---

## Testing

Run the test suite for preprocessing modules:

```bash
# Test FinBERT tokenizer
pytest backend/tests/preprocessing/test_finbert_tokenizer.py -v

# Test text processor
pytest backend/tests/preprocessing/test_preprocessing.py -v

# Run all preprocessing tests
pytest backend/tests/preprocessing/ -v
```

---

## Technical Details

### FinBERT Tokenization Specifications

- **Model**: `ProsusAI/finbert`
- **Tokenizer**: BERT WordPiece tokenizer
- **Vocabulary Size**: 30,522 tokens
- **Max Sequence Length**: 512 tokens
- **Special Tokens**:
  - `[CLS]` (101): Classification token (start of sequence)
  - `[SEP]` (102): Separator token (end of sequence)
  - `[PAD]` (0): Padding token
  - `[UNK]` (100): Unknown token
  - `[MASK]` (103): Mask token (for MLM)

### Attention Masks

Attention masks indicate which tokens are real vs. padding:
- `1`: Real token (model should attend to it)
- `0`: Padding token (model should ignore it)

**Example:**
```python
inputs = tokenize_for_inference("Bull market", padding='max_length', max_length=10)
# input_ids: [101, 7087, 2668, 102, 0, 0, 0, 0, 0, 0]
#             [CLS] bull market [SEP] [PAD] [PAD] ...
# attention_mask: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
#                 Real tokens  Padding
```

### Padding Strategies

- `padding=True`: Pad to longest sequence in batch (most efficient)
- `padding='max_length'`: Pad all to `max_length` (uniform size)
- `padding='longest'`: Same as `True`
- `padding=False`: No padding (sequences will have different lengths)

### Performance Optimization

For large batches, use the `batch_size` parameter to process in chunks:

```python
# Process 10,000 texts in batches of 100
large_texts = ["Text " + str(i) for i in range(10000)]
inputs = tokenize_batch(large_texts, batch_size=100)
```

This prevents memory overflow and allows progress tracking.

---

## Best Practices

1. **Use batch tokenization for multiple texts** - Much faster than tokenizing individually
2. **Use `padding='longest'` for inference** - More efficient than `'max_length'`
3. **Set appropriate `max_length`** - 512 is max, but shorter sequences save memory
4. **Cache tokenizer** - Tokenizer is cached automatically after first load
5. **Clear cache in tests** - Use `clear_tokenizer_cache()` to reset between tests
6. **Preserve financial terms** - Text processor preserves domain-specific vocabulary
7. **Handle errors gracefully** - All functions raise descriptive errors for invalid inputs

---

## Examples

### Example 1: Preprocess and Tokenize for Inference

```python
from app.preprocessing import preprocess_text, tokenize_for_inference

# Raw text from social media
raw_text = "@user Check out $AAPL!! 🚀🚀 http://link.com #stocks"

# Step 1: Clean text (remove noise)
clean_text = preprocess_text(
    raw_text,
    remove_urls=True,
    remove_mentions=True,
    lowercase=True
)
# Result: "check out aapl stocks"

# Step 2: Tokenize for FinBERT
inputs = tokenize_for_inference(clean_text)
# Ready for model inference
```

### Example 2: Batch Processing Pipeline

```python
from app.preprocessing import tokenize_batch
from app.models import get_model

# Load model once
model = get_model()

# Prepare batch of texts
texts = [
    "Market bullish on tech stocks",
    "Economic downturn expected",
    "Federal Reserve maintains rates"
]

# Tokenize batch
inputs = tokenize_batch(texts, padding='longest')

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Process results
for text, pred in zip(texts, predictions):
    label_idx = pred.argmax().item()
    confidence = pred[label_idx].item()
    print(f"{text} → {labels[label_idx]} ({confidence:.2f})")
```

### Example 3: Real-time Streaming Processing

```python
from app.preprocessing import get_tokenizer, tokenize_for_inference

# Load tokenizer once for reuse
tokenizer = get_tokenizer()

def process_stream(text_stream):
    """Process streaming text data efficiently."""
    for text in text_stream:
        try:
            # Tokenize each incoming text
            inputs = tokenize_for_inference(text)
            # Process with model...
            yield inputs
        except ValueError as e:
            # Skip invalid texts
            print(f"Skipping invalid text: {e}")
            continue
```

---

## Troubleshooting

### Issue: "Text input cannot be empty"
**Solution**: Ensure text is non-empty and not whitespace-only before tokenization.

### Issue: "Tokenizer not found"
**Solution**: Ensure `transformers` library is installed: `pip install transformers`

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or max_length:
```python
inputs = tokenize_batch(texts, batch_size=16, max_length=256)
```

### Issue: "Tokenizer returns different results"
**Solution**: Clear cache between tests:
```python
from app.preprocessing import clear_tokenizer_cache
clear_tokenizer_cache()
```

---

## Dependencies

- `transformers>=4.0.0` - Hugging Face Transformers for tokenizer
- `torch>=1.9.0` - PyTorch for tensor operations
- `nltk>=3.6` - Natural Language Toolkit for text processing

Install all dependencies:
```bash
pip install -r backend/requirements.txt
```
