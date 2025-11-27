# Text Preprocessing Guide

## Overview

The text preprocessing pipeline is specifically optimized for **financial sentiment analysis** using transformer models like FinBERT.

## Why FinBERT-Optimized Preprocessing?

Traditional NLP preprocessing (aggressive stopword removal, lemmatization) was designed for classical ML models (Naive Bayes, SVM). Modern transformer models like FinBERT benefit from:

1. **Minimal preprocessing** - Let the model learn context
2. **Preserved negations** - "not profitable" ≠ "profitable"
3. **Numeric context** - "up 25%" vs "up 25" (different meanings)
4. **Case sensitivity** - "Apple" (company) vs "apple" (fruit)
5. **Full word forms** - "running" and "run" have different nuances

## Configuration Presets

### FinBERT (Recommended)

```python
config = {
    "lowercase": False,  # Preserve entity names
    "remove_urls": True,  # Remove noise
    "remove_stopwords": False,  # Keep all context
    "lemmatize": False,  # Keep word forms
    "preserve_financial_punctuation": True,  # Keep %, $, decimals
    "handle_negations": True,  # Mark "not good" as "not_good"
}
```

**When to use:** Always use for sentiment analysis with FinBERT or other transformers.

### Minimal

```python
config = {
    "lowercase": True,
    "remove_urls": True,
    "remove_stopwords": False,
    "lemmatize": False,
}
```

**When to use:** General cleaning without aggressive transformations.

### Standard

```python
config = {
    "lowercase": True,
    "remove_urls": True,
    "remove_stopwords": True,  # Removes common words
    "lemmatize": False,
}
```

**When to use:** Traditional text analysis, topic modeling.

### Full

```python
config = {
    "lowercase": True,
    "remove_urls": True,
    "remove_stopwords": True,
    "lemmatize": True,  # Converts to base forms
}
```

**When to use:** Classical ML models (Naive Bayes, SVM), keyword extraction.

## Key Features

### 1. Financial Term Preservation

50+ financial terms are always preserved during stopword removal:

```python
FINANCIAL_TERMS = {
    "stock", "stocks", "share", "shares", "market", "markets",
    "bullish", "bearish", "bull", "bear", "rally", "crash",
    "gain", "gains", "loss", "losses", "profit", "profits",
    "revenue", "earnings", "dividend", "dividends",
    # ... and more
}
```

### 2. Intensity Modifiers

Sentiment strength words are preserved:

```python
INTENSITY_MODIFIERS = {
    "very", "extremely", "highly", "significantly", "strongly",
    "moderately", "slightly", "barely", "somewhat", "quite",
    "rather", "absolutely", "completely", "totally",
    "exceptionally", "remarkably"
}
```

**Example:**
- Input: `"very bullish market"`
- Without preservation: `"bullish market"` (lost intensity)
- With preservation: `"very bullish market"` ✓

### 3. Negation Handling

Negations are marked to prevent sentiment flip:

```python
_NEGATION_PATTERN = r"\b(not|no|never|neither|nobody|nothing|nowhere|n't)\b"
```

**Example:**
- Input: `"not profitable"`
- Without handling: `"profitable"` (opposite sentiment!)
- With handling: `"not_profitable"` ✓

**Why critical:** FinBERT needs to see "not" + "profitable" together, not separately.

### 4. Financial Punctuation Preservation

Keeps %, $, and decimal points in numeric contexts:

**Examples:**
- `"Stock up 25%"` → `"Stock up 25%"` (not `"Stock up 25"`)
- `"Price $150"` → `"Price $150"` (not `"Price 150"`)
- `"EPS 0.50"` → `"EPS 0.50"` (not `"EPS 050"`)

## Usage Examples

### Python Script

```python
from app.preprocessing import TextProcessor

# Create FinBERT-optimized processor
processor = TextProcessor(
    lowercase=False,
    remove_stopwords=False,
    lemmatize=False,
    preserve_financial_punctuation=True,
    handle_negations=True,
)

# Process single text
text = "Stock up 25%, not declining"
result = processor.process(text, return_string=True)
# Output: "Stock up 25% not_declining"

# Process batch
texts = [
    "Revenue up $100M, very profitable",
    "No growth, market declining",
]
results = processor.process_batch(texts, return_strings=True)
```

### Command Line

```bash
# Use FinBERT config
python backend/app/pipelines/preprocess_data.py \
    --source reddit \
    --config finbert \
    --output data/processed/reddit

# Process all sources
python backend/app/pipelines/preprocess_data.py \
    --source all \
    --config finbert
```

### Jupyter Notebook

```python
import sys
sys.path.append('../backend')

from app.preprocessing import preprocess_text

# Single text
text = "Tesla stock not rising, market very bearish"
result = preprocess_text(
    text,
    preserve_financial_punctuation=True,
    handle_negations=True,
    lowercase=False,
    remove_stopwords_flag=False,
    lemmatize=False,
    return_string=True
)
print(result)
# Output: "Tesla stock not_rising market very bearish"
```

## Testing

Comprehensive test suite with 61 tests covering:

- Text normalization (URLs, emails, mentions, hashtags)
- Tokenization
- Stopword removal with financial term preservation
- Lemmatization
- Negation handling
- Financial punctuation preservation
- Intensity modifier preservation
- FinBERT configuration validation

Run tests:
```bash
cd backend
pytest tests/test_preprocessing.py -v
```

## Performance Considerations

1. **Speed**: Minimal preprocessing is faster than full preprocessing
2. **Memory**: No significant difference between configs
3. **Accuracy**: FinBERT config provides best sentiment accuracy
4. **Scalability**: Can process 1000s of documents per minute

## Common Issues and Solutions

### Issue: "not profitable" becoming "profitable"

**Solution:** Enable `handle_negations=True`

### Issue: Lost percentage/dollar context

**Solution:** Enable `preserve_financial_punctuation=True`

### Issue: Poor sentiment accuracy

**Solution:** Use `finbert` config preset, avoid aggressive preprocessing

### Issue: Case-sensitive entity recognition failing

**Solution:** Set `lowercase=False` for FinBERT

## Related Documentation

- [Data Pipeline](data-pipeline.md) - Full pipeline overview
- [FinBERT Model](finbert-model.md) - Sentiment analysis with FinBERT
- [Notebooks](../notebooks/04-text-preprocessing.ipynb) - Interactive examples
- [Tests](../backend/tests/preprocessing/test_preprocessing.py) - Test examples
