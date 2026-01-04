# Storage Module

This module provides functionality to save and load sentiment predictions in structured formats (CSV and JSON) with comprehensive validation to ensure data integrity.

## Overview

The storage module ensures that all sentiment predictions are saved with complete, validated data. No predictions with missing fields can be stored, guaranteeing data quality for downstream analysis.

## Quick Start

```python
from app.storage import save_prediction, load_predictions, get_storage_stats

# Save a single prediction
save_prediction(
    text="Stock prices surged after earnings",
    source="reddit",
    label="positive",
    confidence=0.95,
    output_file="data/predictions.csv"
)

# Load predictions
predictions = load_predictions("data/predictions.csv")
print(f"Loaded {len(predictions)} predictions")

# Get statistics
stats = get_storage_stats("data/predictions.csv")
print(f"Total: {stats['total']}, Positive: {stats['by_label']['positive']}")
```

## Schema

All predictions must have the following fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `text` | string | The analyzed text (non-empty) | "Stock market rallied today" |
| `source` | string | Data source identifier | "reddit", "twitter", "news" |
| `timestamp` | string | ISO 8601 format timestamp | "2026-01-04T10:30:00" |
| `label` | string | Sentiment label | "positive", "negative", "neutral" |
| `confidence` | float | Confidence score (0-1) | 0.95 |

**Validation Rules:**
- ✅ All fields are required
- ✅ `text` and `source` must be non-empty strings
- ✅ `label` must be one of: "positive", "negative", "neutral"
- ✅ `confidence` must be between 0 and 1
- ✅ `timestamp` must be valid ISO 8601 format
- ❌ No missing fields allowed
- ❌ No invalid values allowed

## Functions

### `PredictionRecord` Class

Data class for creating validated prediction records.

```python
from app.storage import PredictionRecord

# Create a record (validates inputs)
record = PredictionRecord(
    text="Market sentiment is bullish",
    source="reddit",
    label="positive",
    confidence=0.92,
    timestamp="2026-01-04T10:00:00"  # Optional, auto-generated if not provided
)

# Convert to dictionary
pred_dict = record.to_dict()
```

**Raises:** `ValueError` if any field is invalid

---

### `save_prediction()`

Save a single sentiment prediction to file.

```python
from app.storage import save_prediction

record = save_prediction(
    text="Stock prices surged today",
    source="reddit",
    label="positive",
    confidence=0.95,
    output_file="predictions.csv",
    timestamp="2026-01-04T10:30:00",  # Optional
    format="csv"  # or "json"
)
```

**Parameters:**
- `text` (str): The analyzed text
- `source` (str): Data source (e.g., 'reddit', 'twitter', 'news')
- `label` (str): Sentiment label ('positive', 'negative', 'neutral')
- `confidence` (float): Confidence score (0-1)
- `output_file` (str|Path): Path to output file
- `timestamp` (str, optional): ISO timestamp (auto-generated if None)
- `format` (str): Output format ('csv' or 'json')

**Returns:** `PredictionRecord` that was saved

**Raises:** `ValueError` if validation fails, `IOError` if file write fails

---

### `save_predictions_batch()`

Save multiple sentiment predictions efficiently.

```python
from app.storage import save_predictions_batch

predictions = [
    {
        "text": "Market rally continues",
        "source": "twitter",
        "timestamp": "2026-01-04T09:00:00",
        "label": "positive",
        "confidence": 0.92
    },
    {
        "text": "Economic downturn expected",
        "source": "news",
        "timestamp": "2026-01-04T09:15:00",
        "label": "negative",
        "confidence": 0.88
    }
]

count = save_predictions_batch(
    predictions,
    output_file="predictions.csv",
    format="csv",
    append=True  # Append to existing file
)

print(f"Saved {count} predictions")
```

**Parameters:**
- `predictions` (List[Dict]): List of prediction dictionaries
- `output_file` (str|Path): Path to output file
- `format` (str): Output format ('csv' or 'json')
- `append` (bool): If True, append to existing file. If False, overwrite.

**Returns:** Number of predictions saved

**Raises:** `ValueError` if any prediction fails validation

**Important:** All predictions are validated before any are saved. If one fails validation, none are saved.

---

### `load_predictions()`

Load sentiment predictions from file.

```python
from app.storage import load_predictions

# Auto-detect format from extension
predictions = load_predictions("predictions.csv")

# Explicitly specify format
predictions = load_predictions("predictions.json", format="json")

# Skip validation (not recommended)
predictions = load_predictions("predictions.csv", validate=False)

print(f"Loaded {len(predictions)} predictions")
for pred in predictions:
    print(f"{pred['label']}: {pred['text']}")
```

**Parameters:**
- `input_file` (str|Path): Path to input file
- `format` (str, optional): File format ('csv' or 'json'). Auto-detected from extension if None.
- `validate` (bool): If True, validate all loaded predictions

**Returns:** List of prediction dictionaries

**Raises:** `FileNotFoundError`, `ValueError` if validation fails

---

### `validate_prediction()`

Validate that a prediction dictionary has all required fields with correct values.

```python
from app.storage import validate_prediction

prediction = {
    "text": "Stock up",
    "source": "reddit",
    "timestamp": "2026-01-04T10:00:00",
    "label": "positive",
    "confidence": 0.95
}

try:
    validate_prediction(prediction)
    print("✓ Valid prediction")
except ValueError as e:
    print(f"✗ Invalid: {e}")
```

**Returns:** `True` if valid

**Raises:** `ValueError` with details about missing/invalid fields

---

### `get_storage_stats()`

Get statistics about stored predictions.

```python
from app.storage import get_storage_stats

stats = get_storage_stats("predictions.csv")

print(f"Total predictions: {stats['total']}")
print(f"By label: {stats['by_label']}")
print(f"By source: {stats['by_source']}")
print(f"Average confidence: {stats['avg_confidence']:.2f}")
print(f"Date range: {stats['date_range']['first']} to {stats['date_range']['last']}")
```

**Returns:** Dictionary with:
- `total`: Total number of predictions
- `by_label`: Count by sentiment label
- `by_source`: Count by data source
- `avg_confidence`: Average confidence score
- `date_range`: First and last timestamp

---

## File Formats

### CSV Format

```csv
text,source,timestamp,label,confidence
"Stock prices surged today",reddit,2026-01-04T10:30:00,positive,0.95
"Market crash looming",news,2026-01-04T10:31:00,negative,0.88
"Federal Reserve maintains rates",twitter,2026-01-04T10:32:00,neutral,0.85
```

**Features:**
- Header row with field names
- UTF-8 encoding
- Automatic quoting for text with commas
- Efficient for large datasets
- Easy to import into spreadsheets

### JSON Format

```json
[
  {
    "text": "Stock prices surged today",
    "source": "reddit",
    "timestamp": "2026-01-04T10:30:00",
    "label": "positive",
    "confidence": 0.95
  },
  {
    "text": "Market crash looming",
    "source": "news",
    "timestamp": "2026-01-04T10:31:00",
    "label": "negative",
    "confidence": 0.88
  }
]
```

**Features:**
- Pretty-printed with 2-space indentation
- UTF-8 encoding
- Better for nested data structures
- Easy to parse in JavaScript/web apps

---

## Usage Examples

### Example 1: Save Predictions from Sentiment Analysis

```python
from app.models import analyze_batch
from app.storage import save_predictions_batch
from datetime import datetime

# Analyze texts
texts = [
    "Stock market rallied today",
    "Economic downturn expected",
    "Federal Reserve maintains rates"
]

results = analyze_batch(texts)

# Prepare for storage
predictions = []
for text, result in zip(texts, results):
    predictions.append({
        "text": text,
        "source": "reddit",  # Specify your data source
        "timestamp": datetime.utcnow().isoformat(),
        "label": result["label"],
        "confidence": result["score"]
    })

# Save to file
count = save_predictions_batch(predictions, "data/predictions.csv")
print(f"Saved {count} predictions")
```

### Example 2: Append Daily Predictions

```python
from app.storage import save_predictions_batch
from datetime import datetime

def save_daily_predictions(new_predictions):
    """Append today's predictions to the main file."""
    
    output_file = "data/predictions.csv"
    
    # Add timestamps to all predictions
    for pred in new_predictions:
        if "timestamp" not in pred:
            pred["timestamp"] = datetime.utcnow().isoformat()
    
    # Append to existing file
    count = save_predictions_batch(
        new_predictions,
        output_file,
        format="csv",
        append=True
    )
    
    print(f"Appended {count} predictions to {output_file}")
    
    return count
```

### Example 3: Load and Analyze Predictions

```python
from app.storage import load_predictions, get_storage_stats

# Load all predictions
predictions = load_predictions("data/predictions.csv")

# Filter by source
reddit_predictions = [p for p in predictions if p["source"] == "reddit"]
print(f"Reddit predictions: {len(reddit_predictions)}")

# Calculate sentiment distribution
positive = sum(1 for p in predictions if p["label"] == "positive")
negative = sum(1 for p in predictions if p["label"] == "negative")
neutral = sum(1 for p in predictions if p["label"] == "neutral")

print(f"Positive: {positive}, Negative: {negative}, Neutral: {neutral}")

# Or use get_storage_stats for quick analysis
stats = get_storage_stats("data/predictions.csv")
print(f"Stats: {stats}")
```

### Example 4: Convert Between Formats

```python
from app.storage import load_predictions, save_predictions_batch

# Load from CSV
predictions = load_predictions("data/predictions.csv")

# Save as JSON
save_predictions_batch(
    predictions,
    "data/predictions.json",
    format="json",
    append=False  # Overwrite if exists
)

print(f"Converted {len(predictions)} predictions from CSV to JSON")
```

### Example 5: Validation Before Saving

```python
from app.storage import validate_prediction, save_predictions_batch

predictions = [
    {"text": "Market up", "source": "reddit", ...},
    {"text": "Stock down", "source": "twitter", ...}
]

# Validate each prediction individually
valid_predictions = []
for i, pred in enumerate(predictions):
    try:
        validate_prediction(pred)
        valid_predictions.append(pred)
    except ValueError as e:
        print(f"Skipping prediction {i}: {e}")

# Save only valid predictions
if valid_predictions:
    save_predictions_batch(valid_predictions, "data/predictions.csv")
```

### Example 6: Integration with Data Pipelines

```python
from app.pipelines.ingest_reddit import fetch_reddit_posts
from app.models import analyze_batch
from app.storage import save_predictions_batch
from datetime import datetime

def process_reddit_sentiment():
    """Full pipeline: fetch → analyze → store."""
    
    # Step 1: Fetch data
    posts = fetch_reddit_posts(subreddit="investing", limit=100)
    
    # Step 2: Analyze sentiment
    texts = [post["text"] for post in posts]
    results = analyze_batch(texts, batch_size=32)
    
    # Step 3: Prepare predictions
    predictions = []
    for post, result in zip(posts, results):
        predictions.append({
            "text": post["text"],
            "source": "reddit",
            "timestamp": datetime.utcnow().isoformat(),
            "label": result["label"],
            "confidence": result["score"]
        })
    
    # Step 4: Save to storage
    count = save_predictions_batch(
        predictions,
        "data/reddit_predictions.csv",
        append=True
    )
    
    print(f"Processed and saved {count} Reddit predictions")
    
    return count
```

---

## Best Practices

1. **Always validate before saving** - Use `validate_prediction()` or rely on automatic validation
2. **Use batch operations** - `save_predictions_batch()` is more efficient than multiple `save_prediction()` calls
3. **Append incrementally** - Set `append=True` to add new predictions without rewriting the file
4. **Auto-generate timestamps** - Don't provide timestamp unless you have a specific reason
5. **Choose appropriate format**:
   - CSV for large datasets and spreadsheet compatibility
   - JSON for web applications and nested data
6. **Handle validation errors** - Wrap save operations in try/except to handle invalid data gracefully
7. **Check statistics regularly** - Use `get_storage_stats()` to monitor data quality

---

## Error Handling

```python
from app.storage import save_prediction, validate_prediction

def safe_save_prediction(text, source, label, confidence, output_file):
    """Save prediction with comprehensive error handling."""
    
    try:
        # Attempt to save
        record = save_prediction(
            text=text,
            source=source,
            label=label,
            confidence=confidence,
            output_file=output_file
        )
        print(f"✓ Saved: {record}")
        return record
        
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
        return None
        
    except IOError as e:
        print(f"✗ File write failed: {e}")
        return None
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None
```

---

## Testing

Run the test suite:

```bash
# Test storage module
pytest backend/tests/storage/test_prediction_storage.py -v

# Run with coverage
pytest backend/tests/storage/ --cov=app.storage --cov-report=html
```

---

## Troubleshooting

### Issue: "Missing required fields"
**Solution**: Ensure all 5 fields are present: text, source, timestamp, label, confidence

### Issue: "confidence must be between 0 and 1"
**Solution**: Check that confidence values are valid floats in range [0, 1]

### Issue: "label must be 'positive', 'negative', or 'neutral'"
**Solution**: Only these three labels are supported. Check for typos or extra whitespace.

### Issue: "timestamp must be valid ISO format"
**Solution**: Use ISO 8601 format: `"2026-01-04T10:30:00"` or `datetime.utcnow().isoformat()`

### Issue: File is empty after saving
**Solution**: Check file permissions and disk space. Ensure directory exists.

---

## Dependencies

- `csv` (standard library) - CSV file handling
- `json` (standard library) - JSON file handling
- `datetime` (standard library) - Timestamp handling
- `pathlib` (standard library) - Path operations
- `logging` (standard library) - Logging support

No external dependencies required.
