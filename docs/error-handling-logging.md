# Error Handling & Logging - FYP-201

## Overview

This branch implements comprehensive error handling and logging throughout the sentiment analysis inference pipeline, making it production-ready and robust.

## Features

### 1. Centralized Logging Configuration

**File:** `backend/app/logging_config.py`

- Configurable logging with both console and file handlers
- Separate error log file for quick issue identification
- Timestamp-based log file naming
- Helper utilities for exception logging

**Usage:**
```python
from app.logging_config import setup_logging, get_logger

# Setup logging once at application start
setup_logging(log_level="INFO")

# Get logger in any module
logger = get_logger(__name__)
logger.info("Processing started")
```

### 2. Failed Items Tracking

The `FailedItemsTracker` class provides structured tracking of failed items:

```python
from app.logging_config import FailedItemsTracker

tracker = FailedItemsTracker()

# Track a failure
tracker.add_failure(
    item="problematic text",
    error_type="TokenizationError",
    error_message="Text too long",
    additional_info={"length": 50000}
)

# Save to JSON file
tracker.save("data/failed_items/failed_20250105.json")
```

**Failed items JSON format:**
```json
[
  {
    "item": "text content (truncated to 500 chars)",
    "error_type": "TokenizationError",
    "error_message": "Text exceeds maximum length",
    "timestamp": "2025-01-05T10:30:00.123456",
    "additional_info": {
      "text_length": 50000,
      "batch_index": 42
    }
  }
]
```

### 3. Enhanced Sentiment Inference

**File:** `backend/app/models/sentiment_inference.py`

**New Features:**
- Custom exception classes: `SentimentAnalysisError`, `TokenizationError`, `ModelInferenceError`
- Text length warnings for texts > 10,000 characters
- Batch-level error handling with individual item fallback
- Automatic failure tracking
- Returns tuple: `(results, failures)`

**Changes:**
```python
# Old API
results = analyze_batch(texts)

# New API (backward compatible if you ignore failures)
results, failures = analyze_batch(
    texts,
    skip_errors=True,  # Default: True
    track_failures=True  # Default: True
)

# Save failed items if any
if failures and failures.count() > 0:
    failures.save("data/failed_items/failed.json")
```

**Error Handling Flow:**
1. Validates all inputs
2. Filters empty/whitespace texts → tracked as failures
3. Attempts batch processing
4. On batch failure with `skip_errors=True`:
   - Retries each item individually
   - Tracks individual failures
   - Returns `None` for failed items
5. Logs all errors with context

### 4. Robust Tokenization

**File:** `backend/app/preprocessing/finbert_tokenizer.py`

**New Features:**
- Maximum text length validation (50,000 chars)
- Warning for texts > 2,000 chars (likely to be truncated)
- Detailed error messages for edge cases
- Mini-batch error handling with specific failure reporting
- Custom `TokenizationError` exception

**Edge Cases Handled:**
- Empty strings
- Whitespace-only text
- Non-string inputs
- Extremely long texts (>50k chars)
- Tensor concatenation failures
- Network errors loading tokenizer

### 5. Enhanced Storage Module

**File:** `backend/app/storage/prediction_storage.py`

**Improvements:**
- Detailed validation error messages
- File I/O error handling
- JSON encoding/decoding error handling
- Directory creation error handling
- Row-level error logging for CSV parsing
- Graceful handling of corrupted files

### 6. Production-Ready Processing Script

**File:** `backend/scripts/process_existing_data.py`

**New Features:**
- Comprehensive logging setup
- Per-file statistics tracking
- Overall summary statistics
- Automatic failed_items.json creation
- Timestamped output files to avoid conflicts
- Detailed progress reporting

**Statistics Tracked:**
- Total records
- Successfully processed
- Failed (with reasons)
- Skipped (empty texts)
- Saved predictions

**Usage:**
```bash
python backend/scripts/process_existing_data.py \
    --input data/raw/reddit/*.csv \
    --source reddit \
    --text-column text
```

**Output:**
- Predictions: `data/predictions/{source}_predictions_{timestamp}.csv`
- Failed items: `data/failed_items/{source}_failed_{timestamp}.json`
- Logs: `logs/sentiment_pipeline_{timestamp}.log`
- Error logs: `logs/errors_{timestamp}.log`

## Testing

**File:** `backend/tests/test_error_handling.py`

Comprehensive test suite covering:
- Logging configuration
- Failed items tracking
- Sentiment inference error cases
- Tokenization edge cases
- Storage validation errors
- Integration scenarios

**Run tests:**
```bash
cd backend
pytest tests/test_error_handling.py -v
```

## Acceptance Criteria ✓

### ✅ Logs failures and skipped records

- All errors logged with timestamps, context, and full tracebacks
- Separate error log file for quick issue identification
- Structured logging with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Empty/whitespace texts logged as skipped

### ✅ Catches long-text errors or tokenization edge cases

- Maximum text length validation (50,000 chars)
- Warnings for texts that will be truncated (>2,000 chars)
- Specific error messages for different failure types
- Individual item retry on batch failures
- Graceful degradation with informative logging

### ✅ Creates a clean "failed_items.json"

- Structured JSON format with all failure details
- Automatic timestamped file creation
- Text truncation (500 chars) for readability
- Additional context (text length, batch index, etc.)
- Only created when failures occur (no empty files)

## Example Output

### Console Output
```
============================================================
Starting sentiment analysis processing
Source: reddit
Input pattern: data/raw/reddit/*.csv
Text column: text
============================================================
Found 3 file(s) to process

Processing file 1/3: reddit_2025-01-05.csv
------------------------------------------------------------
Loaded 1000 records from reddit_2025-01-05.csv
Found 985 valid texts to analyze (15 skipped)
Starting sentiment analysis...
Analysis complete: 980 successful, 5 failed
✓ Saved 5 failed items to data/failed_items/reddit_failed_20250105_143022.json
Saving 980 predictions to data/predictions/reddit_predictions_20250105_143022.csv
✓ Successfully saved 980 predictions!

============================================================
PROCESSING COMPLETE
============================================================
Files processed: 3/3
Total predictions saved: 2945
Failed items: 12
Check logs/ directory for detailed logs and failed_items.json
============================================================
```

### Log Output (`logs/sentiment_pipeline_20250105_143022.log`)
```
2025-01-05 14:30:22 - app.models.sentiment_inference - INFO - Processing 985/1000 valid texts
2025-01-05 14:30:22 - app.models.sentiment_inference - WARNING - Text is very long (15234 chars), may cause issues. Consider truncating before analysis.
2025-01-05 14:30:25 - app.models.sentiment_inference - ERROR - Batch processing failed: CUDA out of memory
2025-01-05 14:30:25 - app.models.sentiment_inference - INFO - Attempting individual analysis for failed batch
2025-01-05 14:30:26 - app.models.sentiment_inference - ERROR - Failed to analyze text at index 42: Text tokenization failed: Text too long
2025-01-05 14:30:30 - app.models.sentiment_inference - INFO - Batch analysis complete: 980/985 successful, 5 failed
```

### Failed Items JSON
```json
[
  {
    "item": "This is an extremely long text that goes on and on... (truncated at 500 chars)",
    "error_type": "TokenizationError",
    "error_message": "Text is too long (52341 chars). Please truncate to under 50000 characters.",
    "timestamp": "2025-01-05T14:30:26.123456",
    "additional_info": {
      "text_length": 52341,
      "batch_index": 42
    }
  }
]
```

## Migration Guide

### For Existing Code

The changes are **backward compatible** with a minor adjustment:

**Before:**
```python
results = analyze_batch(texts)
for result in results:
    print(result['label'])
```

**After (recommended):**
```python
results, failures = analyze_batch(texts)

# Process successful results
for result in results:
    if result is not None:  # Check for None (failed items)
        print(result['label'])

# Handle failures
if failures and failures.count() > 0:
    failures.save("data/failed_items.json")
    print(f"Failed items: {failures.count()}")
```

### Updating Scripts

Add logging setup at the start of your scripts:

```python
from app.logging_config import setup_logging, get_logger

setup_logging(log_level="INFO")
logger = get_logger(__name__)

logger.info("Script started")
```

## Best Practices

1. **Always setup logging** at application entry point
2. **Check for None results** when processing batch results
3. **Save failed items** to JSON for later analysis
4. **Monitor log files** for warnings and errors
5. **Set appropriate log levels** (DEBUG for development, INFO for production)

## Dependencies

No new dependencies required - uses built-in Python logging and existing packages.

## Files Changed

- ✅ `backend/app/logging_config.py` - NEW
- ✅ `backend/app/models/sentiment_inference.py` - MODIFIED
- ✅ `backend/app/preprocessing/finbert_tokenizer.py` - MODIFIED
- ✅ `backend/app/storage/prediction_storage.py` - MODIFIED
- ✅ `backend/scripts/process_existing_data.py` - MODIFIED
- ✅ `backend/tests/test_error_handling.py` - NEW

## Future Improvements

- Add retry logic with exponential backoff
- Implement circuit breaker pattern for external services
- Add metrics/monitoring integration (Prometheus, etc.)
- Email notifications for critical errors
- Automated error report generation
