# Process Existing Data Script

This script allows you to apply sentiment analysis to your existing collected data and save the predictions to SQLite.

## Prerequisites

Make sure you have pandas installed:
```bash
pip install pandas
```

## Usage

### Process Reddit Data

```bash
cd backend
python scripts/process_existing_data.py --input ../data/reddit_*.csv --source reddit
```

### Process Twitter Data

```bash
cd backend
python scripts/process_existing_data.py --input ../data/twitter_*.csv --source twitter
```

### Process News Data

```bash
cd backend
python scripts/process_existing_data.py --input ../data/news_*.csv --source news
```

### Custom Text Column

If your CSV has a different column name for text:

```bash
python scripts/process_existing_data.py --input data.csv --source reddit --text-column body
```

## What It Does

1. **Reads your CSV file** - Loads existing data from data collection pipelines
2. **Analyzes sentiment** - Uses FinBERT to analyze each text (batch processing for efficiency)
3. **Saves predictions** - Stores results in SQLite (`data/db/sentiments.db`)

## Output Format

Predictions are saved with the following fields:
- `record_type`: `document`
- `document_id`: Stable document ID
- `text`: The analyzed text (truncated to 2000 chars for storage)
- `source`: Data source (reddit, twitter, news)
- `timestamp`: When the analysis was performed
- `sentiment_label`: Sentiment label (positive, negative, neutral)
- `sentiment_score`: Confidence score (0-1)

## Example

```bash
$ python scripts/process_existing_data.py --input ../data/reddit_investing.csv --source reddit

Found 1 file(s) to process
Reading ../data/reddit_investing.csv...
Found 150 texts to analyze
Analyzing sentiment...
Preparing records for SQLite...
Saving 150 predictions to SQLite...
✓ Saved 150 predictions successfully!

Total predictions saved: 150
```

## Notes

- Records are inserted into SQLite and de-duplicated by ID
- Texts are truncated to 2000 characters for storage efficiency
- Empty/null texts are automatically skipped
- Processes in batches of 32 for optimal performance
