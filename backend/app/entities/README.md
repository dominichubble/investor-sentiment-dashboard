# Entity Extraction Module

This module provides stock entity extraction and resolution capabilities for FYP-203.

## Components

### `stock_database.py`
- **StockDatabase**: Manages comprehensive ticker-name mapping for publicly traded stocks
- Downloads data from SEC EDGAR Company Tickers
- Supports ~13,000+ US-listed stocks
- Provides search and lookup functionality

### `company_extractor.py`
- **extract_company_entities()**: Extracts organization entities using spaCy NER
- **extract_financial_keywords()**: Detects financial context in text
- **is_likely_stock_mention()**: Determines if entity is likely a stock mention
- Uses spaCy `en_core_web_sm` model

### `entity_resolver.py`
- **EntityResolver**: Resolves entity mentions to stock ticker symbols
- Supports exact matching and fuzzy matching (using FuzzyWuzzy)
- Handles common variations and abbreviations
- Includes blacklist for common false positives

## Usage

```python
from app.entities import StockDatabase, extract_company_entities, EntityResolver

# Initialize database
stock_db = StockDatabase()
stock_db.load()  # Downloads SEC data if not exists

# Extract entities from text
text = "Apple reported strong earnings while Tesla faced delays"
entities = extract_company_entities(text)

# Resolve entities to tickers
resolver = EntityResolver(stock_db)
for entity in entities:
    result = resolver.resolve(entity['text'])
    if result:
        print(f"{entity['text']} → {result['ticker']}")
```

## Dependencies

- **spaCy**: NER model for entity extraction
- **FuzzyWuzzy**: Fuzzy string matching
- **SEC EDGAR**: Official company ticker data source

## Data Sources

- **SEC EDGAR Company Tickers**: https://www.sec.gov/files/company_tickers.json
  - ~13,000 public companies
  - Updated quarterly
  - Official US government source
