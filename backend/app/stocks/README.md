# Stock Sentiment Analysis Module

Pairs stock entity extraction with sentiment analysis for FYP-203.

## Components

### `stock_sentiment.py`
- **StockSentimentAnalyzer**: Main analyzer class
- **analyze_stock_sentiment()**: Convenience function

## Features

- Extracts stocks from text using both:
  - Ticker symbols ($AAPL)
  - Company names (Apple) via NER
- Analyzes sentiment for each stock mention
- Extracts context around mentions
- Handles multiple stocks in same text
- Distinguishes sentiment for different stocks

## Usage

```python
from app.stocks import analyze_stock_sentiment

text = "Apple reported strong earnings and stock surged 15%. Meanwhile, Tesla faced production delays and dropped 8%."

result = analyze_stock_sentiment(text)

print(f"Overall sentiment: {result['overall_sentiment']['label']}")

for stock in result['stocks']:
    print(f"{stock['ticker']}: {stock['sentiment']['label']} ({stock['sentiment']['score']:.2f})")
    print(f"  Context: {stock['context']}")
```

## Output Format

```python
{
    'text': str,  # Original text
    'overall_sentiment': {
        'label': str,  # positive/negative/neutral
        'score': float,  # Confidence score
        'scores': {
            'positive': float,
            'negative': float,
            'neutral': float
        }
    },
    'stocks': [
        {
            'ticker': str,  # AAPL
            'company_name': str,  # Apple Inc.
            'mentioned_as': str,  # How it appeared: "Apple" or "$AAPL"
            'sentiment': {
                'label': str,  # positive/negative/neutral
                'score': float  # Confidence score
            },
            'context': str,  # Surrounding text
            'position': {'start': int, 'end': int}  # Character positions
        }
    ],
    'metadata': {
        'entities_found': int,
        'tickers_extracted': List[str],
        'processing_time_ms': float
    }
}
```

## Context Extraction

The analyzer uses sentence-level context extraction by default, which provides the most accurate sentiment for each stock mention.

Strategies available:
1. **Sentence-level** (default): Extract sentence containing stock mention
2. **Window-based**: Extract N characters before/after mention
3. **Paragraph-level**: Extract full paragraph

## Dependencies

- **FinBERT**: Sentiment analysis model
- **Entity module**: Stock entity extraction
- **spaCy**: NER for company names
