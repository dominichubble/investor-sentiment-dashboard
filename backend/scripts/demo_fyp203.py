"""
FYP-203 Demo Script

Demonstrates stock entity extraction and sentiment pairing functionality.
"""

import logging
from pathlib import Path

from app.stocks import analyze_stock_sentiment
from app.storage import StockSentimentStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_analysis():
    """Demonstrate basic stock sentiment analysis."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Stock Sentiment Analysis")
    print("=" * 70)

    text = "Apple reported strong Q4 earnings, beating expectations. Stock surged 15% in after-hours trading."

    logger.info(f"Analyzing text: {text}")

    result = analyze_stock_sentiment(text)

    print(f"\nOverall Sentiment: {result['overall_sentiment']['label']} ({result['overall_sentiment']['score']:.2f})")
    print(f"Stocks Found: {result['metadata']['entities_found']}")
    print(f"Processing Time: {result['metadata']['processing_time_ms']:.2f}ms")

    print("\nStock Details:")
    for stock in result["stocks"]:
        print(f"\n  Ticker: {stock['ticker']}")
        print(f"  Company: {stock['company_name']}")
        print(f"  Mentioned as: {stock['mentioned_as']}")
        print(f"  Sentiment: {stock['sentiment']['label']} ({stock['sentiment']['score']:.2f})")
        print(f"  Context: {stock['context']}")


def demo_multiple_stocks():
    """Demonstrate handling multiple stocks with different sentiments."""
    print("\n" + "=" * 70)
    print("DEMO 2: Multiple Stocks with Different Sentiments")
    print("=" * 70)

    text = """
    $AAPL surged 15% on strong earnings while $TSLA dropped 8% due to 
    production delays. Meanwhile, Microsoft announced new AI features.
    """

    logger.info(f"Analyzing text with multiple stocks: {text.strip()}")

    result = analyze_stock_sentiment(text)

    print(f"\nOverall Sentiment: {result['overall_sentiment']['label']}")
    print(f"Stocks Found: {len(result['stocks'])}")

    for stock in result["stocks"]:
        print(
            f"\n  {stock['ticker']}: {stock['sentiment']['label']} ({stock['sentiment']['score']:.2f})"
        )
        print(f"    Mentioned as: {stock['mentioned_as']}")
        print(f"    Context: {stock['context'][:80]}...")


def demo_storage():
    """Demonstrate saving and querying stock sentiment data."""
    print("\n" + "=" * 70)
    print("DEMO 3: Storage and Querying")
    print("=" * 70)

    # Analyze multiple texts
    texts = [
        "$AAPL reported record Q4 earnings. Stock up 12%.",
        "Apple announced new iPhone. Analysts bullish on $AAPL.",
        "Tesla faces production delays. $TSLA down 5%.",
        "$TSLA Cybertruck deliveries begin. Positive sentiment.",
        "Microsoft AI strategy impressing investors. $MSFT gains 8%.",
    ]

    storage = StockSentimentStorage()
    storage.load()

    logger.info("Analyzing and storing multiple texts...")

    for i, text in enumerate(texts, 1):
        logger.info(f"Processing text {i}/{len(texts)}")
        result = analyze_stock_sentiment(text)
        storage.save_analysis_result(result, source="demo")

    print("\nData saved successfully!")

    # Query AAPL sentiment
    print("\n--- AAPL Sentiment ---")
    aapl_records = storage.get_stock_sentiment("AAPL", source="demo")
    print(f"Total AAPL mentions: {len(aapl_records)}")

    aapl_agg = storage.aggregate_sentiment("AAPL")
    print(f"Average score: {aapl_agg['average_score']:.2f}")
    print(f"Distribution: {aapl_agg['sentiment_distribution']}")

    # Get trending stocks
    print("\n--- Trending Stocks ---")
    trending = storage.get_trending_stocks(min_mentions=1, hours=24)
    for i, stock in enumerate(trending[:5], 1):
        print(f"{i}. {stock['ticker']}: {stock['mentions']} mentions")

    # Statistics
    print("\n--- Overall Statistics ---")
    stats = storage.get_statistics()
    print(f"Total records: {stats['total_records']}")
    print(f"Unique tickers: {stats['unique_tickers']}")


def demo_entity_extraction():
    """Demonstrate entity extraction capabilities."""
    print("\n" + "=" * 70)
    print("DEMO 4: Entity Extraction")
    print("=" * 70)

    from app.entities import (
        extract_company_entities,
        extract_financial_keywords,
    )

    text = "Apple, Microsoft, and Tesla are leading tech companies in the stock market."

    logger.info("Extracting entities...")

    entities = extract_company_entities(text, include_context=True)

    print(f"\nFound {len(entities)} entities:")
    for entity in entities:
        print(f"\n  Text: {entity['text']}")
        print(f"  Type: {entity['label']}")
        print(f"  Position: {entity['start']}-{entity['end']}")
        print(f"  Context: {entity.get('context', 'N/A')}")

    # Financial keywords
    print("\n--- Financial Keyword Detection ---")
    keywords = extract_financial_keywords(text)
    for key, value in keywords.items():
        print(f"  {key}: {value}")


def demo_entity_resolution():
    """Demonstrate entity resolution with fuzzy matching."""
    print("\n" + "=" * 70)
    print("DEMO 5: Entity Resolution")
    print("=" * 70)

    from app.entities import EntityResolver, StockDatabase

    stock_db = StockDatabase()
    stock_db.load()

    resolver = EntityResolver(stock_db)

    test_entities = [
        "Apple",
        "AAPL",
        "$AAPL",
        "Apple Inc.",
        "Tesla",
        "TSLA",
        "Microsoft",
        "Alphabet",  # Google's parent company
    ]

    print("\nResolving entities to tickers:")

    for entity in test_entities:
        result = resolver.resolve(entity)
        if result:
            print(f"\n  '{entity}' → {result['ticker']} ({result['company_name']})")
            print(f"    Match: {result['match_type']} (score: {result['match_score']:.2f})")
        else:
            print(f"\n  '{entity}' → Could not resolve")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("FYP-203: Stock Entity Extraction and Sentiment Pairing")
    print("Demo Script")
    print("=" * 70)

    try:
        demo_basic_analysis()
        demo_multiple_stocks()
        demo_entity_extraction()
        demo_entity_resolution()
        demo_storage()

        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run API server: uvicorn api.main:app --reload")
        print("  2. Visit API docs: http://localhost:8000/docs")
        print("  3. Run tests: pytest tests/ -v")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
