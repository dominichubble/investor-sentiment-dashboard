"""
Stock sentiment analyzer - pairs stock entities with sentiment analysis.

Extracts stocks from text and analyzes sentiment for each mention.
"""

import logging
import re
import time
from typing import Dict, List, Optional

from app.entities import (
    EntityResolver,
    StockDatabase,
    extract_company_entities,
    is_likely_stock_mention,
)
from app.models.finbert_model import FinBERTModel
from app.preprocessing.text_processor import extract_tickers

logger = logging.getLogger(__name__)


class StockSentimentAnalyzer:
    """Analyzes sentiment for stock mentions in text."""

    def __init__(
        self,
        model: Optional[FinBERTModel] = None,
        stock_database: Optional[StockDatabase] = None,
    ):
        """
        Initialize stock sentiment analyzer.

        Args:
            model: FinBERTModel instance (creates new if None)
            stock_database: StockDatabase instance (creates new if None)
        """
        self.model = model or FinBERTModel()
        self.stock_db = stock_database or StockDatabase()
        self.stock_db.load()
        
        # Auto-rebuild if database is empty
        if self.stock_db.get_total_stocks() == 0:
            logger.info("Stock database is empty, rebuilding with fallback data...")
            if self.stock_db.database_file.exists():
                self.stock_db.database_file.unlink()
            self.stock_db._loaded = False
            self.stock_db.download_and_build()
            logger.info(f"Rebuilt database with {self.stock_db.get_total_stocks()} stocks")
        
        self.entity_resolver = EntityResolver(self.stock_db)

    def analyze(
        self,
        text: str,
        extract_context: bool = True,
        include_movements: bool = True,
    ) -> Dict:
        """
        Extract stocks and pair with sentiment analysis.

        Args:
            text: Input text
            extract_context: Extract context around stock mentions
            include_movements: Include stock price movement detection

        Returns:
            {
                'text': str,
                'overall_sentiment': {
                    'label': str,
                    'score': float,
                    'scores': {
                        'positive': float,
                        'negative': float,
                        'neutral': float
                    }
                },
                'stocks': [
                    {
                        'ticker': str,
                        'company_name': str,
                        'mentioned_as': str,
                        'sentiment': {
                            'label': str,
                            'score': float
                        },
                        'context': str,
                        'position': {'start': int, 'end': int}
                    }
                ],
                'metadata': {
                    'entities_found': int,
                    'tickers_extracted': List[str],
                    'processing_time_ms': float
                }
            }
        """
        start_time = time.time()

        # Extract overall sentiment
        overall_sentiment = self.model.predict(text)

        # Extract stock entities
        stocks = self._extract_all_stocks(
            text, extract_context, include_movements
        )

        # Build result
        processing_time = (time.time() - start_time) * 1000

        result = {
            "text": text,
            "overall_sentiment": overall_sentiment,
            "stocks": stocks,
            "metadata": {
                "entities_found": len(stocks),
                "tickers_extracted": [s["ticker"] for s in stocks],
                "processing_time_ms": round(processing_time, 2),
            },
        }

        return result

    def _extract_all_stocks(
        self, text: str, extract_context: bool, include_movements: bool
    ) -> List[Dict]:
        """
        Extract all stock mentions from text.

        Combines:
        1. Ticker extraction ($AAPL)
        2. Company name extraction (NER)

        Args:
            text: Input text
            extract_context: Extract context around mentions
            include_movements: Include price movement data

        Returns:
            List of stock mention dicts
        """
        stocks = []
        seen_tickers = set()

        # 1. Extract explicit tickers ($AAPL, $TSLA)
        tickers = extract_tickers(text)
        for ticker in tickers:
            stock_info = self.stock_db.get_by_ticker(ticker)
            if stock_info:
                # Find ticker position in text
                ticker_pattern = f"\\${ticker}"
                match = re.search(ticker_pattern, text, re.IGNORECASE)

                if match:
                    position = {"start": match.start(), "end": match.end()}

                    # Extract context if requested
                    context = (
                        self._extract_context(text, position)
                        if extract_context
                        else None
                    )

                    # Analyze sentiment for context
                    sentiment = (
                        self.model.predict(context)
                        if context
                        else {"label": "neutral", "score": 0.33}
                    )

                    stocks.append(
                        {
                            "ticker": ticker,
                            "company_name": stock_info["company_name"],
                            "mentioned_as": f"${ticker}",
                            "sentiment": {
                                "label": sentiment["label"],
                                "score": sentiment["score"],
                            },
                            "context": context,
                            "position": position,
                        }
                    )

                    seen_tickers.add(ticker)

        # 2. Extract company names using NER
        entities = extract_company_entities(text, include_context=False)

        for entity in entities:
            # Check if likely a stock mention
            if not is_likely_stock_mention(entity, text):
                continue

            # Resolve entity to ticker
            resolved = self.entity_resolver.resolve(entity["text"])

            if resolved and resolved["ticker"] not in seen_tickers:
                ticker = resolved["ticker"]
                position = {"start": entity["start"], "end": entity["end"]}

                # Extract context if requested
                context = (
                    self._extract_context(text, position)
                    if extract_context
                    else None
                )

                # Analyze sentiment for context
                sentiment = (
                    self.model.predict(context)
                    if context
                    else {"label": "neutral", "score": 0.33}
                )

                stocks.append(
                    {
                        "ticker": ticker,
                        "company_name": resolved["company_name"],
                        "mentioned_as": entity["text"],
                        "sentiment": {
                            "label": sentiment["label"],
                            "score": sentiment["score"],
                        },
                        "context": context,
                        "position": position,
                    }
                )

                seen_tickers.add(ticker)

        # 3. Fallback: Direct name matching against stock database
        # This catches cases where spaCy NER doesn't recognize company names
        self._extract_by_name_matching(text, stocks, seen_tickers, extract_context)

        return stocks

    def _extract_by_name_matching(
        self,
        text: str,
        stocks: List[Dict],
        seen_tickers: set,
        extract_context: bool,
    ) -> None:
        """
        Extract stocks by direct name matching against the database.
        
        This is a fallback for when spaCy NER doesn't recognize company names.
        
        Args:
            text: Input text
            stocks: List to append found stocks to
            seen_tickers: Set of already found tickers
            extract_context: Whether to extract context
        """
        # Get all stocks from database for matching
        all_tickers = self.stock_db.get_all_tickers()
        
        for ticker in all_tickers:
            if ticker in seen_tickers:
                continue
                
            stock_info = self.stock_db.get_by_ticker(ticker)
            if not stock_info:
                continue
            
            # Check for common names in text
            common_names = stock_info.get("common_names", [])
            
            for name in common_names:
                # Skip very short names (likely to cause false positives)
                if len(name) < 3:
                    continue
                
                # Skip ticker-only names (already handled)
                if name.upper() == ticker:
                    continue
                
                # Case-insensitive word boundary search
                # \b ensures we match whole words only (not "apple" in "pineapple")
                pattern = r'\b' + re.escape(name) + r'\b'
                match = re.search(pattern, text, re.IGNORECASE)
                
                if match:
                    position = {"start": match.start(), "end": match.end()}
                    
                    # Extract context if requested
                    context = (
                        self._extract_context(text, position)
                        if extract_context
                        else None
                    )
                    
                    # Analyze sentiment for context
                    sentiment = (
                        self.model.predict(context)
                        if context
                        else {"label": "neutral", "score": 0.33}
                    )
                    
                    stocks.append(
                        {
                            "ticker": ticker,
                            "company_name": stock_info["company_name"],
                            "mentioned_as": match.group(),
                            "sentiment": {
                                "label": sentiment["label"],
                                "score": sentiment["score"],
                            },
                            "context": context,
                            "position": position,
                        }
                    )
                    
                    seen_tickers.add(ticker)
                    break  # Found this stock, move to next

    def _extract_context(
        self, text: str, position: Dict, window: int = 80
    ) -> str:
        """
        Extract focused context around a stock mention.

        Uses clause-aware extraction to avoid mixing sentiment from multiple stocks.

        Args:
            text: Full text
            position: Position dict with 'start' and 'end'
            window: Character window size

        Returns:
            Context string focused on this specific stock
        """
        # Extract window around the mention
        start = max(0, position["start"] - window)
        end = min(len(text), position["end"] + window)
        context = text[start:end]
        
        # Find the stock mention within the context
        mention_start = position["start"] - start
        mention_end = position["end"] - start
        
        # Split on clause delimiters that often separate different stocks' sentiments
        # e.g., "AAPL surged while TSLA dropped" - split on "while"
        clause_delimiters = [
            r'\s+while\s+',
            r'\s+but\s+',
            r'\s+however\s+',
            r'\s+whereas\s+',
            r'\s+although\s+',
            r'\s+though\s+',
            r',\s+and\s+',
            r',\s+but\s+',
            r';\s+',
        ]
        
        # Try to find which clause contains the stock mention
        best_clause = context
        min_length = len(context)
        
        for delimiter_pattern in clause_delimiters:
            # Split the context
            parts = re.split(delimiter_pattern, context, flags=re.IGNORECASE)
            
            # Find which part contains the mention
            current_pos = 0
            for part in parts:
                part_start = current_pos
                part_end = current_pos + len(part)
                
                # Check if mention is in this part
                if part_start <= mention_start < part_end:
                    # Use this clause if it's shorter (more focused)
                    if len(part.strip()) < min_length and len(part.strip()) > 10:
                        best_clause = part.strip()
                        min_length = len(best_clause)
                    break
                
                # Account for delimiter length (approximate)
                current_pos = part_end + 5
        
        # Clean up the clause
        best_clause = best_clause.strip()
        
        # If clause is too short, try sentence boundaries
        if len(best_clause) < 20:
            # Extract sentence containing the mention from original text
            sentences = re.split(r'[.!?]+', text)
            char_count = 0
            
            for sentence in sentences:
                sentence_start = char_count
                sentence_end = char_count + len(sentence)
                
                if sentence_start <= position["start"] < sentence_end:
                    best_clause = sentence.strip()
                    break
                
                char_count = sentence_end + 1
        
        return best_clause if best_clause else context.strip()


def analyze_stock_sentiment(
    text: str,
    model: Optional[FinBERTModel] = None,
    stock_database: Optional[StockDatabase] = None,
    extract_context: bool = True,
    include_movements: bool = True,
) -> Dict:
    """
    Convenience function to analyze stock sentiment.

    Args:
        text: Input text
        model: FinBERTModel instance (creates new if None)
        stock_database: StockDatabase instance (creates new if None)
        extract_context: Extract context around stock mentions
        include_movements: Include stock price movement detection

    Returns:
        Stock sentiment analysis result dict

    Example:
        >>> result = analyze_stock_sentiment("Apple reported strong earnings")
        >>> print(result['stocks'][0]['ticker'])
        'AAPL'
        >>> print(result['stocks'][0]['sentiment']['label'])
        'positive'
    """
    analyzer = StockSentimentAnalyzer(model, stock_database)
    return analyzer.analyze(text, extract_context, include_movements)
