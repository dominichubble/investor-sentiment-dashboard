"""
Entity resolution module for mapping company names to stock tickers.

Handles fuzzy matching and disambiguation of company entity mentions.
"""

import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz

from .stock_database import StockDatabase

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolves entity mentions to stock ticker symbols."""

    def __init__(self, stock_database: Optional[StockDatabase] = None):
        """
        Initialize entity resolver.

        Args:
            stock_database: StockDatabase instance. If None, creates new one.
        """
        self.stock_db = stock_database or StockDatabase()
        self.stock_db.load()

        # Blacklist of common false positives
        self.blacklist = {
            "federal reserve",
            "fed",
            "sec",
            "fda",
            "fbi",
            "cia",
            "nasa",
            "un",
            "united nations",
            "senate",
            "congress",
            "white house",
            "supreme court",
        }

    def resolve(self, entity_text: str, threshold: float = 0.85) -> Optional[Dict]:
        """
        Resolve entity text to stock ticker information.

        Priority:
        1. Exact ticker match ($AAPL → AAPL)
        2. Exact company name match
        3. Fuzzy match on common names (>threshold similarity)
        4. None if no match

        Args:
            entity_text: Entity text to resolve (e.g., "Apple", "AAPL")
            threshold: Minimum similarity score for fuzzy matching (0-1)

        Returns:
            Stock info dict or None if not resolved:
            {
                'ticker': str,
                'company_name': str,
                'match_type': str,  # 'exact_ticker', 'exact_name', 'fuzzy'
                'match_score': float  # Similarity score (0-1)
            }
        """
        # Check blacklist
        if entity_text.lower() in self.blacklist:
            logger.debug(f"Entity '{entity_text}' is blacklisted")
            return None

        # Strip dollar sign if present
        if entity_text.startswith("$"):
            entity_text = entity_text[1:]

        # 1. Try exact ticker match
        stock = self.stock_db.get_by_ticker(entity_text)
        if stock:
            return {
                "ticker": stock["ticker"],
                "company_name": stock["company_name"],
                "match_type": "exact_ticker",
                "match_score": 1.0,
            }

        # 2. Try exact name match
        stock = self.stock_db.get_by_name(entity_text)
        if stock:
            return {
                "ticker": stock["ticker"],
                "company_name": stock["company_name"],
                "match_type": "exact_name",
                "match_score": 1.0,
            }

        # 3. Try fuzzy matching
        best_match = self._fuzzy_match(entity_text, threshold)
        if best_match:
            ticker, score = best_match
            stock = self.stock_db.get_by_ticker(ticker)
            if stock:
                return {
                    "ticker": stock["ticker"],
                    "company_name": stock["company_name"],
                    "match_type": "fuzzy",
                    "match_score": score,
                }

        logger.debug(f"Could not resolve entity '{entity_text}' to ticker")
        return None

    def _fuzzy_match(
        self, entity_text: str, threshold: float
    ) -> Optional[Tuple[str, float]]:
        """
        Find best fuzzy match for entity text.

        Args:
            entity_text: Entity text to match
            threshold: Minimum similarity score

        Returns:
            Tuple of (ticker, score) or None if no match above threshold
        """
        entity_lower = entity_text.lower()
        best_ticker = None
        best_score = 0.0

        # Search through all stocks (limit to top candidates for performance)
        candidates = self.stock_db.search(entity_text, limit=20)

        for stock in candidates:
            # Compare with company name
            company_name = stock["company_name"].lower()
            score = fuzz.ratio(entity_lower, company_name) / 100.0

            if score > best_score:
                best_score = score
                best_ticker = stock["ticker"]

            # Compare with common names
            for common_name in stock.get("common_names", []):
                common_lower = common_name.lower()
                score = fuzz.ratio(entity_lower, common_lower) / 100.0

                if score > best_score:
                    best_score = score
                    best_ticker = stock["ticker"]

        if best_score >= threshold and best_ticker:
            return (best_ticker, best_score)

        return None

    def resolve_batch(
        self, entity_texts: List[str], threshold: float = 0.85
    ) -> Dict[str, Optional[Dict]]:
        """
        Resolve multiple entities at once.

        Args:
            entity_texts: List of entity texts
            threshold: Minimum similarity score for fuzzy matching

        Returns:
            Dict mapping entity text to resolved stock info (or None)
        """
        results = {}
        for entity_text in entity_texts:
            results[entity_text] = self.resolve(entity_text, threshold)
        return results


def resolve_entity_to_ticker(
    entity_text: str,
    stock_database: Optional[StockDatabase] = None,
    threshold: float = 0.85,
) -> Optional[str]:
    """
    Convenience function to resolve entity to ticker symbol only.

    Args:
        entity_text: Entity text to resolve
        stock_database: StockDatabase instance (creates new if None)
        threshold: Minimum similarity score for fuzzy matching

    Returns:
        Ticker symbol or None if not resolved

    Example:
        >>> ticker = resolve_entity_to_ticker("Apple")
        >>> print(ticker)
        'AAPL'
    """
    resolver = EntityResolver(stock_database)
    result = resolver.resolve(entity_text, threshold)
    return result["ticker"] if result else None
