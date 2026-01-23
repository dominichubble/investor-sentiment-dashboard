"""
Company name extraction using Named Entity Recognition (NER).

Uses spaCy for extracting organization entities that might be publicly traded stocks.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Lazy load spaCy model
_nlp = None


def _get_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Please run: python -m spacy download en_core_web_sm"
            )
            raise
    return _nlp


def extract_company_entities(
    text: str, include_context: bool = False
) -> List[Dict]:
    """
    Extract organization entities that might be stocks using NER.

    Args:
        text: Input text to extract entities from
        include_context: Whether to include surrounding context

    Returns:
        List of dicts with entity information:
        {
            'text': str,          # Entity text (e.g., "Apple")
            'label': str,         # Entity label (e.g., "ORG")
            'start': int,         # Start character position
            'end': int,           # End character position
            'context': str        # Surrounding context (if include_context=True)
        }

    Example:
        >>> entities = extract_company_entities("Apple reported strong earnings")
        >>> print(entities[0]['text'])
        'Apple'
    """
    nlp = _get_nlp()
    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        # Focus on organizations, products, and geo-political entities
        # ORG: Companies, agencies, institutions
        # PRODUCT: Objects, vehicles, foods, etc. (might include product brands)
        # GPE: Countries, cities, states (some companies named after places)
        if ent.label_ in ["ORG", "PRODUCT", "GPE"]:
            entity_info = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }

            # Add surrounding context if requested
            if include_context:
                # Extract sentence containing the entity
                entity_info["context"] = ent.sent.text if ent.sent else text

            entities.append(entity_info)

    logger.debug(f"Extracted {len(entities)} entities from text")
    return entities


def extract_financial_keywords(text: str) -> Dict[str, bool]:
    """
    Detect financial keywords that indicate text is about stocks/finance.

    This helps filter out false positives (e.g., "Apple" the fruit vs Apple Inc.)

    Args:
        text: Input text

    Returns:
        Dict with boolean flags for detected financial keywords
    """
    text_lower = text.lower()

    # Financial keywords
    stock_keywords = [
        "stock",
        "share",
        "ticker",
        "market",
        "nasdaq",
        "nyse",
        "dow",
        "s&p",
    ]
    company_keywords = [
        "company",
        "inc",
        "corp",
        "corporation",
        "ltd",
        "limited",
    ]
    performance_keywords = [
        "earnings",
        "revenue",
        "profit",
        "loss",
        "quarter",
        "q1",
        "q2",
        "q3",
        "q4",
    ]
    price_keywords = [
        "price",
        "up",
        "down",
        "surge",
        "drop",
        "rise",
        "fall",
        "gain",
        "lose",
        "%",
    ]

    return {
        "has_stock_keywords": any(kw in text_lower for kw in stock_keywords),
        "has_company_keywords": any(
            kw in text_lower for kw in company_keywords
        ),
        "has_performance_keywords": any(
            kw in text_lower for kw in performance_keywords
        ),
        "has_price_keywords": any(kw in text_lower for kw in price_keywords),
        "is_financial_context": any(
            kw in text_lower
            for kw in stock_keywords
            + company_keywords
            + performance_keywords
            + price_keywords
        ),
    }


def is_likely_stock_mention(entity: Dict, text: str) -> bool:
    """
    Determine if an entity is likely a stock/company mention.

    Uses context clues and entity characteristics.

    Args:
        entity: Entity dict from extract_company_entities()
        text: Full text

    Returns:
        True if entity is likely a stock mention
    """
    # Check if text has financial context
    financial_context = extract_financial_keywords(text)

    # If strong financial context, assume entity is stock-related
    if financial_context["is_financial_context"]:
        return True

    # ORG entities are most likely to be companies
    if entity["label"] == "ORG":
        return True

    # PRODUCT entities are less likely unless in financial context
    if entity["label"] == "PRODUCT":
        return financial_context["has_stock_keywords"]

    # GPE entities are unlikely unless explicitly financial
    if entity["label"] == "GPE":
        return financial_context["has_stock_keywords"]

    return False
