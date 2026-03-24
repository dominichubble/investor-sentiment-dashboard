"""
Tests for company entity extraction module.
"""

import pytest

from app.entities.company_extractor import (
    extract_company_entities,
    extract_financial_keywords,
    is_likely_stock_mention,
)


class TestCompanyExtractor:
    """Test suite for company entity extraction."""

    def test_extract_company_entities_basic(self):
        """Test basic company entity extraction."""
        text = "Apple reported strong earnings yesterday."
        entities = extract_company_entities(text)

        assert len(entities) > 0
        assert any(e["text"] == "Apple" for e in entities)
        assert entities[0]["label"] in ["ORG", "PRODUCT", "GPE"]

    def test_extract_multiple_entities(self):
        """Test extracting multiple company entities."""
        text = "Apple and Microsoft announced partnership while Tesla remained quiet."
        entities = extract_company_entities(text)

        # Should find at least 2-3 companies
        assert len(entities) >= 2

        entity_texts = [e["text"] for e in entities]
        assert "Apple" in entity_texts or "Microsoft" in entity_texts

    def test_extract_with_context(self):
        """Test extracting entities with context."""
        text = "Apple reported strong earnings yesterday."
        entities = extract_company_entities(text, include_context=True)

        assert len(entities) > 0
        assert "context" in entities[0]
        assert len(entities[0]["context"]) > 0

    def test_entity_positions(self):
        """Test that entity positions are correct."""
        text = "Apple reported earnings"
        entities = extract_company_entities(text)

        if entities:
            entity = entities[0]
            assert "start" in entity
            assert "end" in entity
            assert entity["start"] >= 0
            assert entity["end"] <= len(text)
            assert text[entity["start"] : entity["end"]] == entity["text"]

    def test_empty_text(self):
        """Test with empty text."""
        entities = extract_company_entities("")
        assert len(entities) == 0

    def test_no_entities(self):
        """Test text with no recognizable entities."""
        text = "The stock market went up today by five percent."
        entities = extract_company_entities(text)

        # May or may not find entities depending on spaCy model
        # Just ensure it doesn't crash
        assert isinstance(entities, list)


class TestFinancialKeywords:
    """Test suite for financial keyword detection."""

    def test_stock_keywords(self):
        """Test detection of stock-related keywords."""
        text = "AAPL stock surged in the market today"
        keywords = extract_financial_keywords(text)

        assert keywords["has_stock_keywords"] is True
        assert keywords["is_financial_context"] is True

    def test_company_keywords(self):
        """Test detection of company keywords."""
        text = "Apple Inc. announced new products"
        keywords = extract_financial_keywords(text)

        assert keywords["has_company_keywords"] is True
        assert keywords["is_financial_context"] is True

    def test_performance_keywords(self):
        """Test detection of performance keywords."""
        text = "Q4 earnings showed strong revenue growth"
        keywords = extract_financial_keywords(text)

        assert keywords["has_performance_keywords"] is True
        assert keywords["is_financial_context"] is True

    def test_price_keywords(self):
        """Test detection of price movement keywords."""
        text = "Stock price surged 15% after earnings"
        keywords = extract_financial_keywords(text)

        assert keywords["has_price_keywords"] is True
        assert keywords["is_financial_context"] is True

    def test_no_financial_keywords(self):
        """Test text without financial keywords."""
        text = "I ate an apple for breakfast"
        keywords = extract_financial_keywords(text)

        assert keywords["is_financial_context"] is False

    def test_mixed_keywords(self):
        """Test text with multiple types of keywords."""
        text = "Apple Inc. stock surged 20% on strong Q4 earnings"
        keywords = extract_financial_keywords(text)

        assert keywords["has_stock_keywords"] is True
        assert keywords["has_company_keywords"] is True
        assert keywords["has_performance_keywords"] is True
        assert keywords["has_price_keywords"] is True
        assert keywords["is_financial_context"] is True


class TestLikelyStockMention:
    """Test suite for stock mention likelihood detection."""

    def test_org_in_financial_context(self):
        """Test ORG entity in financial context."""
        entity = {"text": "Apple", "label": "ORG"}
        text = "Apple stock surged 15% on earnings"

        assert is_likely_stock_mention(entity, text) is True

    def test_org_without_context(self):
        """Test ORG entity without financial context."""
        entity = {"text": "Apple", "label": "ORG"}
        text = "I like apples and oranges"

        # ORG entities are always considered likely
        assert is_likely_stock_mention(entity, text) is True

    def test_product_with_stock_keywords(self):
        """Test PRODUCT entity with stock keywords."""
        entity = {"text": "iPhone", "label": "PRODUCT"}
        text = "iPhone stock continues to rise"

        assert is_likely_stock_mention(entity, text) is True

    def test_product_without_stock_keywords(self):
        """Test PRODUCT entity without stock keywords."""
        entity = {"text": "iPhone", "label": "PRODUCT"}
        text = "I bought a new iPhone yesterday"

        assert is_likely_stock_mention(entity, text) is False

    def test_gpe_with_financial_context(self):
        """Test GPE entity with financial context."""
        entity = {"text": "China", "label": "GPE"}
        text = "China stock market rallied today"

        assert is_likely_stock_mention(entity, text) is True

    def test_gpe_without_financial_context(self):
        """Test GPE entity without financial context."""
        entity = {"text": "China", "label": "GPE"}
        text = "I visited China last summer"

        assert is_likely_stock_mention(entity, text) is False
