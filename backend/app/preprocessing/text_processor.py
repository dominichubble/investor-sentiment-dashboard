#!/usr/bin/env python3
"""
Text Preprocessing Module

Provides functions for tokenization, stopword removal, lemmatization,
and text normalization for sentiment analysis on financial text data.

Usage:
    from backend.app.preprocessing import TextProcessor, preprocess_text

    # Quick preprocessing
    clean_text = preprocess_text("The stocks are rising!", remove_stopwords_flag=True)

    # Custom preprocessing pipeline
    processor = TextProcessor(lowercase=True, remove_stopwords=True)
    tokens = processor.process("Market sentiment is bullish")
"""

import re
import string
from typing import List, Optional, Set, Union

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:
    raise ImportError(
        "NLTK is required for text preprocessing. "
        "Install it with: pip install nltk"
    )

# Download required NLTK data
_NLTK_RESOURCES = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]


def _ensure_nltk_data():
    """Download required NLTK data if not already present."""
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass  # Silent fail if download doesn't work


# Initialize NLTK data
_ensure_nltk_data()

# Compile regex patterns for efficiency
_URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
_EMAIL_PATTERN = re.compile(r"\S+@\S+")
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#(\w+)")
_NUMBER_PATTERN = re.compile(r"\b\d+\.?\d*[kKmMbB]?\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Financial domain stopwords to preserve
FINANCIAL_TERMS = {
    "stock",
    "stocks",
    "market",
    "markets",
    "price",
    "prices",
    "share",
    "shares",
    "buy",
    "sell",
    "bull",
    "bullish",
    "bear",
    "bearish",
    "gain",
    "gains",
    "loss",
    "losses",
    "profit",
    "profits",
    "earnings",
    "revenue",
    "dividend",
    "dividends",
    "growth",
    "rally",
    "crash",
    "volatility",
    "risk",
    "return",
    "returns",
    "investment",
    "investor",
    "investors",
    "trading",
    "trade",
    "trades",
    "portfolio",
    "asset",
    "assets",
    "equity",
    "bond",
    "bonds",
    "fund",
    "funds",
    "index",
    "indices",
}


def normalize_text(
    text: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_mentions: bool = True,
    expand_hashtags: bool = True,
    remove_numbers: bool = False,
    lowercase: bool = True,
) -> str:
    """
    Normalize text by removing/replacing various patterns.

    Args:
        text: Input text string
        remove_urls: Remove URLs
        remove_emails: Remove email addresses
        remove_mentions: Remove @mentions
        expand_hashtags: Convert #HashTag to "Hash Tag"
        remove_numbers: Remove numeric values
        lowercase: Convert to lowercase

    Returns:
        Normalized text string
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove URLs
    if remove_urls:
        text = _URL_PATTERN.sub("", text)

    # Remove emails
    if remove_emails:
        text = _EMAIL_PATTERN.sub("", text)

    # Remove mentions
    if remove_mentions:
        text = _MENTION_PATTERN.sub("", text)

    # Expand hashtags: #CamelCase -> "Camel Case"
    if expand_hashtags:

        def expand_hashtag(match):
            word = match.group(1)
            # Insert space before uppercase letters
            expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", word)
            return expanded

        text = _HASHTAG_PATTERN.sub(expand_hashtag, text)

    # Remove numbers (optional - useful for some financial text)
    if remove_numbers:
        text = _NUMBER_PATTERN.sub("", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Normalize whitespace
    text = _WHITESPACE_PATTERN.sub(" ", text)

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words using NLTK's word_tokenize.

    Args:
        text: Input text string

    Returns:
        List of tokens
    """
    if not text or not isinstance(text, str):
        return []

    try:
        tokens = word_tokenize(text)
        return [t for t in tokens if t.strip()]
    except Exception:
        # Fallback to simple split if NLTK fails
        return text.split()


def remove_stopwords(
    tokens: List[str],
    language: str = "english",
    preserve_financial: bool = True,
    custom_stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Remove stopwords from token list.

    Args:
        tokens: List of tokens
        language: Language for stopwords (default: 'english')
        preserve_financial: Keep financial domain terms
        custom_stopwords: Additional stopwords to remove

    Returns:
        List of tokens with stopwords removed
    """
    if not tokens:
        return []

    try:
        stop_words = set(stopwords.words(language))
    except Exception:
        # Fallback to basic stopwords if NLTK fails
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
        }

    # Preserve financial terms
    if preserve_financial:
        stop_words = stop_words - FINANCIAL_TERMS

    # Add custom stopwords
    if custom_stopwords:
        stop_words = stop_words.union(custom_stopwords)

    # Filter tokens
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

    return filtered_tokens


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatize tokens using WordNet lemmatizer.

    Args:
        tokens: List of tokens

    Returns:
        List of lemmatized tokens
    """
    if not tokens:
        return []

    try:
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(token.lower()) for token in tokens]
        return lemmatized
    except Exception:
        # Return original tokens if lemmatization fails
        return tokens


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_stopwords_flag: bool = False,
    lemmatize: bool = False,
    return_string: bool = True,
    preserve_financial: bool = True,
    custom_stopwords: Optional[Set[str]] = None,
) -> Union[str, List[str]]:
    """
    Complete preprocessing pipeline for text.

    This function combines normalization, tokenization, stopword removal,
    and lemmatization into a single convenient function.

    Args:
        text: Input text string
        lowercase: Convert to lowercase
        remove_urls: Remove URLs and emails
        remove_stopwords_flag: Remove stopwords
        lemmatize: Apply lemmatization
        return_string: Return processed text as string (vs list of tokens)
        preserve_financial: Preserve financial domain terms when removing stopwords
        custom_stopwords: Additional stopwords to remove

    Returns:
        Preprocessed text as string or list of tokens

    Examples:
        >>> preprocess_text("The stocks are rising! #bullish")
        'stocks rising bullish'

        >>> preprocess_text("Markets crashed", remove_stopwords_flag=True, return_string=False)
        ['markets', 'crashed']
    """
    if not text or not isinstance(text, str):
        return "" if return_string else []

    # Step 1: Normalize text
    normalized = normalize_text(
        text,
        remove_urls=remove_urls,
        remove_emails=True,
        remove_mentions=True,
        expand_hashtags=True,
        remove_numbers=False,
        lowercase=lowercase,
    )

    # Step 2: Tokenize
    tokens = tokenize(normalized)

    # Step 3: Remove stopwords (optional)
    if remove_stopwords_flag and tokens:
        tokens = remove_stopwords(
            tokens,
            language="english",
            preserve_financial=preserve_financial,
            custom_stopwords=custom_stopwords,
        )

    # Step 4: Lemmatize (optional)
    if lemmatize and tokens:
        tokens = lemmatize_tokens(tokens)

    # Return as string or list
    if return_string:
        return " ".join(tokens)
    else:
        return tokens


class TextProcessor:
    """
    Configurable text preprocessing pipeline.

    This class provides a stateful processor that can be configured once
    and applied to multiple texts with consistent settings.

    Attributes:
        lowercase: Convert text to lowercase
        remove_urls: Remove URLs and links
        remove_stopwords: Remove stopwords
        lemmatize: Apply lemmatization
        preserve_financial: Keep financial domain terms
        custom_stopwords: Additional stopwords to remove

    Examples:
        >>> processor = TextProcessor(lowercase=True, remove_stopwords=True)
        >>> processor.process("The market is bullish!")
        ['market', 'bullish']

        >>> processor.process_batch(["Stock rising", "Market falling"])
        [['stock', 'rising'], ['market', 'falling']]
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        preserve_financial: bool = True,
        custom_stopwords: Optional[Set[str]] = None,
    ):
        """
        Initialize text processor with configuration.

        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs and links
            remove_stopwords: Remove stopwords
            lemmatize: Apply lemmatization
            preserve_financial: Keep financial domain terms
            custom_stopwords: Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.preserve_financial = preserve_financial
        self.custom_stopwords = custom_stopwords or set()

    def process(self, text: str, return_string: bool = False) -> Union[List[str], str]:
        """
        Process a single text.

        Args:
            text: Input text
            return_string: Return as string instead of token list

        Returns:
            Processed tokens or string
        """
        return preprocess_text(
            text,
            lowercase=self.lowercase,
            remove_urls=self.remove_urls,
            remove_stopwords_flag=self.remove_stopwords,
            lemmatize=self.lemmatize,
            return_string=return_string,
            preserve_financial=self.preserve_financial,
            custom_stopwords=self.custom_stopwords,
        )

    def process_batch(
        self, texts: List[str], return_strings: bool = False
    ) -> Union[List[List[str]], List[str]]:
        """
        Process multiple texts.

        Args:
            texts: List of input texts
            return_strings: Return as strings instead of token lists

        Returns:
            List of processed tokens or strings
        """
        results = []
        for text in texts:
            result = self.process(text, return_string=return_strings)
            results.append(result)
        return results
