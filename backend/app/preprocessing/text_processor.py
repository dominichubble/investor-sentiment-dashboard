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
from functools import lru_cache
from typing import Dict, List, Optional, Set, Union

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:
    raise ImportError(
        "NLTK is required for text preprocessing. " "Install it with: pip install nltk"
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
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),\\]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
_EMAIL_PATTERN = re.compile(r"\S+@\S+")
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#(\w+)")
_NUMBER_PATTERN = re.compile(r"\b\d+\.?\d*[kKmMbB]?\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NEGATION_PATTERN = re.compile(
    r"\b(not|no|never|neither|nobody|nothing|nowhere|n't)\b", re.IGNORECASE
)
_TICKER_PATTERN = re.compile(r"\$[A-Z]{1,5}\b")  # $AAPL, $TSLA, etc.
_STOCK_MOVEMENT_PATTERN = re.compile(
    r"\b(up|down|gained?|lost|rose|fell|increased?|decreased?)\s+[\d.]+\s*[%$]|[\$][\d.]+",
    re.IGNORECASE,
)
_CASHTAG_PATTERN = re.compile(r"\$[A-Z]{1,5}")  # Standalone cashtags

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

# Intensity modifiers that affect sentiment strength
INTENSITY_MODIFIERS = {
    "very",
    "extremely",
    "highly",
    "significantly",
    "strongly",
    "moderately",
    "slightly",
    "barely",
    "somewhat",
    "quite",
    "rather",
    "absolutely",
    "completely",
    "totally",
    "exceptionally",
    "remarkably",
}


def extract_tickers(text: str) -> Set[str]:
    """
    Extract stock ticker symbols from text.

    Recognizes patterns like $AAPL, $TSLA, $MSFT

    Args:
        text: Input text string

    Returns:
        Set of ticker symbols (without $ prefix)

    Examples:
        >>> extract_tickers("$AAPL is up 5%, $TSLA down 3%")
        {'AAPL', 'TSLA'}
    """
    if not text or not isinstance(text, str):
        return set()

    matches = _TICKER_PATTERN.findall(text)
    # Remove $ prefix and return unique tickers
    return {ticker[1:] for ticker in matches}


def detect_stock_movements(text: str) -> List[Dict[str, str]]:
    """
    Detect stock price movement patterns in text.

    Recognizes patterns like "up 5%", "down $2.50", "gained 10%"

    Args:
        text: Input text string

    Returns:
        List of dictionaries with movement information

    Examples:
        >>> detect_stock_movements("Stock up 25% today")
        [{'movement': 'up 25%', 'direction': 'up', 'value': '25%'}]
    """
    if not text or not isinstance(text, str):
        return []

    matches = _STOCK_MOVEMENT_PATTERN.findall(text)
    movements = []

    for match in _STOCK_MOVEMENT_PATTERN.finditer(text):
        movement_text = match.group(0)
        direction = match.group(1).lower()

        # Normalize direction
        if direction in ["gained", "gain", "rose", "increased", "increase", "up"]:
            normalized_direction = "positive"
        else:
            normalized_direction = "negative"

        movements.append(
            {
                "movement": movement_text,
                "direction": normalized_direction,
                "raw_direction": direction,
            }
        )

    return movements


def normalize_text(
    text: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_mentions: bool = True,
    expand_hashtags: bool = True,
    remove_numbers: bool = False,
    lowercase: bool = True,
    preserve_financial_punctuation: bool = False,
    handle_negations: bool = False,
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
        preserve_financial_punctuation: Keep %, $, and decimal points (for FinBERT)
        handle_negations: Mark negations to preserve sentiment context (for FinBERT)

    Returns:
        Normalized text string
    """
    if not text or not isinstance(text, str):
        return ""

    # Handle negations before processing (for FinBERT)
    if handle_negations:
        # Mark negations: "not good" -> "not_good", "isn't profitable" -> "isnt_profitable"
        words = text.split()
        processed_words = []
        i = 0
        while i < len(words):
            word = words[i]
            # Check if current word is a negation
            if _NEGATION_PATTERN.match(word) and i + 1 < len(words):
                # Combine negation with next word
                next_word = words[i + 1]
                # Remove punctuation from negation word only
                clean_neg = word.rstrip(".,!?;:")
                processed_words.append(f"{clean_neg}_{next_word}")
                i += 2  # Skip next word as it's now combined
            else:
                processed_words.append(word)
                i += 1
        text = " ".join(processed_words)

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

    # Remove punctuation (but preserve financial punctuation and underscores if requested)
    if preserve_financial_punctuation:
        # Keep % $ and decimal points, remove others (keep underscore for negations)
        # Create translation table that keeps %, $, ., _
        punct_to_remove = (
            string.punctuation.replace("%", "")
            .replace("$", "")
            .replace(".", "")
            .replace("_", "")
        )
        text = text.translate(str.maketrans("", "", punct_to_remove))
    elif handle_negations:
        # Keep underscores for negations
        punct_to_remove = string.punctuation.replace("_", "")
        text = text.translate(str.maketrans("", "", punct_to_remove))
    else:
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
        stop_words = (
            stop_words - INTENSITY_MODIFIERS
        )  # Also preserve intensity modifiers

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
        lemmatized = [_cached_lemmatize(token.lower(), lemmatizer) for token in tokens]
        return lemmatized
    except Exception:
        # Return original tokens if lemmatization fails
        return tokens


@lru_cache(maxsize=10000)
def _cached_lemmatize(word: str, lemmatizer: WordNetLemmatizer) -> str:
    """
    Cached lemmatization for performance.

    Args:
        word: Word to lemmatize
        lemmatizer: WordNetLemmatizer instance

    Returns:
        Lemmatized word
    """
    return lemmatizer.lemmatize(word)


def calculate_preprocessing_quality(
    original: str, processed: str, tokens: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate quality metrics for preprocessing.

    Helps assess information retention and preprocessing impact.

    Args:
        original: Original text before preprocessing
        processed: Processed text after preprocessing
        tokens: Optional list of tokens (will tokenize if not provided)

    Returns:
        Dictionary with quality metrics:
        - retention_rate: Ratio of tokens kept (0-1)
        - unique_token_ratio: Vocabulary diversity (0-1)
        - financial_term_density: Percentage of financial terms (0-1)
        - avg_token_length: Average character length of tokens
        - ticker_count: Number of stock tickers detected
        - has_negations: Whether negations are present

    Examples:
        >>> original = "The stock market is very bullish! $AAPL up 25%"
        >>> processed = "stock market very bullish $AAPL up 25%"
        >>> metrics = calculate_preprocessing_quality(original, processed)
        >>> metrics['retention_rate']  # ~0.78 (7 out of 9 words)
        0.78
    """
    if not original or not processed:
        return {
            "retention_rate": 0.0,
            "unique_token_ratio": 0.0,
            "financial_term_density": 0.0,
            "avg_token_length": 0.0,
            "ticker_count": 0,
            "has_negations": False,
        }

    # Get tokens
    if tokens is None:
        tokens = processed.split()

    orig_tokens = original.split()
    proc_tokens = tokens if tokens else processed.split()

    # Calculate metrics
    retention_rate = len(proc_tokens) / len(orig_tokens) if orig_tokens else 0.0

    unique_token_ratio = (
        len(set(proc_tokens)) / len(proc_tokens) if proc_tokens else 0.0
    )

    financial_count = sum(
        1 for token in proc_tokens if token.lower() in FINANCIAL_TERMS
    )
    financial_term_density = financial_count / len(proc_tokens) if proc_tokens else 0.0

    avg_token_length = (
        sum(len(token) for token in proc_tokens) / len(proc_tokens)
        if proc_tokens
        else 0.0
    )

    # Detect tickers and negations
    tickers = extract_tickers(processed)
    has_negations = "_" in processed or any(
        word in processed.lower() for word in ["not", "no", "never", "neither"]
    )

    return {
        "retention_rate": round(retention_rate, 3),
        "unique_token_ratio": round(unique_token_ratio, 3),
        "financial_term_density": round(financial_term_density, 3),
        "avg_token_length": round(avg_token_length, 2),
        "ticker_count": len(tickers),
        "has_negations": has_negations,
    }


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_stopwords_flag: bool = False,
    lemmatize: bool = False,
    return_string: bool = True,
    preserve_financial: bool = True,
    preserve_financial_punctuation: bool = False,
    handle_negations: bool = False,
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
        preserve_financial_punctuation: Keep %, $, and decimal points (for FinBERT)
        handle_negations: Mark negations to preserve sentiment context (for FinBERT)
        custom_stopwords: Additional stopwords to remove

    Returns:
        Preprocessed text as string or list of tokens

    Examples:
        >>> preprocess_text("The stocks are rising! #bullish")
        'stocks rising bullish'

        >>> preprocess_text("Markets crashed", remove_stopwords_flag=True, return_string=False)
        ['markets', 'crashed']

        >>> preprocess_text("Stock up 25%", preserve_financial_punctuation=True)
        'stock up 25%'

        >>> preprocess_text("not profitable", handle_negations=True)
        'not_profitable'
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
        preserve_financial_punctuation=preserve_financial_punctuation,
        handle_negations=handle_negations,
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
        preserve_financial_punctuation: Keep %, $, and decimal points (for FinBERT)
        handle_negations: Mark negations to preserve sentiment context (for FinBERT)
        custom_stopwords: Additional stopwords to remove

    Examples:
        >>> processor = TextProcessor(lowercase=True, remove_stopwords=True)
        >>> processor.process("The market is bullish!")
        ['market', 'bullish']

        >>> processor.process_batch(["Stock rising", "Market falling"])
        [['stock', 'rising'], ['market', 'falling']]

        >>> # FinBERT-optimized processor
        >>> finbert_processor = TextProcessor(
        ...     lowercase=False,
        ...     remove_stopwords=False,
        ...     lemmatize=False,
        ...     preserve_financial_punctuation=True,
        ...     handle_negations=True
        ... )
        >>> finbert_processor.process("Stock up 25%, not declining", return_string=True)
        'Stock up 25% not_declining'
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        preserve_financial: bool = True,
        preserve_financial_punctuation: bool = False,
        handle_negations: bool = False,
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
            preserve_financial_punctuation: Keep %, $, and decimal points (for FinBERT)
            handle_negations: Mark negations to preserve sentiment context (for FinBERT)
            custom_stopwords: Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.preserve_financial = preserve_financial
        self.preserve_financial_punctuation = preserve_financial_punctuation
        self.handle_negations = handle_negations
        self.custom_stopwords = custom_stopwords or set()

        # Cache NLTK resources for better performance
        self._stopwords_cache: Optional[Set[str]] = None
        self._lemmatizer_cache: Optional[WordNetLemmatizer] = None

    def _get_stopwords(self) -> Set[str]:
        """Get cached stopwords."""
        if self._stopwords_cache is None:
            try:
                self._stopwords_cache = set(stopwords.words("english"))
            except Exception:
                self._stopwords_cache = set()
        return self._stopwords_cache

    def _get_lemmatizer(self) -> WordNetLemmatizer:
        """Get cached lemmatizer."""
        if self._lemmatizer_cache is None:
            self._lemmatizer_cache = WordNetLemmatizer()
        return self._lemmatizer_cache

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
            preserve_financial_punctuation=self.preserve_financial_punctuation,
            handle_negations=self.handle_negations,
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
