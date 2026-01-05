"""
FinBERT Tokenizer Module

Provides tokenization, padding, truncation, and attention mask creation
specifically for FinBERT sentiment analysis inference.

This module handles the conversion of raw text into tensors suitable for
FinBERT model input, including proper padding and attention masks.

Usage:
    from app.preprocessing.finbert_tokenizer import tokenize_for_inference, tokenize_batch

    # Single text tokenization
    inputs = tokenize_for_inference("Stock prices are rising")
    # Returns: {'input_ids': tensor, 'attention_mask': tensor}

    # Batch tokenization
    texts = ["Market up", "Crisis ahead", "Neutral news"]
    inputs = tokenize_batch(texts, batch_size=2)
"""

import logging
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from ..logging_config import log_exception

logger = logging.getLogger(__name__)

# FinBERT model identifier
FINBERT_MODEL = "ProsusAI/finbert"

# Global tokenizer cache
_tokenizer_cache: Optional[AutoTokenizer] = None

# Maximum reasonable text length (characters)
MAX_REASONABLE_LENGTH = 50000


class TokenizationError(Exception):
    """Exception raised when tokenization fails."""
    pass


def get_tokenizer(cache_dir: Optional[str] = None) -> AutoTokenizer:
    """
    Load and cache the FinBERT tokenizer.

    Args:
        cache_dir: Optional directory to cache the tokenizer.
                   If None, uses default Hugging Face cache.

    Returns:
        AutoTokenizer instance for FinBERT

    Raises:
        RuntimeError: If tokenizer fails to load

    Example:
        >>> tokenizer = get_tokenizer()
        >>> tokenizer.model_max_length
        512
    """
    global _tokenizer_cache

    if _tokenizer_cache is not None:
        logger.debug("Using cached FinBERT tokenizer")
        return _tokenizer_cache

    try:
        logger.info(f"Loading FinBERT tokenizer: {FINBERT_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(
            FINBERT_MODEL,
            cache_dir=cache_dir,
            use_fast=True,  # Use fast tokenizer for better performance
        )
        _tokenizer_cache = tokenizer
        logger.info("✓ FinBERT tokenizer loaded successfully")
        return tokenizer

    except OSError as e:
        logger.error(f"Failed to load tokenizer - network or file error: {e}")
        raise RuntimeError(
            f"Failed to load FinBERT tokenizer. Check network connection or cache directory. Error: {e}"
        ) from e
    except Exception as e:
        log_exception(logger, e, "Unexpected error loading FinBERT tokenizer")
        raise RuntimeError(f"Failed to load FinBERT tokenizer: {e}") from e


def tokenize_for_inference(
    text: str,
    max_length: int = 512,
    padding: Union[bool, str] = True,
    truncation: bool = True,
    return_tensors: str = "pt",
    add_special_tokens: bool = True,
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a single text for FinBERT inference.

    Converts raw text into PyTorch tensors with proper padding, truncation,
    and attention masks suitable for FinBERT model input.

    Args:
        text: Raw text string to tokenize
        max_length: Maximum sequence length (default: 512, FinBERT's max)
        padding: Whether to pad sequences. True pads to max_length,
                 'max_length' pads to max_length, False for no padding
        truncation: Whether to truncate sequences longer than max_length
        return_tensors: Format for returned tensors ('pt' for PyTorch)
        add_special_tokens: Whether to add [CLS] and [SEP] tokens
        cache_dir: Optional directory to cache the tokenizer

    Returns:
        Dictionary containing:
            - input_ids: Tensor of token IDs [1, seq_len]
            - attention_mask: Tensor of attention masks [1, seq_len]
            - (optional) token_type_ids: Tensor of token type IDs

    Raises:
        ValueError: If text is empty or not a string
        RuntimeError: If tokenization fails

    Example:
        >>> inputs = tokenize_for_inference("Stock prices surged today")
        >>> inputs['input_ids'].shape
        torch.Size([1, 512])
        >>> inputs['attention_mask'].sum()  # Number of non-padding tokens
        tensor(8)
    """
    # Validate input
    if not isinstance(text, str):
        logger.error(f"Invalid text type: {type(text).__name__}")
        raise ValueError(f"Text must be a string, got {type(text)}")

    if not text or text.isspace():
        logger.error("Empty or whitespace-only text provided")
        raise ValueError("Text input cannot be empty or whitespace-only")
    
    # Check for extremely long texts
    text_length = len(text)
    if text_length > MAX_REASONABLE_LENGTH:
        logger.error(
            f"Text is extremely long ({text_length} chars). "
            f"Maximum reasonable length is {MAX_REASONABLE_LENGTH}."
        )
        raise ValueError(
            f"Text is too long ({text_length} chars). "
            f"Please truncate to under {MAX_REASONABLE_LENGTH} characters."
        )
    
    # Warn about long texts that might be truncated
    if text_length > 2000:
        logger.warning(
            f"Text is long ({text_length} chars) and will likely be truncated "
            f"to {max_length} tokens. Consider pre-processing."
        )

    try:
        tokenizer = get_tokenizer(cache_dir=cache_dir)

        logger.debug(f"Tokenizing text: {text[:50]}..." if len(text) > 50 else text)

        # Tokenize with all required parameters
        inputs = tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )

        logger.debug(f"Tokenization complete. Input shape: {inputs['input_ids'].shape}")
        
        # Check if text was truncated
        num_tokens = inputs['input_ids'].shape[-1]
        if num_tokens >= max_length:
            logger.warning(
                f"Text was truncated to {max_length} tokens. "
                f"Original length: {text_length} chars"
            )

        return inputs

    except ValueError as e:
        logger.error(f"Tokenization validation failed: {e}")
        raise TokenizationError(f"Tokenization validation failed: {e}") from e
    except RuntimeError as e:
        logger.error(f"Tokenizer runtime error: {e}")
        raise TokenizationError(f"Tokenizer runtime error: {e}") from e
    except Exception as e:
        log_exception(logger, e, f"Tokenization failed for text: {text[:100]}...")
        raise TokenizationError(f"Tokenization failed: {e}") from e


def tokenize_batch(
    texts: List[str],
    max_length: int = 512,
    padding: Union[bool, str] = True,
    truncation: bool = True,
    return_tensors: str = "pt",
    add_special_tokens: bool = True,
    batch_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize multiple texts for FinBERT inference with batch processing.

    Efficiently converts a list of texts into batched PyTorch tensors
    with automatic padding to the longest sequence in the batch.

    Args:
        texts: List of raw text strings to tokenize
        max_length: Maximum sequence length (default: 512)
        padding: Padding strategy. True pads to longest in batch,
                 'max_length' pads all to max_length
        truncation: Whether to truncate sequences longer than max_length
        return_tensors: Format for returned tensors ('pt' for PyTorch)
        add_special_tokens: Whether to add [CLS] and [SEP] tokens
        batch_size: If provided, process texts in mini-batches (for very large inputs)
                    If None, processes all texts at once
        cache_dir: Optional directory to cache the tokenizer

    Returns:
        Dictionary containing:
            - input_ids: Tensor of token IDs [batch_size, seq_len]
            - attention_mask: Tensor of attention masks [batch_size, seq_len]
            - (optional) token_type_ids: Tensor of token type IDs

    Raises:
        ValueError: If texts is empty or contains invalid inputs
        RuntimeError: If tokenization fails

    Example:
        >>> texts = ["Market up 5%", "Stock crash", "Neutral trading"]
        >>> inputs = tokenize_batch(texts, padding='longest')
        >>> inputs['input_ids'].shape
        torch.Size([3, 10])  # 3 texts, padded to longest (10 tokens)
        >>> inputs['attention_mask'].sum(dim=1)  # Tokens per text
        tensor([6, 5, 5])
    """
    # Validate inputs
    if not texts:
        logger.error("Empty texts list provided")
        raise ValueError("texts list cannot be empty")

    if not isinstance(texts, list):
        logger.error(f"Invalid texts type: {type(texts).__name__}")
        raise ValueError(f"texts must be a list, got {type(texts)}")

    # Check for empty strings and invalid types
    invalid_items = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            invalid_items.append((i, "non-string", type(t).__name__))
        elif not t or not t.strip():
            invalid_items.append((i, "empty", "empty string"))
        elif len(t) > MAX_REASONABLE_LENGTH:
            invalid_items.append((i, "too_long", f"{len(t)} chars"))
    
    if invalid_items:
        logger.error(f"Found {len(invalid_items)} invalid texts")
        error_details = [f"Index {i}: {reason} ({detail})" for i, reason, detail in invalid_items[:5]]
        raise ValueError(
            f"Found {len(invalid_items)} invalid texts. First few: {'; '.join(error_details)}"
        )
    
    # Warn about long texts
    long_texts = [(i, len(t)) for i, t in enumerate(texts) if len(t) > 2000]
    if long_texts:
        logger.warning(
            f"Found {len(long_texts)} texts longer than 2000 chars. "
            f"These will likely be truncated to {max_length} tokens."
        )

    try:
        tokenizer = get_tokenizer(cache_dir=cache_dir)

        logger.info(f"Tokenizing batch of {len(texts)} texts")

        # If batch_size is specified, process in chunks
        if batch_size is not None and len(texts) > batch_size:
            logger.debug(f"Processing in mini-batches of size {batch_size}")

            all_input_ids = []
            all_attention_masks = []
            all_token_type_ids = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                
                try:
                    batch_inputs = tokenizer(
                        batch_texts,
                        max_length=max_length,
                        padding=padding,
                        truncation=truncation,
                        return_tensors=return_tensors,
                        add_special_tokens=add_special_tokens,
                    )

                    all_input_ids.append(batch_inputs["input_ids"])
                    all_attention_masks.append(batch_inputs["attention_mask"])
                    if "token_type_ids" in batch_inputs:
                        all_token_type_ids.append(batch_inputs["token_type_ids"])
                
                except Exception as batch_error:
                    logger.error(
                        f"Failed to tokenize mini-batch starting at index {i}: {batch_error}"
                    )
                    raise TokenizationError(
                        f"Mini-batch tokenization failed at index {i}: {batch_error}"
                    ) from batch_error

            # Concatenate all batches
            try:
                inputs = {
                    "input_ids": torch.cat(all_input_ids, dim=0),
                    "attention_mask": torch.cat(all_attention_masks, dim=0),
                }
                if all_token_type_ids:
                    inputs["token_type_ids"] = torch.cat(all_token_type_ids, dim=0)
            except Exception as concat_error:
                logger.error(f"Failed to concatenate tokenized batches: {concat_error}")
                raise TokenizationError(
                    f"Failed to concatenate tokenized batches: {concat_error}"
                ) from concat_error

        else:
            # Process all texts at once
            inputs = tokenizer(
                texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens,
            )

        logger.info(f"Batch tokenization complete. Shape: {inputs['input_ids'].shape}")

        return inputs

    except TokenizationError:
        # Re-raise our custom exceptions
        raise
    except ValueError as e:
        logger.error(f"Validation error during batch tokenization: {e}")
        raise TokenizationError(f"Batch tokenization validation failed: {e}") from e
    except RuntimeError as e:
        logger.error(f"Runtime error during batch tokenization: {e}")
        raise TokenizationError(f"Batch tokenization runtime error: {e}") from e
    except Exception as e:
        log_exception(logger, e, "Unexpected error during batch tokenization")
        raise TokenizationError(f"Batch tokenization failed: {e}") from e


def get_tokenizer_info(cache_dir: Optional[str] = None) -> Dict[str, Union[str, int]]:
    """
    Get information about the FinBERT tokenizer.

    Args:
        cache_dir: Optional directory where tokenizer is cached

    Returns:
        Dictionary with tokenizer metadata:
            - model_name: Name of the model
            - vocab_size: Size of the vocabulary
            - model_max_length: Maximum sequence length
            - pad_token: Padding token
            - cls_token: Classification token
            - sep_token: Separator token
            - unk_token: Unknown token

    Example:
        >>> info = get_tokenizer_info()
        >>> info['model_max_length']
        512
        >>> info['vocab_size']
        30522
    """
    tokenizer = get_tokenizer(cache_dir=cache_dir)

    return {
        "model_name": FINBERT_MODEL,
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "pad_token": tokenizer.pad_token,
        "cls_token": tokenizer.cls_token,
        "sep_token": tokenizer.sep_token,
        "unk_token": tokenizer.unk_token,
        "mask_token": getattr(tokenizer, "mask_token", None),
    }


def clear_tokenizer_cache():
    """
    Clear the cached tokenizer instance.

    Useful for testing or when you need to reload the tokenizer
    with different parameters.

    Example:
        >>> clear_tokenizer_cache()
        >>> tokenizer = get_tokenizer(cache_dir="/custom/path")
    """
    global _tokenizer_cache
    _tokenizer_cache = None
    logger.debug("Tokenizer cache cleared")
