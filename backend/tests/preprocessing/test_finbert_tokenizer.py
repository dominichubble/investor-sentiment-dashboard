"""
Tests for FinBERT Tokenizer Module

Tests tokenization, padding, truncation, and attention mask creation
for FinBERT sentiment analysis inference.
"""

import pytest
import torch
from transformers import AutoTokenizer

from app.preprocessing.finbert_tokenizer import (
    FINBERT_MODEL,
    clear_tokenizer_cache,
    get_tokenizer,
    get_tokenizer_info,
    tokenize_batch,
    tokenize_for_inference,
)


@pytest.fixture(autouse=True)
def reset_tokenizer_cache():
    """Clear tokenizer cache before each test."""
    clear_tokenizer_cache()
    yield
    clear_tokenizer_cache()


class TestGetTokenizer:
    """Test tokenizer loading and caching."""

    def test_get_tokenizer_loads_successfully(self):
        """Test that tokenizer loads without errors."""
        tokenizer = get_tokenizer()
        assert tokenizer is not None
        # Check that it's a valid tokenizer (AutoTokenizer returns specific subclass)
        assert hasattr(tokenizer, "vocab_size")
        assert hasattr(tokenizer, "model_max_length")

    def test_get_tokenizer_returns_finbert_tokenizer(self):
        """Test that the correct FinBERT tokenizer is loaded."""
        tokenizer = get_tokenizer()
        # Check vocabulary size matches FinBERT
        assert tokenizer.vocab_size == 30522

    def test_get_tokenizer_caches_instance(self):
        """Test that tokenizer is cached after first load."""
        tokenizer1 = get_tokenizer()
        tokenizer2 = get_tokenizer()
        assert tokenizer1 is tokenizer2  # Same object reference

    def test_get_tokenizer_with_cache_dir(self):
        """Test that tokenizer accepts custom cache directory."""
        # Should not raise an error
        tokenizer = get_tokenizer(cache_dir="/tmp/test_cache")
        assert tokenizer is not None

    def test_clear_tokenizer_cache(self):
        """Test that cache can be cleared."""
        tokenizer1 = get_tokenizer()
        clear_tokenizer_cache()
        tokenizer2 = get_tokenizer()
        # After clearing cache, should be different instances
        assert tokenizer1 is not tokenizer2


class TestTokenizeForInference:
    """Test single text tokenization."""

    def test_tokenize_simple_text(self):
        """Test tokenization of simple text."""
        text = "Stock prices are rising"
        inputs = tokenize_for_inference(text)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert isinstance(inputs["input_ids"], torch.Tensor)
        assert isinstance(inputs["attention_mask"], torch.Tensor)

    def test_tokenize_returns_correct_shapes(self):
        """Test that output tensors have correct shapes."""
        text = "Market sentiment is positive"
        inputs = tokenize_for_inference(text)

        # Shape should be [1, seq_len] for single text
        assert inputs["input_ids"].ndim == 2
        assert inputs["attention_mask"].ndim == 2
        assert inputs["input_ids"].shape[0] == 1  # Batch size of 1
        assert inputs["input_ids"].shape == inputs["attention_mask"].shape

    def test_tokenize_pads_to_max_length(self):
        """Test that short text is padded to max_length when padding=True."""
        text = "Bull market"
        inputs = tokenize_for_inference(text, max_length=512, padding="max_length")

        assert inputs["input_ids"].shape[1] == 512
        assert inputs["attention_mask"].shape[1] == 512

    def test_tokenize_truncates_long_text(self):
        """Test that long text is truncated to max_length."""
        long_text = "market " * 600  # Very long text
        inputs = tokenize_for_inference(long_text, max_length=128, truncation=True)

        assert inputs["input_ids"].shape[1] <= 128

    def test_tokenize_adds_special_tokens(self):
        """Test that [CLS] and [SEP] tokens are added."""
        text = "Stock up"
        inputs = tokenize_for_inference(text, add_special_tokens=True)

        tokenizer = get_tokenizer()
        token_ids = inputs["input_ids"][0].tolist()

        # First token should be [CLS]
        assert token_ids[0] == tokenizer.cls_token_id
        # Should contain [SEP] token
        assert tokenizer.sep_token_id in token_ids

    def test_tokenize_attention_mask_correct(self):
        """Test that attention mask correctly marks non-padding tokens."""
        text = "Short text"
        inputs = tokenize_for_inference(text, padding="max_length", max_length=20)

        # Attention mask should be 1 for real tokens, 0 for padding
        mask = inputs["attention_mask"][0]
        assert (mask == 0).any() or (mask == 1).any()  # Contains 0s or 1s
        assert mask.sum() > 0  # At least some real tokens
        assert mask.sum() < 20  # Not all tokens are real (some padding)

    def test_tokenize_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            tokenize_for_inference("")

    def test_tokenize_whitespace_only_raises_error(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            tokenize_for_inference("   \n\t  ")

    def test_tokenize_non_string_raises_error(self):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            tokenize_for_inference(123)

    def test_tokenize_with_financial_terms(self):
        """Test tokenization of financial domain text."""
        text = "The stock market crashed, causing bearish sentiment and losses"
        inputs = tokenize_for_inference(text)

        # Should successfully tokenize without errors
        assert inputs["input_ids"].shape[0] == 1
        assert inputs["attention_mask"].sum() > 0


class TestTokenizeBatch:
    """Test batch tokenization."""

    def test_tokenize_batch_multiple_texts(self):
        """Test tokenization of multiple texts."""
        texts = ["Market up", "Stock down", "Neutral trading"]
        inputs = tokenize_batch(texts)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"].shape[0] == 3  # Batch size of 3

    def test_tokenize_batch_returns_correct_shapes(self):
        """Test that batch outputs have correct shapes."""
        texts = ["Text one", "Text two longer", "Short"]
        inputs = tokenize_batch(texts, padding="longest")

        batch_size = len(texts)
        assert inputs["input_ids"].shape[0] == batch_size
        assert inputs["attention_mask"].shape[0] == batch_size
        # All texts should have same sequence length (padded)
        assert inputs["input_ids"].shape == inputs["attention_mask"].shape

    def test_tokenize_batch_pads_to_longest(self):
        """Test that texts are padded to longest sequence in batch."""
        texts = ["Short", "This is a much longer text with many tokens"]
        inputs = tokenize_batch(texts, padding="longest")

        # All sequences should have same length (longest in batch)
        seq_length = inputs["input_ids"].shape[1]
        assert seq_length > 5  # Longer than "Short"

        # Shorter text should have padding (attention_mask = 0)
        mask0 = inputs["attention_mask"][0]
        mask1 = inputs["attention_mask"][1]
        assert mask0.sum() < mask1.sum()  # First text has fewer real tokens

    def test_tokenize_batch_pads_to_max_length(self):
        """Test padding all sequences to max_length."""
        texts = ["A", "B", "C"]
        inputs = tokenize_batch(texts, padding="max_length", max_length=128)

        assert inputs["input_ids"].shape[1] == 128

    def test_tokenize_batch_truncates_long_texts(self):
        """Test that long texts are truncated in batch."""
        texts = ["short", "word " * 300]  # One short, one very long
        inputs = tokenize_batch(texts, max_length=64, truncation=True)

        assert inputs["input_ids"].shape[1] <= 64

    def test_tokenize_batch_with_mini_batches(self):
        """Test processing large batches with batch_size parameter."""
        texts = [f"Text number {i}" for i in range(10)]
        inputs = tokenize_batch(texts, batch_size=3)

        # Should process all texts correctly
        assert inputs["input_ids"].shape[0] == 10

    def test_tokenize_batch_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            tokenize_batch([])

    def test_tokenize_batch_with_empty_string_raises_error(self):
        """Test that list with empty string raises ValueError."""
        with pytest.raises(ValueError, match="empty or non-string"):
            tokenize_batch(["Valid text", "", "Another text"])

    def test_tokenize_batch_with_non_string_raises_error(self):
        """Test that list with non-string raises ValueError."""
        with pytest.raises(ValueError, match="empty or non-string"):
            tokenize_batch(["Text", 123, "More text"])

    def test_tokenize_batch_not_list_raises_error(self):
        """Test that non-list input raises ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            tokenize_batch("Not a list")

    def test_tokenize_batch_single_text(self):
        """Test that batch tokenization works with single text."""
        texts = ["Single text"]
        inputs = tokenize_batch(texts)

        assert inputs["input_ids"].shape[0] == 1

    def test_tokenize_batch_large_batch(self):
        """Test tokenization of large batch."""
        texts = [f"Market sentiment {i}" for i in range(100)]
        inputs = tokenize_batch(texts)

        assert inputs["input_ids"].shape[0] == 100

    def test_tokenize_batch_attention_masks_correct(self):
        """Test that attention masks are correct for varied length texts."""
        texts = ["Short", "This is much longer text", "Mid length text here"]
        inputs = tokenize_batch(texts, padding="longest")

        # Each text should have different number of real tokens
        token_counts = inputs["attention_mask"].sum(dim=1)
        assert token_counts[0] < token_counts[1]  # "Short" has fewer tokens
        assert token_counts[2] < token_counts[1]  # Middle < longest


class TestGetTokenizerInfo:
    """Test tokenizer information retrieval."""

    def test_get_tokenizer_info_returns_dict(self):
        """Test that tokenizer info returns a dictionary."""
        info = get_tokenizer_info()
        assert isinstance(info, dict)

    def test_get_tokenizer_info_contains_required_fields(self):
        """Test that info contains all expected fields."""
        info = get_tokenizer_info()

        required_fields = [
            "model_name",
            "vocab_size",
            "model_max_length",
            "pad_token",
            "cls_token",
            "sep_token",
            "unk_token",
        ]

        for field in required_fields:
            assert field in info

    def test_get_tokenizer_info_correct_model_name(self):
        """Test that model name matches FinBERT."""
        info = get_tokenizer_info()
        assert info["model_name"] == FINBERT_MODEL

    def test_get_tokenizer_info_correct_vocab_size(self):
        """Test that vocabulary size is correct for FinBERT."""
        info = get_tokenizer_info()
        assert info["vocab_size"] == 30522  # FinBERT uses BERT base vocab

    def test_get_tokenizer_info_correct_max_length(self):
        """Test that max length is correct."""
        info = get_tokenizer_info()
        assert info["model_max_length"] == 512  # BERT max length


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_tokenize_then_check_info(self):
        """Test that tokenizer info matches tokenization behavior."""
        info = get_tokenizer_info()
        text = "Test text"

        inputs = tokenize_for_inference(
            text, max_length=info["model_max_length"], padding="max_length"
        )

        assert inputs["input_ids"].shape[1] == info["model_max_length"]

    def test_single_vs_batch_tokenization_consistency(self):
        """Test that single tokenization matches batch with one text."""
        text = "Stock market sentiment"

        single_inputs = tokenize_for_inference(text, padding="max_length")
        batch_inputs = tokenize_batch([text], padding="max_length")

        # Should produce identical results
        assert torch.equal(single_inputs["input_ids"], batch_inputs["input_ids"])
        assert torch.equal(
            single_inputs["attention_mask"], batch_inputs["attention_mask"]
        )

    def test_batch_tokenization_preserves_order(self):
        """Test that batch tokenization preserves input order."""
        texts = ["First text", "Second text", "Third text"]
        inputs = tokenize_batch(texts, padding="longest")

        # Decode first tokens to verify order
        tokenizer = get_tokenizer()
        decoded = [
            tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
            for i in range(3)
        ]

        # Order should be preserved (case-insensitive)
        assert "first" in decoded[0].lower()
        assert "second" in decoded[1].lower()
        assert "third" in decoded[2].lower()

    def test_realistic_financial_texts(self):
        """Test with realistic financial sentiment texts."""
        texts = [
            "Apple stock surged 5% after strong earnings report",
            "Market crash wipes out billions in investor wealth",
            "Federal Reserve maintains interest rates, markets flat",
        ]

        inputs = tokenize_batch(texts)

        assert inputs["input_ids"].shape[0] == 3
        assert inputs["attention_mask"].sum() > 0  # All texts have tokens

    def test_mixed_length_batch_processing(self):
        """Test batch with highly varied text lengths."""
        texts = [
            "Up",  # Very short
            "The stock market experienced significant volatility today",  # Medium
            "In a surprising turn of events, " * 20,  # Very long
        ]

        inputs = tokenize_batch(texts, max_length=128, truncation=True)

        # All should be processed successfully
        assert inputs["input_ids"].shape[0] == 3
        assert inputs["input_ids"].shape[1] <= 128

        # Different attention mask sums (different real token counts)
        token_counts = inputs["attention_mask"].sum(dim=1)
        assert token_counts[0] < token_counts[1]  # "Up" < medium text
