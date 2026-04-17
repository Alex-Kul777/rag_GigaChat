"""
test_llm_manager.py - Unit tests for LLM manager and token counter (BKL-003)
"""
import pytest
import time
from rag_gigachat.token_counter import TokenCounter


@pytest.mark.unit
def test_token_counter_cache_hit():
    """Verify token counter uses cache for repeated text"""
    counter = TokenCounter()

    text = "This is a test string for token counting"

    count1 = counter.count_text_tokens(text)
    assert count1 > 0

    initial_cache_size = len(counter._cache)
    count2 = counter.count_text_tokens(text)
    assert count1 == count2

    assert len(counter._cache) == initial_cache_size, \
        "Cache should not grow for repeated text (indicates cache hit)"


@pytest.mark.unit
def test_token_counter_cache_different_texts():
    """Verify cache correctly distinguishes different texts"""
    counter = TokenCounter()

    text1 = "Short text"
    text2 = "This is a much longer text that should have more tokens than the short text"

    count1 = counter.count_text_tokens(text1)
    count2 = counter.count_text_tokens(text2)

    assert count1 < count2, "Longer text should have more tokens"
    assert len(counter._cache) >= 2, "Cache should have entries for both texts"


@pytest.mark.unit
def test_token_counter_cache_performance():
    """Verify cache hit is faster than re-encoding"""
    counter = TokenCounter()

    text = "X" * 1000

    start = time.perf_counter()
    count1 = counter.count_text_tokens(text)
    time_first = time.perf_counter() - start

    start = time.perf_counter()
    count2 = counter.count_text_tokens(text)
    time_cached = time.perf_counter() - start

    assert count1 == count2
    if time_first > 0.001:
        assert time_cached < time_first, "Cache hit should be faster than re-encoding"


@pytest.mark.unit
def test_token_counter_empty_text():
    """Verify token counter handles empty text"""
    counter = TokenCounter()

    count = counter.count_text_tokens("")
    assert count == 0


@pytest.mark.unit
def test_token_counter_cache_reuse():
    """Verify cache is reused across calls"""
    counter = TokenCounter()

    text = "Test for cache reuse"
    initial_cache_size = len(counter._cache)

    counter.count_text_tokens(text)
    cache_after_first = len(counter._cache)

    counter.count_text_tokens(text)
    cache_after_second = len(counter._cache)

    assert cache_after_first > initial_cache_size, "Cache should grow after first call"
    assert cache_after_first == cache_after_second, "Cache should not grow for repeated text"


@pytest.mark.unit
def test_max_new_tokens_reduced():
    """Verify max_new_tokens is reduced to 500 for latency optimization"""
    from rag_gigachat.config import model_config

    assert model_config.max_new_tokens == 500, \
        f"BKL-003: max_new_tokens must be 500 for latency, got {model_config.max_new_tokens}"
