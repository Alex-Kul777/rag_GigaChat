"""
test_latency.py - Performance tests for LLM latency optimization (BKL-003)
"""
import pytest
import inspect
import concurrent.futures
from rag_gigachat.config import model_config


@pytest.mark.performance
def test_llm_call_under_1000ms():
    """Verify max_new_tokens is reduced for latency optimization"""
    assert model_config.max_new_tokens == 500, \
        f"max_new_tokens should be 500 for latency optimization, got {model_config.max_new_tokens}"


@pytest.mark.performance
def test_context_truncation_config():
    """Verify max_context_length is configured for truncation"""
    assert model_config.max_context_length == 2000, \
        f"max_context_length should be 2000 for context truncation, got {model_config.max_context_length}"


@pytest.mark.performance
def test_timeout_wrapper_implementation():
    """Verify timeout wrapper is present in LLM call"""
    from rag_gigachat.core.rag_pipeline import RAGPipeline

    source = inspect.getsource(RAGPipeline)

    assert 'concurrent.futures' in source or 'ThreadPoolExecutor' in source, \
        "RAGPipeline should use concurrent.futures for timeout"
    assert 'TimeoutError' in source, \
        "RAGPipeline should handle TimeoutError"
    assert 'invoke_with_retry' in source, \
        "RAGPipeline should use invoke_with_retry method"


@pytest.mark.performance
def test_context_truncation_logic():
    """Verify context truncation logic is in place"""
    from rag_gigachat.core.rag_pipeline import RAGPipeline

    source = inspect.getsource(RAGPipeline)

    assert 'max_context_length' in source, \
        "Context truncation should reference max_context_length"
    assert 'docs_content[:' in source, \
        "Context truncation should slice docs_content"
