"""
tests/test_token_counter.py - Тесты для счетчика токенов
"""

import pytest
from rag_gigachat.token_counter import TokenCounter


class TestTokenCounter:
    """Тесты для TokenCounter"""
    
    def test_initialization(self):
        """Тест инициализации счетчика"""
        counter = TokenCounter()
        assert counter.total_tokens == 0
        assert counter.num_requests == 0
        assert counter.prompt_tokens == 0
        assert counter.completion_tokens == 0
    
    def test_add_request(self):
        """Тест добавления запроса"""
        counter = TokenCounter()
        tokens = counter.add_request("Привет мир", "Ответ мир")
        assert tokens > 0
        assert counter.num_requests == 1
        assert counter.total_tokens > 0
    
    def test_count_text_tokens(self):
        """Тест подсчета токенов в тексте"""
        counter = TokenCounter()
        text = "Это тестовый текст для подсчета токенов"
        tokens = counter.count_text_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_get_stats(self):
        """Тест получения статистики"""
        counter = TokenCounter()
        counter.add_request("q1", "a1")
        counter.add_request("q2", "a2")
        
        stats = counter.get_stats()
        assert 'num_requests' in stats
        assert 'total_tokens' in stats
        assert stats['num_requests'] == 2
    
    def test_reset(self):
        """Тест сброса счетчика"""
        counter = TokenCounter()
        counter.add_request("test", "response")
        assert counter.num_requests == 1
        
        counter.reset()
        assert counter.num_requests == 0
        assert counter.total_tokens == 0
    
    def test_estimate_cost(self):
        """Тест оценки стоимости"""
        counter = TokenCounter()
        counter.add_request("test", "response" * 100)
        cost = counter.estimate_cost()
        assert cost >= 0
        assert isinstance(cost, float)
