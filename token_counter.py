"""
token_counter.py - Счетчик токенов для RAG системы
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TokenCounter:
    """Счетчик токенов для RAG системы"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Сброс счетчиков"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.num_requests = 0
        self.details = []
    
    def count_text_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте (приблизительный)"""
        if not text:
            return 0
        # Приблизительный подсчет: 1 токен ≈ 4 символа
        return len(text) // 4
    
    def add_request(self, prompt: str, response: str = None, response_metadata: Dict = None) -> int:
        """Добавление информации о запросе"""
        self.num_requests += 1
        
        prompt_tokens = self.count_text_tokens(prompt)
        completion_tokens = self.count_text_tokens(response) if response else 0
        total_tokens = prompt_tokens + completion_tokens
        
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        
        self.details.append({
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'prompt_preview': prompt[:100]
        })
        
        return total_tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        return {
            'num_requests': self.num_requests,
            'total_prompt_tokens': self.prompt_tokens,
            'total_completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_request': self.total_tokens / self.num_requests if self.num_requests > 0 else 0,
            'estimated_cost_usd': self.estimate_cost()
        }
    
    def estimate_cost(self, model: str = "gigachat") -> float:
        """Оценка стоимости"""
        if model == "gigachat":
            return self.total_tokens * 0.0001 / 1000
        else:
            return self.total_tokens * 0.002 / 1000
    
    def print_summary(self):
        """Вывод сводки"""
        stats = self.get_stats()
        print(f"""
        {'='*50}
        📊 СТАТИСТИКА ТОКЕНОВ
        {'='*50}
        📝 Запросов: {stats['num_requests']}
        🔢 Всего токенов: {stats['total_tokens']:,}
        📤 Prompt токены: {stats['total_prompt_tokens']:,}
        📥 Completion токены: {stats['total_completion_tokens']:,}
        📊 Среднее: {stats['avg_tokens_per_request']:.0f}
        💰 Стоимость: ${stats['estimated_cost_usd']:.6f}
        {'='*50}
        """)
    
    def get_stats_for_json(self) -> Dict[str, Any]:
        """Получение статистики в формате для JSON"""
        return self.get_stats()
