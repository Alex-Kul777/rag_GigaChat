"""
token_counter.py - Счетчик токенов для RAG системы
"""

import json
import logging
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _tiktoken_available = True
except ImportError:
    _tiktoken_available = False


class TokenCounter:
    """Счетчик токенов для RAG системы"""

    def __init__(self):
        """Инициализация счетчика токенов"""
        self.reset()
        self.balance_history = []
        self.last_balance = None
        self.last_balance_time = None
        self._cache = {}

        try:
            self.encoder = tiktoken.get_encoding("cl100k_base") if _tiktoken_available else None
        except Exception:
            self.encoder = None

    def reset(self):
        """Сброс счетчиков"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.num_requests = 0
        self.details = []

    def count_text_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте (с кешированием)"""
        if not text:
            return 0
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        if self.encoder:
            count = len(self.encoder.encode(text))
        else:
            count = len(text) // 4
        self._cache[key] = count
        return count

    def add_request(self, prompt: str, response: str = None, response_metadata: Dict = None) -> int:
        """
        Добавление информации о запросе

        Args:
            prompt: Текст запроса
            response: Текст ответа (опционально)
            response_metadata: Метаданные ответа от API

        Returns:
            Количество использованных токенов
        """
        self.num_requests += 1

        if response_metadata and 'token_usage' in response_metadata:
            token_usage = response_metadata['token_usage']
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            total_tokens = token_usage.get('total_tokens', 0)
        else:
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

    def calculate_balance_delta(
            self, initial_balance: Optional[Dict], final_balance: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Расчет расхода по дельте баланса

        Args:
            initial_balance: Начальный баланс (может быть None)
            final_balance: Конечный баланс (может быть None)

        Returns:
            Словарь с дельтой
        """
        delta: Dict[str, Any] = {
            'error': None,
            'has_data': False,
            'timestamp': datetime.now().isoformat()
        }

        if initial_balance is None:
            delta['error'] = 'initial_balance is None'
            logger.warning("initial_balance is None, cannot calculate delta")
            return delta

        if final_balance is None:
            delta['error'] = 'final_balance is None'
            logger.warning("final_balance is None, cannot calculate delta")
            return delta

        if not isinstance(initial_balance, dict):
            delta['error'] = f'initial_balance is not a dict: {type(initial_balance)}'
            logger.warning(delta['error'])
            return delta

        if not isinstance(final_balance, dict):
            delta['error'] = f'final_balance is not a dict: {type(final_balance)}'
            logger.warning(delta['error'])
            return delta

        balance_fields = ['balance', 'available', 'total', 'amount', 'value', 'credits']
        for field in balance_fields:
            if field in initial_balance and field in final_balance:
                try:
                    initial_value = float(initial_balance[field])
                    final_value = float(final_balance[field])
                    delta[field] = initial_value - final_value
                    delta['has_data'] = True
                except (ValueError, TypeError) as e:
                    logger.debug(f"Не удалось преобразовать поле {field}: {e}")

        token_fields = ['tokens_used', 'tokens', 'total_tokens', 'usage']
        for field in token_fields:
            if field in final_balance:
                try:
                    delta[field] = final_balance.get(field, 0)
                    delta['has_data'] = True
                except (ValueError, TypeError):
                    pass

        if not delta.get('has_data'):
            delta['error'] = 'No valid balance fields found'
            delta['available_fields_initial'] = list(initial_balance.keys())
            delta['available_fields_final'] = list(final_balance.keys())
            logger.warning(
                f"No valid balance fields found. "
                f"Initial fields: {list(initial_balance.keys())}, "
                f"Final fields: {list(final_balance.keys())}"
            )

        return delta

    def add_request_with_balance(self,
                                 prompt: str,
                                 response: str = None,
                                 response_metadata: Dict = None,
                                 client=None) -> int:
        """
        Добавление запроса с учетом баланса

        Args:
            prompt: Текст запроса
            response: Текст ответа
            response_metadata: Метаданные ответа от API
            client: GigaChat клиент (для получения баланса)

        Returns:
            Количество использованных токенов
        """
        balance_before = self.get_balance_info(client) if client else None
        tokens_used = self.add_request(prompt, response, response_metadata)
        balance_after = self.get_balance_info(client) if client else None

        if balance_before and balance_after:
            delta = self.calculate_balance_delta(balance_before, balance_after)
            logger.info(f"Расход по балансу: {delta}")
            self.details[-1]['balance_delta'] = delta

        return tokens_used

    def get_balance_info(self, client) -> Optional[Dict[str, Any]]:
        """
        Получение информации о балансе из GigaChat клиента

        Args:
            client: GigaChat клиент

        Returns:
            Словарь с информацией о балансе или None
        """
        try:
            balance_obj = None
            if hasattr(client, 'get_balance'):
                balance_obj = client.get_balance()
            elif hasattr(client, 'balance'):
                balance_obj = client.balance
            elif hasattr(client, 'get_account_balance'):
                balance_obj = client.get_account_balance()

            if balance_obj is None:
                logger.warning("Не удалось получить баланс: ни один метод не сработал")
                return None

            if hasattr(balance_obj, 'model_dump'):
                balance_dict = balance_obj.model_dump()
            elif hasattr(balance_obj, 'dict'):
                balance_dict = balance_obj.dict()
            elif hasattr(balance_obj, '__dict__'):
                balance_dict = vars(balance_obj)
            else:
                balance_dict = {'balance': str(balance_obj)}

            balance_dict['timestamp'] = datetime.now().isoformat()
            self.balance_history.append(balance_dict)
            self.last_balance = balance_dict
            self.last_balance_time = datetime.now()

            logger.info(f"Баланс получен: {balance_dict}")
            return balance_dict

        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return None

    def get_balance_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по балансу

        Returns:
            Словарь со статистикой баланса
        """
        if not self.balance_history:
            return {'has_balance_data': False}

        first_balance = self.balance_history[0]
        last_balance = self.balance_history[-1]
        total_delta = self.calculate_balance_delta(first_balance, last_balance)

        return {
            'has_balance_data': True,
            'first_balance': first_balance,
            'last_balance': last_balance,
            'total_delta': total_delta,
            'num_balance_checks': len(self.balance_history),
            'balance_history': self.balance_history
        }

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        return {
            'num_requests': self.num_requests,
            'total_prompt_tokens': self.prompt_tokens,
            'total_completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_request': (
                self.total_tokens / self.num_requests if self.num_requests > 0 else 0
            ),
            'estimated_cost_usd': self.estimate_cost()
        }

    def get_stats_for_json(self) -> Dict[str, Any]:
        """Получение статистики в формате для JSON (включая баланс)"""
        stats = self.get_stats()

        balance_stats = self.get_balance_statistics()
        if balance_stats.get('has_balance_data'):
            stats['balance'] = {
                'first_balance': balance_stats.get('first_balance'),
                'last_balance': balance_stats.get('last_balance'),
                'total_delta': balance_stats.get('total_delta'),
                'num_checks': balance_stats.get('num_balance_checks')
            }

        return stats

    def estimate_cost(self, model: str = "gigachat") -> float:
        """Оценка стоимости (примерные цены)"""
        if model == "gigachat":
            return self.total_tokens * 0.0001 / 1000
        return self.total_tokens * 0.002 / 1000

    def print_summary(self):
        """Вывод сводки"""
        stats = self.get_stats()
        print(f"""
        {'='*50}
        📊 СТАТИСТИКА ТОКЕНОВ ЗА ЭКСПЕРИМЕНТ
        {'='*50}
        📝 Количество запросов: {stats['num_requests']}
        🔢 Всего токенов: {stats['total_tokens']:,}
        📤 Prompt токены: {stats['total_prompt_tokens']:,}
        📥 Completion токены: {stats['total_completion_tokens']:,}
        📊 Среднее токенов на запрос: {stats['avg_tokens_per_request']:.0f}
        💰 Оценочная стоимость: ${stats['estimated_cost_usd']:.4f}
        {'='*50}
        """)

    def save_to_file(self, filepath: str):
        """Сохранение статистики в файл"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': self.get_stats(),
                'details': self.details
            }, f, indent=2, ensure_ascii=False)
