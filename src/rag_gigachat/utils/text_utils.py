"""Утилиты для анализа и обработки текста"""

import re
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def analyze_text_quality(text: str) -> Dict[str, Any]:
    """Анализ качества текста - выявляет артефакты

    Args:
        text: Входной текст

    Returns:
        Словарь с результатами анализа
    """
    if not text:
        return {
            'size': 0,
            'char_count': 0,
            'word_count': 0,
            'line_count': 0,
            'issues': {
                'multiple_spaces': 0,
                'multiple_newlines': 0,
                'tabs': 0,
                'no_break_spaces': 0,
                'broken_words': 0,
            },
            'total_issues': 0,
            'waste_percent': 0.0,
            'samples': {}
        }

    analysis = {
        'size': len(text),
        'char_count': len(text),
        'word_count': len(text.split()),
        'line_count': len(text.split('\n')),
        'issues': {
            'multiple_spaces': len(re.findall(r' {2,}', text)),
            'multiple_newlines': len(re.findall(r'\n{3,}', text)),
            'tabs': text.count('\t'),
            'no_break_spaces': text.count('\u00A0'),
            'broken_words': len(re.findall(r'[а-яёa-z]\n[а-яёa-z]', text, re.IGNORECASE)),
        },
        'samples': {
            'first_100_chars': text[:100],
            'with_multiple_spaces': None,
            'with_broken_word': None,
        }
    }

    # Извлекаем примеры проблем
    match = re.search(r' {2,}[^ ]+', text)
    if match:
        analysis['samples']['with_multiple_spaces'] = match.group(0)

    match = re.search(r'[а-яёa-z]\n[а-яёa-z]', text, re.IGNORECASE)
    if match:
        analysis['samples']['with_broken_word'] = match.group(0)

    # Подсчет процентов потерь
    total_issues = sum(analysis['issues'].values())
    analysis['total_issues'] = total_issues
    analysis['waste_percent'] = round((total_issues / max(len(text), 1)) * 100, 2)

    return analysis


def normalize_text(text: str) -> str:
    """Нормализация текста для улучшения качества эмбеддингов.

    Действия:
    - Удаляет множественные пробелы (2+) → один пробел
    - Удаляет множественные переносы строк (3+) → два переноса (абзац)
    - Убирает табуляции и странные whitespace
    - Чистит разрывы внутри слов (буква+перенос→буква)
    - Удаляет пространство в начале/конце строк

    Args:
        text: Сырой текст из PDF

    Returns:
        Нормализованный текст

    Example:
        >>> normalize_text("Текст    с     пробелами.\\n\\n\\n")
        'Текст с пробелами.'
    """
    if not text:
        return ""

    # 1. Заменяем табуляции на пробелы
    text = text.replace('\t', ' ')

    # 2. Убираем no-break space (U+00A0)
    text = text.replace('\u00A0', ' ')

    # 3. Удаляем множественные пробелы (2+) → один пробел
    text = re.sub(r' {2,}', ' ', text)

    # 4. Нормализуем переносы строк (CRLF → LF)
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 5. Чистим разрывы внутри слов
    # Паттерн: строчная буква (кириллица или латиница) + перевод строки + строчная буква
    text = re.sub(r'([а-яёa-z])\n([а-яёa-z])', r'\1\2', text, flags=re.IGNORECASE)

    # 6. Удаляем множественные переносы строк (3+) → два переноса (абзац)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 7. Убираем пробелы в начале/конце каждой строки
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # 8. Финальный trim
    text = text.strip()

    return text
