"""Утилиты для анализа и обработки текста"""

import re
from typing import Dict, Any


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
