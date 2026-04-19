"""ФАЗА 1: Тесты диагностики качества текста"""

import pytest
from pathlib import Path
import sys

# Добавляем src в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.utils.text_utils import analyze_text_quality


class TestTextQualityAnalysis:
    """Тесты анализа качества текста"""

    def test_empty_text(self):
        """Пустой текст"""
        result = analyze_text_quality("")
        assert result['size'] == 0
        assert result['char_count'] == 0

    def test_clean_text(self):
        """Чистый текст без проблем"""
        text = "Это чистый текст. Без проблем."
        result = analyze_text_quality(text)

        assert result['char_count'] == len(text)
        assert result['issues']['multiple_spaces'] == 0
        assert result['issues']['tabs'] == 0
        assert result['issues']['broken_words'] == 0
        assert result['total_issues'] == 0
        assert result['waste_percent'] == 0.0

    def test_multiple_spaces(self):
        """Множественные пробелы"""
        text = "Текст    с     лишними    пробелами"
        result = analyze_text_quality(text)

        assert result['issues']['multiple_spaces'] > 0
        assert result['total_issues'] > 0
        assert result['waste_percent'] > 0

    def test_multiple_newlines(self):
        """Множественные переносы строк"""
        text = "Абзац 1.\n\n\n\nАбзац 2."
        result = analyze_text_quality(text)

        assert result['issues']['multiple_newlines'] > 0

    def test_tabs(self):
        """Табуляции"""
        text = "Текст\t\tс\tтабуляциями"
        result = analyze_text_quality(text)

        assert result['issues']['tabs'] == 3

    def test_no_break_spaces(self):
        """No-break space символы"""
        text = "Текст\u00A0с\u00A0no-break\u00A0space"
        result = analyze_text_quality(text)

        assert result['issues']['no_break_spaces'] == 3

    def test_broken_words(self):
        """Разрывы внутри слов"""
        text = "При\nнять решение"
        result = analyze_text_quality(text)

        assert result['issues']['broken_words'] == 1

    def test_broken_words_english(self):
        """Разрывы в английских словах"""
        text = "Trans\nport network"
        result = analyze_text_quality(text)

        assert result['issues']['broken_words'] == 1

    def test_complex_text(self):
        """Комплексный текст с множеством проблем"""
        text = """Первое  предложение.   Второе
предложение  здесь.  Третье\tпредложение.

Новый абзац\u00A0с проблемами."""

        result = analyze_text_quality(text)

        # Проверяем что все проблемы обнаружены
        assert result['issues']['multiple_spaces'] > 0
        assert result['issues']['tabs'] > 0
        assert result['issues']['no_break_spaces'] > 0
        assert result['issues']['broken_words'] > 0
        assert result['total_issues'] > 0
        assert result['waste_percent'] > 0

    def test_word_count(self):
        """Подсчет слов"""
        text = "Первое второе третье четвёртое"
        result = analyze_text_quality(text)

        assert result['word_count'] == 4

    def test_line_count(self):
        """Подсчет строк"""
        text = "Строка 1\nСтрока 2\nСтрока 3"
        result = analyze_text_quality(text)

        assert result['line_count'] == 3

    def test_waste_percent_calculation(self):
        """Расчет процента потерь"""
        # Текст с известными проблемами
        text = "A    B"  # Множественные пробелы
        result = analyze_text_quality(text)

        # Должны быть обнаружены множественные пробелы
        assert result['waste_percent'] > 0
        assert result['total_issues'] >= 1  # Минимум одно найденное множество пробелов

    def test_real_world_pdf_text(self):
        """Реальный текст из PDF"""
        # Симуляция текста который может быть из PDF
        text = """Документ   с   множеством   проблем.


Новый  абзац  идёт  здесь.  Слово  при\nнято
в  документ.  Автор:  д-р  Иванов\t\t(контакт)."""

        result = analyze_text_quality(text)

        # Должны быть проблемы
        assert result['issues']['multiple_spaces'] > 0
        assert result['issues']['multiple_newlines'] > 0  # Есть два пустых строк подряд (3+ переноса)
        assert result['issues']['tabs'] > 0
        assert result['total_issues'] > 0


class TestDiagnosticsMetrics:
    """Тесты метрик диагностики"""

    def test_no_issues_minimal_waste(self):
        """Чистый текст = минимальные потери"""
        text = "Чистый текст без проблем."
        result = analyze_text_quality(text)

        assert result['waste_percent'] == 0.0

    def test_high_issues_high_waste(self):
        """Много проблем = большие потери"""
        text = "Текст    с    очень    много    проблемами\t\t\tи табуляциями\n\n\n\nи переносами"
        result = analyze_text_quality(text)

        assert result['waste_percent'] > 10.0  # Более 10% потерь

    def test_samples_extraction(self):
        """Извлечение примеров проблем"""
        text = "Начало  с проблемой и При\nнятым словом"
        result = analyze_text_quality(text)

        # Проверяем что примеры содержат проблемы
        assert result['samples']['first_100_chars'] == text[:100]
        if result['samples']['with_multiple_spaces']:
            assert '  ' in result['samples']['with_multiple_spaces']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
