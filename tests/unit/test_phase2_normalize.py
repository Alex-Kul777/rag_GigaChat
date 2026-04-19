"""ФАЗА 2.1: Тесты нормализации текста"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.utils.text_utils import normalize_text, analyze_text_quality


class TestNormalizeText:
    """Тесты функции normalize_text()"""

    def test_empty_text(self):
        """Пустой текст"""
        assert normalize_text("") == ""
        assert normalize_text(None) == "" if None is None else True

    def test_none_input(self):
        """None как входной параметр"""
        try:
            result = normalize_text(None)
            # Должна вернуть пустую строку
            assert result == ""
        except AttributeError:
            # Или выбросить ошибку - это тоже нормально
            pass

    def test_clean_text(self):
        """Чистый текст - не должен измениться"""
        text = "Это чистый текст."
        assert normalize_text(text) == text

    def test_multiple_spaces_single(self):
        """Два пробела → один"""
        assert normalize_text("Текст  с  пробелами") == "Текст с пробелами"

    def test_multiple_spaces_many(self):
        """Много пробелов → один"""
        assert normalize_text("Текст          с          пробелами") == "Текст с пробелами"

    def test_tabs_to_spaces(self):
        """Табуляции заменяются на пробелы, потом объединяются"""
        # После замены табуляции на пробелы и объединения получим один пробел
        assert normalize_text("Текст\t\tс\tтабуляциями") == "Текст с табуляциями"

    def test_no_break_space(self):
        """No-break space (U+00A0) → обычный пробел"""
        text = "Текст\u00A0с\u00A0no-break"
        # После замены на пробелы: "Текст с no-break"
        assert normalize_text(text) == "Текст с no-break"

    def test_crlf_to_lf(self):
        """CRLF нормализуется в LF"""
        text = "Строка 1\r\nСтрока 2"
        result = normalize_text(text)
        assert "\r" not in result
        assert "\n" in result

    def test_broken_word_russian(self):
        """Разрыв внутри русского слова"""
        text = "При\nнять решение"
        assert normalize_text(text) == "Принять решение"

    def test_broken_word_english(self):
        """Разрыв внутри английского слова"""
        text = "Trans\nport network"
        assert normalize_text(text) == "Transport network"

    def test_multiple_newlines(self):
        """Множественные переносы строк (3+) → два"""
        text = "Абзац 1.\n\n\n\nАбзац 2."
        assert normalize_text(text) == "Абзац 1.\n\nАбзац 2."

    def test_multiple_newlines_many(self):
        """Много переносов → два"""
        text = "Строка 1.\n\n\n\n\n\nСтрока 2."
        result = normalize_text(text)
        assert result == "Строка 1.\n\nСтрока 2."
        assert "\n\n\n" not in result

    def test_leading_trailing_spaces_in_lines(self):
        """Пробелы в начале/конце строк удаляются"""
        text = "  Строка 1  \n  Строка 2  "
        result = normalize_text(text)
        assert result == "Строка 1\nСтрока 2"

    def test_combined_issues(self):
        """Комбинированный случай с множеством проблем"""
        text = """Первое  предложение.   Второе
предложение  здесь.  Третье\tпредложение.

Новый абзац\u00A0с проблемами."""

        result = normalize_text(text)

        # Проверяем что все проблемы исправлены
        assert "  " not in result  # Нет двойных пробелов
        assert "\t" not in result  # Нет табуляций
        assert "\u00A0" not in result  # Нет no-break spaces
        assert "Третье предложение" in result  # Слово объединено
        assert "\n\n\n" not in result  # Нет множественных переносов

    def test_preserves_punctuation(self):
        """Пунктуация сохраняется"""
        text = "Текст  с  пунктуацией!  Вопрос?  Восклицание!"
        result = normalize_text(text)
        assert "!" in result
        assert "?" in result

    def test_mixed_languages(self):
        """Смешанный RU+EN текст"""
        text = "Привет  world.  Привет\nworld."
        result = normalize_text(text)
        assert "Привет world" in result

    def test_real_world_pdf_example(self):
        """Реальный пример из PDF"""
        text = """Документ   с   множеством   проблем.


Новый  абзац  идёт  здесь.  Слово  при\nнято
в  документ.  Автор:  д-р  Иванов\t\t(контакт)."""

        result = normalize_text(text)

        # Ключевые проверки
        assert "Документ с множеством проблем" in result
        assert "Новый абзац идёт здесь" in result
        assert "Слово принято" in result  # Слово объединено
        assert "Иванов (контакт)" in result  # Табуляции убраны
        assert "\n\n\n" not in result  # Множественные переносы убраны
        assert "    " not in result  # Множественные пробелы убраны

    def test_maintains_paragraph_structure(self):
        """Структура абзацев сохраняется"""
        text = "Абзац 1.\n\nАбзац 2.\n\nАбзац 3."
        result = normalize_text(text)
        assert result == text  # Чистая структура не меняется

    def test_single_newline_preserved(self):
        """Одиночные переносы строк сохраняются"""
        text = "Строка 1\nСтрока 2\nСтрока 3"
        result = normalize_text(text)
        assert result == text


class TestNormalizationEffectiveness:
    """Тесты эффективности нормализации"""

    def test_size_reduction(self):
        """Размер текста уменьшается при нормализации"""
        text = "Текст    с    лишними    пробелами\t\t\tи табуляциями\n\n\n"
        normalized = normalize_text(text)
        assert len(normalized) < len(text)

    def test_waste_reduction(self):
        """Процент потерь данных уменьшается"""
        text = "Текст    с    проблемами\t\t\tи переносами\n\n\n"

        before = analyze_text_quality(text)
        after = analyze_text_quality(normalize_text(text))

        # После нормализации должно быть меньше проблем
        assert after['total_issues'] < before['total_issues']
        assert after['waste_percent'] < before['waste_percent']

    def test_quality_score_improves(self):
        """Качество текста улучшается"""
        text = """Первое  предложение.   Второе
предложение  здесь.  Третье\tпредложение."""

        before = analyze_text_quality(text)
        after = analyze_text_quality(normalize_text(text))

        # Проверяем что все категории проблем уменьшились
        for issue_type in after['issues']:
            assert after['issues'][issue_type] <= before['issues'][issue_type]


class TestNormalizeTextEdgeCases:
    """Граничные случаи для normalize_text()"""

    def test_only_spaces(self):
        """Текст только из пробелов"""
        assert normalize_text("    ") == ""

    def test_only_newlines(self):
        """Текст только из переносов"""
        assert normalize_text("\n\n\n") == ""

    def test_only_tabs(self):
        """Текст только из табуляций"""
        assert normalize_text("\t\t\t") == ""

    def test_mixed_whitespace(self):
        """Смешанные whitespace символы"""
        text = "  \t  \n  \t  "
        result = normalize_text(text)
        assert result == ""

    def test_single_character(self):
        """Один символ"""
        assert normalize_text("A") == "A"

    def test_very_long_text(self):
        """Очень длинный текст"""
        text = ("Текст    с     пробелами. " * 1000)
        result = normalize_text(text)
        assert "  " not in result  # Нет двойных пробелов

    def test_special_characters(self):
        """Специальные символы"""
        text = "Текст  с  спец. символами: @#$%^&*()"
        result = normalize_text(text)
        assert "@" in result
        assert "#" in result

    def test_numbers_in_text(self):
        """Числа в тексте"""
        text = "Число  3.14  это  пи.  Число  при\nложе это словосочетание."
        result = normalize_text(text)
        assert "3.14" in result
        assert "приложе" in result  # Разрыв объединен (внутри слова)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
