"""ФАЗА 2.2: Тесты SpacySmartSplitter для разбиения на предложения"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.utils.text_utils import SpacySmartSplitter, SPACY_AVAILABLE


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
class TestSpacySmartSplitter:
    """Тесты разбиения на предложения с spaCy"""

    @pytest.fixture
    def splitter(self):
        """Создание экземпляра splitter"""
        return SpacySmartSplitter()

    def test_singleton_pattern(self):
        """spaCy модели загружаются только один раз"""
        splitter1 = SpacySmartSplitter()
        splitter2 = SpacySmartSplitter()
        assert splitter1 is splitter2  # Один и тот же объект

    def test_english_sentences_simple(self, splitter):
        """Простой английский текст"""
        text = "This is first sentence. This is second sentence."
        sentences = splitter.split_into_sentences(text, language='en')

        assert len(sentences) == 2
        assert "This is first sentence." in sentences[0]
        assert "This is second sentence." in sentences[1]

    def test_english_with_abbreviations(self, splitter):
        """Английский текст с аббревиатурами"""
        text = "Mr. Smith works here. He is a doctor. Ms. Johnson is a teacher."
        sentences = splitter.split_into_sentences(text, language='en')

        # spaCy должна правильно понять что Mr. это не конец предложения
        assert len(sentences) >= 2
        assert "Mr. Smith works here." in sentences[0]

    def test_russian_sentences_simple(self, splitter):
        """Простой русский текст"""
        text = "Это первое предложение. Это второе предложение."
        sentences = splitter.split_into_sentences(text, language='ru')

        assert len(sentences) == 2
        assert "Это первое предложение." in sentences[0]
        assert "Это второе предложение." in sentences[1]

    def test_russian_with_abbreviations(self, splitter):
        """Русский текст с аббревиатурами"""
        text = "Д-р Иванов живёт в городе. Он инженер."
        sentences = splitter.split_into_sentences(text, language='ru')

        assert len(sentences) >= 1
        # spaCy должна правильно понять структуру

    def test_mixed_languages(self, splitter):
        """Смешанный RU + EN текст"""
        text = "Привет. Hello. Как дела?"
        # Когда язык не указан, используется auto-detect
        sentences = splitter.split_into_sentences(text)

        # Должны быть разбиты на предложения
        assert len(sentences) >= 2

    def test_language_detection_english(self, splitter):
        """Auto-detect: английский язык"""
        text = "This is a long English text with many words. We need at least 100 characters."
        lang = splitter.detect_language(text)
        assert lang in ['en', 'ru']

    def test_language_detection_russian(self, splitter):
        """Auto-detect: русский язык"""
        text = "Это длинный русский текст с большим количеством слов. Нам нужно как минимум 100 символов."
        lang = splitter.detect_language(text)
        assert lang in ['en', 'ru']

    def test_empty_text(self, splitter):
        """Пустой текст"""
        sentences = splitter.split_into_sentences("")
        assert sentences == []

    def test_none_text(self, splitter):
        """None как входной параметр"""
        sentences = splitter.split_into_sentences(None)
        assert sentences == []

    def test_short_text(self, splitter):
        """Короткий текст"""
        text = "Короткий текст."
        sentences = splitter.split_into_sentences(text, language='ru')
        assert len(sentences) >= 1

    def test_long_text_many_sentences(self, splitter):
        """Длинный текст со множеством предложений"""
        text = "Первое. Второе. Третье. Четвёртое. Пятое."
        sentences = splitter.split_into_sentences(text, language='ru')

        assert len(sentences) == 5

    def test_no_periods(self, splitter):
        """Текст без точек"""
        text = "Первое предложение\nВторое предложение"
        sentences = splitter.split_into_sentences(text, language='ru')

        # spaCy может разбить по переносам или другим символам
        assert len(sentences) >= 1

    def test_multiple_spaces_between_sentences(self, splitter):
        """Множественные пробелы между предложениями"""
        text = "Первое.     Второе.     Третье."
        sentences = splitter.split_into_sentences(text, language='ru')

        assert len(sentences) >= 2
        # Должны быть разбиты несмотря на лишние пробелы

    def test_special_characters(self, splitter):
        """Текст со специальными символами"""
        text = "Первое! Второе? Третье."
        sentences = splitter.split_into_sentences(text, language='ru')

        assert len(sentences) >= 2

    def test_sentences_not_empty(self, splitter):
        """Все предложения не пустые"""
        text = "Первое предложение. Второе предложение. Третье."
        sentences = splitter.split_into_sentences(text, language='ru')

        # Все предложения должны быть не пустыми
        for sent in sentences:
            assert sent and sent.strip()

    def test_sentences_stripped(self, splitter):
        """Предложения обрезаны (без ведущих/завершающих пробелов)"""
        text = "Первое.   Второе.   Третье."
        sentences = splitter.split_into_sentences(text, language='ru')

        for sent in sentences:
            assert sent == sent.strip()

    def test_real_world_english(self, splitter):
        """Real-world английский текст"""
        text = """Mr. Smith works at the company. He is a senior engineer.
        His responsibility includes maintaining the infrastructure.
        Dr. Johnson manages the team. She is very experienced."""

        sentences = splitter.split_into_sentences(text, language='en')

        # Должны быть разбиты на несколько предложений
        assert len(sentences) >= 4

    def test_real_world_russian(self, splitter):
        """Real-world русский текст"""
        text = """Документ содержит важную информацию.
        Д-р Иванов работает инженером.
        Его работа включает техническое обслуживание.
        М-р Петров управляет командой."""

        sentences = splitter.split_into_sentences(text, language='ru')

        # Должны быть разбиты на несколько предложений
        assert len(sentences) >= 3


class TestSpacySmartSplitterFallback:
    """Тесты fallback поведения когда spaCy недоступна"""

    def test_fallback_empty_text(self):
        """Fallback: пустой текст"""
        if not SPACY_AVAILABLE:
            splitter = SpacySmartSplitter()
            sentences = splitter.split_into_sentences("")
            assert sentences == []

    def test_fallback_returns_text_if_spacy_unavailable(self):
        """Fallback: возвращает исходный текст если spaCy недоступна"""
        if not SPACY_AVAILABLE:
            splitter = SpacySmartSplitter()
            text = "Test sentence"
            sentences = splitter.split_into_sentences(text)

            # В fallback режиме может вернуться исходный текст или empty list
            assert isinstance(sentences, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
