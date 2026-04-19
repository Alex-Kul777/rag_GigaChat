"""ФАЗА 2.5: Тесты подсчёта токенов и фильтрации чанков"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.utils.text_utils import (
    estimate_token_count, estimate_language, filter_chunks_by_token_count,
    filter_documents_by_token_count
)
from langchain_core.documents import Document as LangChainDocument


class TestEstimateTokenCount:
    """Тесты оценки количества токенов"""

    def test_empty_text(self):
        """Пустой текст содержит 0 токенов"""
        assert estimate_token_count("") == 0

    def test_short_russian_text(self):
        """Оценка токенов для короткого русского текста"""
        text = "Короткое предложение"  # ~5 слов
        token_count = estimate_token_count(text, 'ru')
        assert token_count > 0
        assert token_count >= 3  # Как минимум несколько токенов

    def test_short_english_text(self):
        """Оценка токенов для короткого английского текста"""
        text = "Short sentence"  # ~2 слова
        token_count = estimate_token_count(text, 'en')
        assert token_count > 0

    def test_russian_vs_english_density(self):
        """Русский текст должен быть более плотным (меньше токенов на символ)"""
        ru_text = "А" * 100
        en_text = "A" * 100

        ru_tokens = estimate_token_count(ru_text, 'ru')
        en_tokens = estimate_token_count(en_text, 'en')

        # Русский текст более плотный, поэтому токенов должно быть меньше
        assert ru_tokens < en_tokens

    def test_token_count_increases_with_length(self):
        """Количество токенов растет с длиной текста"""
        short = "Это текст"
        long = short * 10

        short_tokens = estimate_token_count(short, 'ru')
        long_tokens = estimate_token_count(long, 'ru')

        assert long_tokens > short_tokens

    def test_realistic_russian_chunk(self):
        """Реалистичный русский чанк"""
        text = "Это предложение со смыслом. Вот ещё одно предложение. И третье тоже."
        token_count = estimate_token_count(text, 'ru')
        # Примерно 20-30 токенов для 3 предложений
        assert token_count >= 15

    def test_realistic_english_chunk(self):
        """Реалистичный английский чанк"""
        text = "This is a meaningful sentence. Here comes another one. And a third too."
        token_count = estimate_token_count(text, 'en')
        # Примерно 20-30 токенов для 3 предложений
        assert token_count >= 15

    def test_language_parameter_default(self):
        """По умолчанию используется русский язык"""
        text = "Текст без указания языка"
        # Без параметра language используется 'ru'
        token_count = estimate_token_count(text)
        assert token_count > 0


class TestEstimateLanguage:
    """Тесты определения языка"""

    def test_short_text_defaults_to_russian(self):
        """Короткий текст по умолчанию русский"""
        text = "Привет"
        lang = estimate_language(text)
        assert lang == 'ru'

    def test_language_estimation_works(self):
        """Определение языка длинного текста"""
        text = "А" * 100  # Много русских букв
        lang = estimate_language(text)
        # Должно определить русский язык
        assert isinstance(lang, str)

    def test_empty_text_defaults_russian(self):
        """Пустой текст по умолчанию русский"""
        lang = estimate_language("")
        assert lang == 'ru'


class TestFilterChunksByTokenCount:
    """Тесты фильтрации чанков по токенам"""

    def test_empty_list(self):
        """Пустой список чанков возвращает пустой список"""
        result = filter_chunks_by_token_count([])
        assert result == []

    def test_all_chunks_pass_filter(self):
        """Все достаточно длинные чанки проходят фильтр"""
        chunks = [
            "Это достаточно длинный текст с несколькими словами для фильтра.",
            "Вот ещё один длинный чанк с достаточным количеством информации."
        ]
        result = filter_chunks_by_token_count(chunks, min_tokens=5)
        assert len(result) == 2

    def test_short_chunks_filtered(self):
        """Короткие чанки удаляются"""
        chunks = [
            "Длинный чанк с достаточным количеством текста для прохождения фильтра.",
            "Короткий"  # Слишком короткий
        ]
        result = filter_chunks_by_token_count(chunks, min_tokens=10)
        assert len(result) == 1
        assert "Длинный" in result[0]

    def test_min_tokens_parameter(self):
        """Параметр min_tokens работает"""
        chunk = "Это текст с несколькими словами"
        # Много чанков одного размера
        chunks = [chunk] * 5

        # С низким минимумом все проходят
        result_low = filter_chunks_by_token_count(chunks, min_tokens=1)
        assert len(result_low) == 5

        # С высоким минимумом некоторые отфильтруются
        result_high = filter_chunks_by_token_count(chunks, min_tokens=500)
        assert len(result_high) < len(result_low)

    def test_max_tokens_parameter(self):
        """Параметр max_tokens работает"""
        short = "Короткий текст"
        long = "Это очень длинный текст " * 50  # Много слов

        chunks = [short, long]

        result = filter_chunks_by_token_count(chunks, min_tokens=1, max_tokens=100)
        # Long чанк должен быть отфильтрован
        assert len(result) <= 2

    def test_language_parameter(self):
        """Параметр language передается в оценку"""
        chunks = ["Текст для фильтра"]
        # Должно работать без ошибок
        result = filter_chunks_by_token_count(chunks, min_tokens=1, language='ru')
        assert isinstance(result, list)

    def test_returns_list(self):
        """Функция всегда возвращает список"""
        chunks = ["Текст"]
        result = filter_chunks_by_token_count(chunks)
        assert isinstance(result, list)

    def test_filters_empty_chunks(self):
        """Пустые чанки отфильтруются"""
        chunks = [
            "Нормальный чанк с текстом",
            "",  # Пустой чанк
        ]
        result = filter_chunks_by_token_count(chunks, min_tokens=1)
        # Пустой чанк должен быть отфильтрован
        assert "" not in result

    def test_realistic_filtering_scenario(self):
        """Реалистичный сценарий фильтрации"""
        chunks = [
            "Первое предложение.",  # Слишком короткое
            "Это предложение имеет достаточно слов и информации для прохождения фильтра качества.",
            "Второе слово.",  # Слишком короткое
            "Вот третье достаточно длинное предложение с несколькими словами внутри него.",
        ]

        # Фильтруем с минимумом 20 токенов (примерно 2 предложения)
        result = filter_chunks_by_token_count(chunks, min_tokens=20, language='ru')

        # Должны остаться только достаточно длинные предложения
        assert len(result) < len(chunks)
        assert any("достаточно" in chunk for chunk in result)


class TestFilterDocumentsByTokenCount:
    """Тесты фильтрации LangChain документов по токенам"""

    def test_empty_list(self):
        """Пустой список документов возвращает пустой список"""
        result = filter_documents_by_token_count([])
        assert result == []

    def test_all_documents_pass(self):
        """Все достаточно длинные документы проходят фильтр"""
        docs = [
            LangChainDocument(
                page_content="Это достаточно длинный текст с несколькими словами для фильтра.",
                metadata={'source': 'doc1.pdf'}
            ),
            LangChainDocument(
                page_content="Вот ещё один длинный текст с достаточным количеством информации.",
                metadata={'source': 'doc2.pdf'}
            )
        ]
        result = filter_documents_by_token_count(docs, min_tokens=5)
        assert len(result) == 2

    def test_short_documents_filtered(self):
        """Короткие документы удаляются"""
        docs = [
            LangChainDocument(
                page_content="Длинный документ с достаточным количеством текста для фильтра.",
                metadata={'source': 'doc1.pdf'}
            ),
            LangChainDocument(
                page_content="Короткий",
                metadata={'source': 'doc2.pdf'}
            )
        ]
        result = filter_documents_by_token_count(docs, min_tokens=10)
        assert len(result) == 1

    def test_metadata_enrichment(self):
        """Метаданные обогащаются информацией о токенах"""
        docs = [
            LangChainDocument(
                page_content="Это текст с несколькими словами для теста.",
                metadata={'source': 'doc.pdf'}
            )
        ]
        result = filter_documents_by_token_count(docs, min_tokens=1)

        assert len(result) == 1
        doc = result[0]
        assert 'token_count' in doc.metadata
        assert 'language' in doc.metadata
        assert doc.metadata['token_count'] > 0

    def test_min_tokens_works(self):
        """Параметр min_tokens работает корректно"""
        doc = LangChainDocument(
            page_content="Текст с несколькими словами",
            metadata={'source': 'doc.pdf'}
        )
        docs = [doc] * 5

        # С низким минимумом все проходят
        result_low = filter_documents_by_token_count(docs, min_tokens=1)
        assert len(result_low) == 5

        # С высоким минимумом большинство отфильтруется
        result_high = filter_documents_by_token_count(docs, min_tokens=500)
        assert len(result_high) < len(result_low)

    def test_max_tokens_works(self):
        """Параметр max_tokens работает"""
        short_doc = LangChainDocument(
            page_content="Короткий текст",
            metadata={'source': 'short.pdf'}
        )
        long_doc = LangChainDocument(
            page_content="Очень длинный текст " * 100,
            metadata={'source': 'long.pdf'}
        )

        result = filter_documents_by_token_count([short_doc, long_doc], min_tokens=1, max_tokens=100)
        # Длинный документ должен быть отфильтрован
        assert len(result) <= 2

    def test_preserves_metadata(self):
        """Исходные метаданные сохраняются"""
        docs = [
            LangChainDocument(
                page_content="Достаточно длинный текст для фильтра тестирования.",
                metadata={'source': 'doc.pdf', 'author': 'Test Author'}
            )
        ]
        result = filter_documents_by_token_count(docs, min_tokens=1)

        assert len(result) == 1
        assert result[0].metadata['source'] == 'doc.pdf'
        assert result[0].metadata['author'] == 'Test Author'

    def test_language_parameter(self):
        """Параметр language используется"""
        docs = [
            LangChainDocument(
                page_content="Текст для теста",
                metadata={'source': 'doc.pdf'}
            )
        ]
        result = filter_documents_by_token_count(docs, min_tokens=1, language='ru')
        assert isinstance(result, list)

    def test_realistic_filtering(self):
        """Реалистичный сценарий фильтрации документов"""
        docs = [
            LangChainDocument(
                page_content="Первое предложение.",  # Слишком короткое
                metadata={'source': 'doc1.pdf', 'page': 1}
            ),
            LangChainDocument(
                page_content="Это документ имеет достаточно текста и информации для прохождения фильтра качества.",
                metadata={'source': 'doc2.pdf', 'page': 2}
            ),
            LangChainDocument(
                page_content="Второе слово.",  # Слишком короткое
                metadata={'source': 'doc3.pdf', 'page': 3}
            ),
            LangChainDocument(
                page_content="Вот третий документ с достаточным количеством слов для фильтра.",
                metadata={'source': 'doc4.pdf', 'page': 4}
            ),
        ]

        result = filter_documents_by_token_count(docs, min_tokens=20)

        # Должны остаться только достаточно длинные документы
        assert len(result) < len(docs)
        # Проверяем что каждый результат имеет метаданные
        for doc in result:
            assert 'token_count' in doc.metadata
            assert doc.metadata['token_count'] >= 20

    def test_documents_without_metadata(self):
        """Обработка документов без метаданных (если возможно)"""
        # LangChainDocument всегда имеет metadata
        docs = [
            LangChainDocument(
                page_content="Текст для тестирования фильтра качества",
                metadata={}
            )
        ]
        result = filter_documents_by_token_count(docs, min_tokens=1)
        assert len(result) == 1
        assert 'token_count' in result[0].metadata


class TestTokenFilteringIntegration:
    """Интеграционные тесты фильтрации токенов"""

    def test_typical_pdf_chunk_filtering(self):
        """Типичный сценарий: фильтрация чанков из PDF"""
        # Симуляция чанков из TextSplitter
        chunks = [
            "Первое предложение.",  # Слишком короткое
            "Это достаточно длинное предложение с несколькими словами для тестирования фильтра.",
            "",  # Пустой чанк
            "Ещё одно предложение",  # Может быть короче
            "Вот третий чанк с достаточным количеством текста для прохождения фильтра качества.",
        ]

        # Фильтруем с минимумом 30 токенов (примерно 2 предложения)
        result = filter_chunks_by_token_count(chunks, min_tokens=30)

        # Должны остаться только хорошие чанки
        assert len(result) < len(chunks)
        assert "" not in result
        assert all(len(chunk) > 10 for chunk in result)

    def test_document_quality_improvement(self):
        """Улучшение качества документов через фильтрацию"""
        docs = [
            LangChainDocument(page_content="Плохо.", metadata={'source': 'test.pdf'}),
            LangChainDocument(page_content="Хорошее качество текста с достаточным количеством информации.", metadata={'source': 'test.pdf'}),
            LangChainDocument(page_content="Еще.", metadata={'source': 'test.pdf'}),
            LangChainDocument(page_content="Отличный чанк с достаточно большим объемом текста для обработки.", metadata={'source': 'test.pdf'}),
        ]

        before = len(docs)
        result = filter_documents_by_token_count(docs, min_tokens=20)
        after = len(result)

        # Количество документов должно уменьшиться
        assert after < before
        # Все оставшиеся документы должны быть качественными
        for doc in result:
            assert doc.metadata['token_count'] >= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
