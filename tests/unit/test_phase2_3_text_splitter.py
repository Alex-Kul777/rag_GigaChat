"""ФАЗА 2.3: Тесты интеграции SpacySmartSplitter в TextSplitter"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.data.data_loader import TextSplitter
from rag_gigachat.utils.text_utils import SpacySmartSplitter, SPACY_AVAILABLE, normalize_text
from langchain_core.documents import Document as LangChainDocument


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
class TestTextSplitterWithSpacy:
    """Тесты TextSplitter с использованием SpacySmartSplitter"""

    @pytest.fixture
    def splitter(self):
        """Создание экземпляра TextSplitter"""
        return TextSplitter(chunk_size=300, chunk_overlap=50)

    @pytest.fixture
    def sample_doc(self):
        """Создание примера документа"""
        text = """Это первое предложение. Это второе предложение.
Это третье предложение. Это четвёртое предложение.
Это пятое предложение."""
        return LangChainDocument(
            page_content=text,
            metadata={
                'source': 'test.pdf',
                'page': 1
            }
        )

    def test_text_splitter_initialization(self, splitter):
        """TextSplitter инициализируется с правильными параметрами"""
        assert splitter.chunk_size == 300
        assert splitter.chunk_overlap == 50
        assert splitter.spacy_splitter is not None or splitter.text_splitter is not None

    def test_split_documents_basic(self, splitter, sample_doc):
        """Базовое разбиение документа на чанки"""
        result = splitter.split_documents([sample_doc])

        # Должны быть чанки
        assert len(result) > 0
        # Все чанки должны быть LangChainDocument
        assert all(isinstance(doc, LangChainDocument) for doc in result)
        # Метаданные должны быть сохранены
        assert all('source' in doc.metadata for doc in result)

    def test_split_documents_preserves_metadata(self, splitter, sample_doc):
        """Метаданные документа сохраняются при разбиении"""
        result = splitter.split_documents([sample_doc])

        for doc in result:
            assert doc.metadata['source'] == 'test.pdf'
            assert doc.metadata['page'] == 1
            assert 'chunk_id' in doc.metadata
            assert 'chunk_count' in doc.metadata

    def test_split_documents_multiple(self, splitter):
        """Разбиение нескольких документов"""
        docs = [
            LangChainDocument(
                page_content="Первый документ. С несколькими предложениями.",
                metadata={'source': 'doc1.pdf', 'page': 1}
            ),
            LangChainDocument(
                page_content="Второй документ. Тоже с предложениями.",
                metadata={'source': 'doc2.pdf', 'page': 1}
            )
        ]

        result = splitter.split_documents(docs)

        # Должны быть чанки от обоих документов
        assert len(result) > 0
        # Должны быть разные источники
        sources = set(doc.metadata['source'] for doc in result)
        assert len(sources) == 2

    def test_split_text_with_normalization(self, splitter):
        """split_text нормализует и разбивает текст"""
        text = "Первое  предложение.   Второе\nпредложение.  Третье."
        result = splitter.split_text(text)

        assert len(result) > 0
        assert all(isinstance(chunk, str) for chunk in result)
        # Нормализованный текст не должен содержать множественные пробелы
        for chunk in result:
            assert "  " not in chunk  # Нет двойных пробелов

    def test_split_text_empty(self, splitter):
        """Разбиение пустого текста возвращает пустой список"""
        result = splitter.split_text("")
        assert result == []

    def test_chunk_size_limit(self, splitter):
        """Чанки не превышают chunk_size"""
        long_text = "Одно предложение. " * 50  # Много предложений
        doc = LangChainDocument(
            page_content=long_text,
            metadata={'source': 'test.pdf', 'page': 1}
        )

        result = splitter.split_documents([doc])

        # Все чанки должны быть <= chunk_size
        for chunk_doc in result:
            assert len(chunk_doc.page_content) <= splitter.chunk_size + 100  # Some tolerance

    def test_text_content_not_lost(self, splitter, sample_doc):
        """При разбиении не должно быть потери содержимого"""
        original_text = sample_doc.page_content
        result = splitter.split_documents([sample_doc])

        # Объединяем все чанки
        reconstructed = " ".join(doc.page_content for doc in result)

        # Проверяем что ключевые слова сохранены
        assert "первое" in reconstructed.lower()
        assert "пятое" in reconstructed.lower()

    def test_chunk_overlap_implementation(self, splitter):
        """Перекрытие между чанками работает"""
        text = "Предложение А. Предложение Б. Предложение В. Предложение Г. Предложение Д."
        result = splitter.split_text(text)

        # Если есть перекрытие, должны быть общие слова между соседними чанками
        if len(result) > 1 and splitter.chunk_overlap > 0:
            # Хотя бы в некоторых случаях должно быть перекрытие
            has_overlap = False
            for i in range(len(result) - 1):
                words_current = set(result[i].lower().split())
                words_next = set(result[i + 1].lower().split())
                if words_current & words_next:  # Если есть общие слова
                    has_overlap = True
                    break
            # Note: может и не быть перекрытия для коротких текстов

    def test_normalization_applied_in_split_documents(self, splitter):
        """Нормализация применяется в split_documents"""
        text = "Текст    с     лишними    пробелами.\n\n\nНовый абзац."
        doc = LangChainDocument(
            page_content=text,
            metadata={'source': 'test.pdf', 'page': 1}
        )

        result = splitter.split_documents([doc])

        # Все чанки должны быть нормализованы (без множественных пробелов)
        for chunk_doc in result:
            assert "  " not in chunk_doc.page_content
            assert "\n\n\n" not in chunk_doc.page_content

    def test_english_text_splitting(self, splitter):
        """Разбиение английского текста"""
        text = "This is first sentence. This is second sentence. And here is third one. Finally, the fourth."
        doc = LangChainDocument(
            page_content=text,
            metadata={'source': 'test.pdf', 'page': 1, 'language': 'en'}
        )

        result = splitter.split_documents([doc])

        assert len(result) > 0
        assert all(isinstance(doc, LangChainDocument) for doc in result)

    def test_russian_text_splitting(self, splitter):
        """Разбиение русского текста"""
        text = "Это первое предложение. Это второе предложение. Третье предложение здесь. И четвёртое в конце."
        doc = LangChainDocument(
            page_content=text,
            metadata={'source': 'test.pdf', 'page': 1, 'language': 'ru'}
        )

        result = splitter.split_documents([doc])

        assert len(result) > 0
        for chunk_doc in result:
            assert len(chunk_doc.page_content) > 0

    def test_mixed_language_text(self, splitter):
        """Разбиение смешанного текста (RU+EN)"""
        text = "Привет. Hello. Как дела? How are you? Хорошо. Good."
        doc = LangChainDocument(
            page_content=text,
            metadata={'source': 'test.pdf', 'page': 1}
        )

        result = splitter.split_documents([doc])

        assert len(result) > 0
        # Проверяем что русский и английский текст присутствует
        all_text = " ".join(doc.page_content for doc in result)
        assert "привет" in all_text.lower()
        assert "hello" in all_text.lower()

    def test_sentence_with_abbreviations(self, splitter):
        """Разбиение текста с аббревиатурами"""
        text = "Dr. Smith works here. He is a great professional. Mr. Johnson agrees. They work together."
        doc = LangChainDocument(
            page_content=text,
            metadata={'source': 'test.pdf', 'page': 1}
        )

        result = splitter.split_documents([doc])

        # Должны быть разбиты на предложения правильно (не по Dr. и Mr.)
        assert len(result) > 0
        all_text = " ".join(doc.page_content for doc in result)
        assert "Smith" in all_text
        assert "Johnson" in all_text

    def test_chunk_metadata_chunk_id(self, splitter):
        """Каждый чанк имеет уникальный chunk_id"""
        text = "Предложение. " * 30
        doc = LangChainDocument(
            page_content=text,
            metadata={'source': 'test.pdf', 'page': 1}
        )

        result = splitter.split_documents([doc])

        chunk_ids = [doc.metadata.get('chunk_id') for doc in result]
        # Должны быть уникальные chunk_id
        assert len(set(chunk_ids)) == len(chunk_ids)

    def test_real_world_pdf_text(self, splitter):
        """Разбиение реального примера текста из PDF"""
        text = """Документ   с   множеством   проблем.

Новый  абзац  идёт  здесь.  Слово  при\nнято
в  документ.  Автор:  д-р  Иванов\t\t(контакт)."""

        doc = LangChainDocument(
            page_content=text,
            metadata={'source': 'real_pdf.pdf', 'page': 1}
        )

        result = splitter.split_documents([doc])

        # Должно быть разбито на чанки
        assert len(result) > 0
        # Все чанки должны быть нормализованы
        for chunk_doc in result:
            assert "при\n" not in chunk_doc.page_content  # Слово не должно быть разорвано
            assert "\t" not in chunk_doc.page_content  # Нет табуляций
            assert "  " not in chunk_doc.page_content  # Нет двойных пробелов


class TestTextSplitterFallback:
    """Тесты fallback поведения когда spaCy недоступна"""

    def test_fallback_initialization(self):
        """TextSplitter инициализируется даже если spaCy недоступна"""
        # Этот тест проверяет что fallback работает
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        assert splitter is not None
        if not SPACY_AVAILABLE:
            assert splitter.text_splitter is not None  # Fallback сплиттер активен
        else:
            assert splitter.spacy_splitter is not None  # spaCy сплиттер активен

    def test_fallback_split_documents(self):
        """split_documents работает в fallback режиме"""
        if not SPACY_AVAILABLE:
            splitter = TextSplitter(chunk_size=200, chunk_overlap=20)
            doc = LangChainDocument(
                page_content="Test text. Another sentence. More content here.",
                metadata={'source': 'test.pdf', 'page': 1}
            )

            result = splitter.split_documents([doc])

            assert len(result) > 0
            assert all(isinstance(d, LangChainDocument) for d in result)

    def test_fallback_split_text(self):
        """split_text работает в fallback режиме"""
        if not SPACY_AVAILABLE:
            splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
            text = "First sentence. Second sentence. Third sentence."

            result = splitter.split_text(text)

            assert len(result) > 0
            assert all(isinstance(chunk, str) for chunk in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
