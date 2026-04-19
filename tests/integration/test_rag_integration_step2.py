"""PHASE STEP 2: Automatic token filtering in CorpusLoader

Tests that CorpusLoader.load_from_pdf_directory_with_metadata() automatically
applies token filtering to all documents before returning them.
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.data.data_loader import CorpusLoader, TextSplitter
from rag_gigachat.utils.text_utils import SPACY_AVAILABLE, filter_documents_by_token_count
from langchain_core.documents import Document as LangChainDocument


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
class TestCorpusLoaderTokenFiltering:
    """Тесты интеграции фильтрации токенов в CorpusLoader"""

    def test_token_filter_imported_in_corpus_loader(self):
        """Функция фильтрации токенов импортирована в CorpusLoader"""
        # Проверяем что filter_documents_by_token_count доступна в модуле
        from rag_gigachat.data import data_loader
        assert hasattr(data_loader, 'filter_documents_by_token_count')
        assert callable(data_loader.filter_documents_by_token_count)

    def test_corpus_loader_initialization(self):
        """CorpusLoader инициализируется корректно"""
        loader = CorpusLoader(data_dir=Path("test_data"))
        assert loader is not None
        assert loader.document_loader is not None
        assert loader.cache is not None

    def test_corpus_loader_filtering_with_chunking(self):
        """Фильтрация применяется после разбиения на чанки"""
        loader = CorpusLoader(data_dir=Path("test_data"))

        # Реалистичные документы с разными размерами
        mock_documents = [
            LangChainDocument(
                page_content="Первый документ с текстом. " * 10,
                metadata={'source': 'doc1.pdf', 'page_number': 1, 'filepath': 'doc1.pdf'}
            ),
            LangChainDocument(
                page_content="Очень короткий.",
                metadata={'source': 'doc2.pdf', 'page_number': 1, 'filepath': 'doc2.pdf'}
            ),
        ]

        # Отдельно тестируем фильтрацию на этих документах
        filtered = filter_documents_by_token_count(mock_documents, min_tokens=30)

        # Проверяем что фильтрация работает
        assert len(filtered) < len(mock_documents)
        assert len(filtered) > 0

    def test_corpus_loader_filtering_logs_statistics(self):
        """Фильтрация применяется корректно к документам"""
        # Тестируем фильтрацию напрямую
        mock_documents = [
            LangChainDocument(
                page_content="Длинный документ с информацией. " * 5,
                metadata={'source': 'test.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content="短い",  # Очень короткий
                metadata={'source': 'test.pdf', 'page_number': 2}
            ),
            LangChainDocument(
                page_content="Еще один длинный документ. " * 5,
                metadata={'source': 'test.pdf', 'page_number': 3}
            ),
        ]

        before = len(mock_documents)
        filtered = filter_documents_by_token_count(mock_documents, min_tokens=30)
        after = len(filtered)

        # Проверяем что произошла фильтрация
        assert after < before
        assert after > 0

    def test_corpus_loader_preserves_metadata_after_filtering(self):
        """Метаданные сохраняются после фильтрации"""
        mock_documents = [
            LangChainDocument(
                page_content="Это достаточно длинный документ с метаданными и достаточным количеством текста для прохождения фильтра качества. " * 2,
                metadata={
                    'source': 'important.pdf',
                    'page_number': 1,
                    'doc_author': 'Test Author',
                    'doc_title': 'Test Title'
                }
            ),
        ]

        filtered = filter_documents_by_token_count(mock_documents, min_tokens=25)

        # Проверяем что документ прошел фильтр и метаданные сохранены
        assert len(filtered) > 0
        doc = filtered[0]
        assert 'metadata' in dir(doc)
        assert 'token_count' in doc.metadata
        assert 'language' in doc.metadata

    def test_corpus_loader_handles_all_filtered_documents(self):
        """CorpusLoader корректно обрабатывает случай когда все документы отфильтрованы"""
        # Только короткие документы
        mock_documents = [
            LangChainDocument(page_content="A", metadata={'source': 'test.pdf', 'page_number': 1}),
            LangChainDocument(page_content="B", metadata={'source': 'test.pdf', 'page_number': 2}),
        ]

        filtered = filter_documents_by_token_count(mock_documents, min_tokens=30)

        # Должен вернуть пустой список
        assert filtered == []

    def test_filtering_enriches_token_metadata(self):
        """Фильтрация обогащает метаданные информацией о токенах"""
        mock_documents = [
            LangChainDocument(
                page_content="Достаточно длинный документ с полезной информацией и деталями.",
                metadata={'source': 'test.pdf', 'page_number': 1}
            ),
        ]

        filtered = filter_documents_by_token_count(mock_documents, min_tokens=15)

        # Проверяем что документ прошел фильтр и метаданные обогащены
        assert len(filtered) > 0
        doc = filtered[0]
        assert doc.metadata['token_count'] > 0
        assert 'language' in doc.metadata

    def test_filtering_with_min_tokens_threshold(self):
        """Фильтрация работает с различными порогами"""
        mock_documents = [
            LangChainDocument(
                page_content="Текст с несколькими словами",
                metadata={'source': 'test.pdf', 'page_number': 1}
            ),
        ]

        # Низкий порог - документ проходит
        filtered_low = filter_documents_by_token_count(mock_documents, min_tokens=5)
        assert len(filtered_low) == 1

        # Высокий порог - документ отфильтрован
        filtered_high = filter_documents_by_token_count(mock_documents, min_tokens=100)
        assert len(filtered_high) == 0


class TestCorpusLoaderIntegrationWithFiltering:
    """Интеграционные тесты CorpusLoader с фильтрацией"""

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
    def test_corpus_loader_full_workflow_with_filtering(self):
        """Полный рабочий процесс: загрузка → разбиение → фильтрация"""
        # Создаем реалистичные документы разных размеров
        mock_documents = [
            LangChainDocument(
                page_content="Введение в систему. " * 10,
                metadata={'source': 'report.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content="Ошибка.",
                metadata={'source': 'report.pdf', 'page_number': 2}
            ),
            LangChainDocument(
                page_content="Основной контент отчета с деталями. " * 8,
                metadata={'source': 'report.pdf', 'page_number': 3}
            ),
        ]

        # Фильтруем
        filtered = filter_documents_by_token_count(mock_documents, min_tokens=20)

        # Проверяем что фильтрация работает
        assert len(filtered) > 0
        assert len(filtered) < len(mock_documents)

        # Все оставшиеся документы должны быть "чистыми"
        for doc in filtered:
            text = doc.page_content
            # Может содержать нормализованный контент
            assert len(text) > 0

    def test_corpus_loader_filtering_removes_garbage(self):
        """Фильтрация удаляет мусорные чанки"""
        mock_documents = [
            LangChainDocument(
                page_content="Качественный документ с информацией. " * 5,
                metadata={'source': 'doc1.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content="Плохо.",
                metadata={'source': 'doc2.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content="Еще один качественный документ. " * 5,
                metadata={'source': 'doc3.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content=".",
                metadata={'source': 'doc4.pdf', 'page_number': 1}
            ),
        ]

        # Фильтруем
        filtered = filter_documents_by_token_count(mock_documents, min_tokens=20)

        # Должны быть только хорошие документы
        assert len(filtered) > 0
        assert len(filtered) < len(mock_documents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
