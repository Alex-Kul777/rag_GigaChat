"""
Тесты для проверки исправлений score в RAG системе.
Проверяет:
1. similarity_search_with_scores возвращает реальные scores
2. Scores не равны константе 1.0
3. Логирование работает правильно
"""
import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from rag_gigachat.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@pytest.fixture
def vector_store_manager():
    """Создать VectorStoreManager для тестирования"""
    with patch('rag_gigachat.core.vector_store.HuggingFaceEmbeddings'):
        manager = VectorStoreManager(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_type="huggingface",
            persist_dir=Path("/tmp/test_vector_store")
        )
        return manager


def test_similarity_search_with_scores_exists(vector_store_manager):
    """Проверить, что метод similarity_search_with_scores существует"""
    assert hasattr(vector_store_manager, 'similarity_search_with_scores')
    assert callable(getattr(vector_store_manager, 'similarity_search_with_scores'))


def test_similarity_search_with_scores_raises_when_not_initialized(vector_store_manager):
    """Проверить, что similarity_search_with_scores выбрасывает исключение если индекс не инициализирован"""
    with pytest.raises(ValueError, match="FAISS индекс не инициализирован"):
        vector_store_manager.similarity_search_with_scores("test query")


def test_similarity_search_with_scores_returns_tuples():
    """Проверить, что similarity_search_with_scores возвращает кортежи (Document, score)"""
    with patch('rag_gigachat.core.vector_store.HuggingFaceEmbeddings'):
        manager = VectorStoreManager(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_type="huggingface"
        )

        # Мокируем vector_store с методом similarity_search_with_score
        mock_docs_with_scores = [
            (Document(page_content="Doc 1", metadata={"source": "file1.pdf"}), 0.95),
            (Document(page_content="Doc 2", metadata={"source": "file2.pdf"}), 0.87),
            (Document(page_content="Doc 3", metadata={"source": "file3.pdf"}), 0.72),
        ]

        manager.vector_store = Mock()
        manager.vector_store.similarity_search_with_score = Mock(
            return_value=mock_docs_with_scores
        )
        manager.is_initialized = True

        # Проверяем результаты
        results = manager.similarity_search_with_scores("test query", k=3)

        assert len(results) == 3
        assert isinstance(results, list)

        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0 <= score <= 1  # Scores должны быть между 0 и 1

        # Проверяем, что scores разные (не все 1.0)
        scores = [score for _, score in results]
        assert len(set(scores)) > 1, "Все scores не должны быть одинаковыми"
        assert 1.0 not in scores or scores.count(1.0) < len(scores), "Не все scores могут быть 1.0"


def test_similarity_search_with_scores_logs_correctly(caplog):
    """Проверить, что логирование работает правильно в similarity_search_with_scores"""
    with patch('rag_gigachat.core.vector_store.HuggingFaceEmbeddings'):
        manager = VectorStoreManager(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_type="huggingface"
        )

        mock_docs_with_scores = [
            (Document(page_content="Doc 1", metadata={"source": "file1.pdf"}), 0.95),
        ]

        manager.vector_store = Mock()
        manager.vector_store.similarity_search_with_score = Mock(
            return_value=mock_docs_with_scores
        )
        manager.is_initialized = True

        with caplog.at_level(logging.DEBUG):
            manager.similarity_search_with_scores("test query", k=1)

        log_text = caplog.text

        # Проверяем, что в логах есть ключевые строки
        assert "ПОИСК" in log_text or "RAW SCORES" in log_text, "Логирование должно содержать информацию о поиске"


def test_similarity_search_with_scores_vs_old_method():
    """Сравнить similarity_search и similarity_search_with_scores"""
    with patch('rag_gigachat.core.vector_store.HuggingFaceEmbeddings'):
        manager = VectorStoreManager(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_type="huggingface"
        )

        # Мокируем оба метода
        mock_docs = [
            Document(page_content="Doc 1", metadata={"source": "file1.pdf"}),
            Document(page_content="Doc 2", metadata={"source": "file2.pdf"}),
        ]

        mock_docs_with_scores = [
            (Document(page_content="Doc 1", metadata={"source": "file1.pdf"}), 0.95),
            (Document(page_content="Doc 2", metadata={"source": "file2.pdf"}), 0.87),
        ]

        manager.vector_store = Mock()
        manager.vector_store.similarity_search = Mock(return_value=mock_docs)
        manager.vector_store.similarity_search_with_score = Mock(
            return_value=mock_docs_with_scores
        )
        manager.is_initialized = True

        # similarity_search не имеет scores
        docs_only = manager.similarity_search("test")
        assert len(docs_only) == 2
        assert all(isinstance(d, Document) for d in docs_only)

        # similarity_search_with_scores имеет scores
        docs_with_scores = manager.similarity_search_with_scores("test")
        assert len(docs_with_scores) == 2
        assert all(isinstance(d, Document) and isinstance(s, float) for d, s in docs_with_scores)

        # Scores в новом методе должны быть разными
        scores = [s for _, s in docs_with_scores]
        assert scores[0] != scores[1], "Scores должны быть разными для разных документов"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
