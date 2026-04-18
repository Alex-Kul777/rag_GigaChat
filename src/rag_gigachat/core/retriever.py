"""
retriever.py - Strategy pattern для поиска документов.

Определяет протокол BaseRetriever и конкретные стратегии:
  DenseRetriever  — векторный поиск через FAISS (dense)
  SparseRetriever — заглушка для BM25/TF-IDF (sparse, future)
  HybridRetriever — заглушка для гибридного поиска (future)
"""
import logging
from typing import List, Protocol, runtime_checkable

from langchain_core.documents import Document

from rag_gigachat.models import RetrievalType

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseRetriever(Protocol):
    """Протокол стратегии поиска документов"""

    retrieval_type: RetrievalType

    def search(self, query: str, k: int) -> List[Document]:
        """
        Поиск документов по запросу

        Args:
            query: Текст запроса
            k: Количество результатов

        Returns:
            Список релевантных документов
        """
        ...


class DenseRetriever:
    """
    Стратегия векторного поиска (dense).
    Использует FAISS через VectorStoreManager.
    """

    retrieval_type = RetrievalType.DENSE

    def __init__(self, vector_store_manager):
        """
        Args:
            vector_store_manager: Инициализированный VectorStoreManager
        """
        self._vsm = vector_store_manager

    def search(self, query: str, k: int) -> List[Document]:
        """Поиск через FAISS similarity search"""
        logger.debug(f"DenseRetriever: поиск '{query[:50]}...', k={k}")
        results = self._vsm.similarity_search(query, k=k)

        # 🔍 ДИАГНОСТИКА: Логируем что вернулось от поиска
        for i, doc in enumerate(results):
            logger.debug(
                f"🔎 RETRIEVER_RESULT[{i}]: source={doc.metadata.get('source')}, "
                f"metadata_keys={list(doc.metadata.keys())}"
            )

        return results


class SparseRetriever:
    """
    Заглушка стратегии лексического поиска (sparse / BM25).
    Реализация добавляется в будущем.
    """

    retrieval_type = RetrievalType.SPARSE

    def __init__(self, vector_store_manager):
        self._vsm = vector_store_manager
        logger.warning(
            "SparseRetriever не реализован — используется DenseRetriever как fallback"
        )

    def search(self, query: str, k: int) -> List[Document]:
        """Fallback на dense поиск до реализации BM25"""
        return self._vsm.similarity_search(query, k=k)


class HybridRetriever:
    """
    Заглушка гибридного поиска (dense + sparse).
    Реализация добавляется в будущем.
    """

    retrieval_type = RetrievalType.HYBRID

    def __init__(self, vector_store_manager, dense_weight: float = 0.7):
        """
        Args:
            vector_store_manager: Инициализированный VectorStoreManager
            dense_weight: Вес dense компоненты (0–1)
        """
        self._vsm = vector_store_manager
        self.dense_weight = dense_weight
        logger.warning(
            "HybridRetriever не реализован — используется DenseRetriever как fallback"
        )

    def search(self, query: str, k: int) -> List[Document]:
        """Fallback на dense поиск до реализации гибридного поиска"""
        return self._vsm.similarity_search(query, k=k)


def make_retriever(
        retrieval_type: RetrievalType,
        vector_store_manager,
        **kwargs
) -> BaseRetriever:
    """
    Фабрика ретриверов по типу поиска

    Args:
        retrieval_type: Тип поиска
        vector_store_manager: Инициализированный VectorStoreManager
        **kwargs: Дополнительные параметры (например, dense_weight)

    Returns:
        Экземпляр соответствующего ретривера
    """
    if retrieval_type == RetrievalType.SPARSE:
        return SparseRetriever(vector_store_manager)
    if retrieval_type == RetrievalType.HYBRID:
        return HybridRetriever(
            vector_store_manager,
            dense_weight=kwargs.get('dense_weight', 0.7)
        )
    return DenseRetriever(vector_store_manager)
