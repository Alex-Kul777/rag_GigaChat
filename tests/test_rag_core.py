"""
tests/test_rag_core.py - Тесты для RAG пайплайна
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_core import RAGPipeline, VectorStoreManager, LLMManager


class TestRAGPipeline:
    """Тесты для RAGPipeline"""
    
    def test_initialization(self):
        """Тест инициализации RAGPipeline"""
        pipeline = RAGPipeline()
        assert pipeline is not None
        assert pipeline.vector_store_initialized is False
    
    def test_load_documents_from_dict(self, sample_documents, mock_embeddings):
        """Тест загрузки документов из словаря"""
        pipeline = RAGPipeline()
        pipeline.load_documents_from_dict(sample_documents)
        assert pipeline.vector_store_initialized is True

    def test_get_stats(self, sample_documents, mock_embeddings):
        """Тест получения статистики"""
        pipeline = RAGPipeline()
        pipeline.load_documents_from_dict(sample_documents)

        stats = pipeline.get_stats()
        assert 'vector_store_initialized' in stats
        assert stats['vector_store_initialized'] is True
        assert 'chunk_size' in stats
        assert 'chunk_overlap' in stats

    @pytest.mark.mock
    def test_process_query_returns_result(self, sample_documents, mock_embeddings, mock_gigachat):
        """Тест обработки запроса возвращает результат"""
        pipeline = RAGPipeline()
        pipeline.load_documents_from_dict(sample_documents)
        
        result = pipeline.process_query("Что такое нейросети?", k=1)
        
        assert result is not None
        assert result.query_text == "Что такое нейросети?"
        assert result.answer is not None


class TestVectorStoreManager:
    """Тесты для VectorStoreManager"""
    
    def test_initialization(self):
        """Тест инициализации VectorStoreManager"""
        manager = VectorStoreManager()
        assert manager is not None
        assert manager.is_initialized is False
    
    def test_get_hash(self):
        """Тест генерации хеша"""
        manager = VectorStoreManager()
        docs = {"doc1": "text1", "doc2": "text2"}
        hash1 = manager._get_hash(docs)
        hash2 = manager._get_hash(docs)
        assert hash1 == hash2
    
    def test_check_cache_exists(self):
        """Тест проверки существования кэша"""
        manager = VectorStoreManager()
        exists = manager.check_cache_exists("test_hash")
        # Несуществующий кэш
        assert exists is False
