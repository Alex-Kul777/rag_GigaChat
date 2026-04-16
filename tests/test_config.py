"""
tests/test_config.py - Тесты для конфигурации
"""

import pytest
from pathlib import Path
from rag_gigachat.config import model_config, data_config, gigachat_config, vectorstore_config


class TestConfig:
    """Тесты конфигурации"""
    
    def test_model_config_defaults(self):
        """Тест значений по умолчанию модели"""
        assert model_config.llm_model_name is not None
        assert model_config.temperature >= 0
        assert model_config.max_new_tokens > 0
        assert model_config.default_k_retrieve > 0
    
    def test_data_config_defaults(self):
        """Тест значений по умолчанию данных"""
        assert data_config.chunk_size > 0
        assert data_config.chunk_overlap >= 0
        assert data_config.cache_dir is not None
        assert data_config.vectorstore_dir is not None
    
    def test_gigachat_config(self):
        """Тест конфигурации GigaChat"""
        assert gigachat_config.scope is not None
        assert gigachat_config.timeout > 0
        assert gigachat_config.model is not None
    
    def test_vectorstore_config(self):
        """Тест конфигурации векторного хранилища"""
        assert vectorstore_config.vector_store_type == "faiss"
        assert vectorstore_config.persist_dir is not None
    
    def test_config_paths_are_pathlib(self):
        """Тест что пути - объекты Path"""
        assert isinstance(data_config.cache_dir, Path)
        assert isinstance(data_config.vectorstore_dir, Path)
        assert isinstance(vectorstore_config.persist_dir, Path)
