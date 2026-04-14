"""
tests/test_smoke.py - Дымовые тесты для быстрой проверки
"""

import pytest


class TestSmoke:
    """Дымовые тесты"""
    
    def test_import_all_modules(self):
        """Проверка импорта всех основных модулей"""
        from rag_core import RAGPipeline
        from token_counter import TokenCounter
        from config import model_config, data_config
        from evaluator import WikiEvalEvaluator
        from excel_reporter import ExcelReporter
        assert True
    
    def test_config_loaded(self):
        """Проверка загрузки конфигурации"""
        from config import model_config, data_config, gigachat_config
        assert model_config is not None
        assert data_config is not None
        assert gigachat_config is not None
    
    def test_token_counter_import(self):
        """Проверка импорта счетчика токенов"""
        from token_counter import TokenCounter
        counter = TokenCounter()
        assert counter is not None
    
    @pytest.mark.slow
    def test_pipeline_initialization_slow(self):
        """Проверка инициализации пайплайна (медленный тест)"""
        from rag_core import RAGPipeline
        pipeline = RAGPipeline()
        assert pipeline is not None
        assert pipeline.vector_store_initialized is False
