"""
tests/test_smoke.py - Дымовые тесты для быстрой проверки
"""

import pytest


class TestSmoke:
    """Дымовые тесты"""
    
    def test_import_all_modules(self):
        """Проверка импорта всех основных модулей"""
        from rag_gigachat.core.rag_pipeline import RAGPipeline
        from rag_gigachat.token_counter import TokenCounter
        from rag_gigachat.config import model_config, data_config
        from rag_gigachat.reporting.evaluator import WikiEvalEvaluator
        from rag_gigachat.reporting.excel_reporter import ExcelReporter
        assert True

    def test_config_loaded(self):
        """Проверка загрузки конфигурации"""
        from rag_gigachat.config import model_config, data_config, gigachat_config
        assert model_config is not None
        assert data_config is not None
        assert gigachat_config is not None

    def test_token_counter_import(self):
        """Проверка импорта счетчика токенов"""
        from rag_gigachat.token_counter import TokenCounter
        counter = TokenCounter()
        assert counter is not None

    @pytest.mark.slow
    def test_pipeline_initialization_slow(self):
        """Проверка инициализации пайплайна (медленный тест)"""
        from rag_gigachat.core.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        assert pipeline is not None
        assert pipeline.vector_store_initialized is False
