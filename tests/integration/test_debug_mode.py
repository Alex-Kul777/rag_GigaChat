"""
Integration test для Debug Mode — проверка инициализации и выбора модели
Покрывает регрессионные сценарии для RAG_DEBUG_MODE=true
"""
import pytest
import logging
import os
from pathlib import Path

from rag_gigachat.core.llm_manager import LLMManager
from rag_gigachat.config import model_config, debug_config

# Настройка логирования для тестов
logger = logging.getLogger(__name__)

# Параметры для теста
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data/domain_2_Debug/books"


class TestDebugModeInitialization:
    """Тесты для проверки инициализации Debug Mode"""

    def test_debug_mode_env_detection(self, monkeypatch):
        """Тест 1: Проверка что RAG_DEBUG_MODE=true выбирает facebook/opt-125m"""
        # Устанавливаем переменную окружения в текущий процесс
        monkeypatch.setenv("RAG_DEBUG_MODE", "true")

        # Создаем новый LLMManager с debug mode
        manager = LLMManager(model_type="local")

        # Проверяем что выбрана debug модель (facebook/opt-125m)
        assert manager.model_name == debug_config.debug_model_name, \
            f"В debug режиме должна выбраться {debug_config.debug_model_name}, " \
            f"но выбрана {manager.model_name}"

        # Убеждаемся что это НЕ production модель
        assert manager.model_name != model_config.llm_model_name, \
            f"Debug модель должна отличаться от production модели {model_config.llm_model_name}"

        # Проверяем конкретные ожидаемые значения
        assert manager.model_name == "facebook/opt-125m", \
            f"Debug модель должна быть facebook/opt-125m, получена {manager.model_name}"

        logger.info("✅ Debug mode инициализирован корректно с facebook/opt-125m")

    def test_production_mode_uses_correct_model(self, monkeypatch):
        """Тест 2: Без RAG_DEBUG_MODE используется production модель (Qwen)"""
        # Убеждаемся что RAG_DEBUG_MODE не установлена
        monkeypatch.delenv("RAG_DEBUG_MODE", raising=False)

        # Создаем новый LLMManager без debug mode
        manager = LLMManager(model_type="local")

        # Проверяем что выбрана production модель
        assert manager.model_name == model_config.llm_model_name, \
            f"В production режиме должна выбраться {model_config.llm_model_name}, " \
            f"но выбрана {manager.model_name}"

        # Проверяем что это НЕ debug модель
        assert manager.model_name != debug_config.debug_model_name, \
            f"Production модель должна отличаться от debug модели {debug_config.debug_model_name}"

        # Проверяем конкретные ожидаемые значения
        assert manager.model_name == "Qwen/Qwen2.5-0.5B-Instruct", \
            f"Production модель должна быть Qwen/Qwen2.5-0.5B-Instruct, получена {manager.model_name}"

        logger.info("✅ Production mode инициализирован корректно с Qwen/Qwen2.5-0.5B-Instruct")

    def test_debug_vs_production_model_difference(self, monkeypatch):
        """Тест 3: Контрольный тест на разницу моделей"""
        # Debug режим
        monkeypatch.setenv("RAG_DEBUG_MODE", "true")
        debug_manager = LLMManager(model_type="local")
        debug_model = debug_manager.model_name

        # Production режим
        monkeypatch.delenv("RAG_DEBUG_MODE", raising=False)
        prod_manager = LLMManager(model_type="local")
        prod_model = prod_manager.model_name

        # Проверяем что модели разные
        assert debug_model != prod_model, \
            f"Debug модель ({debug_model}) должна отличаться от production ({prod_model})"

        # Проверяем размер: debug должна быть меньше
        assert "opt-125m" in debug_model.lower(), f"Debug модель должна быть opt-125m, получена {debug_model}"
        assert "qwen" in prod_model.lower(), f"Production модель должна быть Qwen, получена {prod_model}"

        logger.info(f"✅ Модели корректно различаются: debug={debug_model}, prod={prod_model}")
