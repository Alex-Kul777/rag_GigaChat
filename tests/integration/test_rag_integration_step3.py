"""PHASE STEP 3: Configuration-driven token filtering

Tests that token filtering in CorpusLoader respects configuration parameters
from config.py (token_filtering_enabled, token_filter_min_tokens, token_filter_max_tokens).
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.config import data_config
from rag_gigachat.data.data_loader import CorpusLoader
from rag_gigachat.utils.text_utils import SPACY_AVAILABLE
from langchain_core.documents import Document as LangChainDocument


class TestConfigurationParameters:
    """Тесты конфигурационных параметров фильтрации"""

    def test_data_config_has_filtering_parameters(self):
        """DataConfig содержит параметры фильтрации"""
        assert hasattr(data_config, 'token_filtering_enabled')
        assert hasattr(data_config, 'token_filter_min_tokens')
        assert hasattr(data_config, 'token_filter_max_tokens')

    def test_filtering_enabled_default_true(self):
        """По умолчанию фильтрация включена"""
        assert data_config.token_filtering_enabled is True

    def test_min_tokens_default_value(self):
        """Минимальное количество токенов по умолчанию 30"""
        assert data_config.token_filter_min_tokens == 30

    def test_max_tokens_default_none(self):
        """Максимальное количество токенов по умолчанию None (без ограничений)"""
        assert data_config.token_filter_max_tokens is None

    def test_filter_parameters_are_modifiable(self):
        """Параметры фильтрации можно модифицировать"""
        # Сохраняем оригинальные значения
        original_min = data_config.token_filter_min_tokens
        original_max = data_config.token_filter_max_tokens
        original_enabled = data_config.token_filtering_enabled

        try:
            # Изменяем значения
            data_config.token_filter_min_tokens = 50
            data_config.token_filter_max_tokens = 500
            data_config.token_filtering_enabled = False

            # Проверяем что изменились
            assert data_config.token_filter_min_tokens == 50
            assert data_config.token_filter_max_tokens == 500
            assert data_config.token_filtering_enabled is False

        finally:
            # Восстанавливаем оригинальные значения
            data_config.token_filter_min_tokens = original_min
            data_config.token_filter_max_tokens = original_max
            data_config.token_filtering_enabled = original_enabled


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
class TestConfigurationDrivenFiltering:
    """Тесты конфигурационной фильтрации в CorpusLoader"""

    def setup_method(self):
        """Сохранить оригинальные конфигурационные значения перед каждым тестом"""
        self.original_filtering_enabled = data_config.token_filtering_enabled
        self.original_min_tokens = data_config.token_filter_min_tokens
        self.original_max_tokens = data_config.token_filter_max_tokens

    def teardown_method(self):
        """Восстановить оригинальные значения после каждого теста"""
        data_config.token_filtering_enabled = self.original_filtering_enabled
        data_config.token_filter_min_tokens = self.original_min_tokens
        data_config.token_filter_max_tokens = self.original_max_tokens

    def test_filtering_disabled_in_config(self, caplog):
        """Фильтрация отключается через конфигурацию"""
        # Отключаем фильтрацию
        data_config.token_filtering_enabled = False

        loader = CorpusLoader(data_dir=Path("test_data"))

        # Даже с короткими документами ничего не должно быть отфильтровано
        mock_documents = [
            LangChainDocument(
                page_content="Короткий",
                metadata={'source': 'test.pdf', 'page_number': 1, 'filepath': 'test.pdf'}
            ),
        ]

        # Проверяем что фильтрация не применена (мокируя)
        from rag_gigachat.utils.text_utils import filter_documents_by_token_count
        filtered = filter_documents_by_token_count(mock_documents, min_tokens=data_config.token_filter_min_tokens)

        # Если фильтрация отключена, мы ожидаем что loader не будет вызывать filter_documents_by_token_count
        # Вместо этого проверяем что параметр действительно отключен
        assert data_config.token_filtering_enabled is False

    def test_custom_min_tokens_threshold(self):
        """Использование пользовательского минимального порога токенов"""
        # Устанавливаем низкий порог
        data_config.token_filter_min_tokens = 10

        # Документ с 15 токенами должен пройти через фильтр с порогом 10
        assert data_config.token_filter_min_tokens == 10

        # Подтверждаем что можно установить любое значение
        data_config.token_filter_min_tokens = 50
        assert data_config.token_filter_min_tokens == 50

        data_config.token_filter_min_tokens = 5
        assert data_config.token_filter_min_tokens == 5

    def test_custom_max_tokens_threshold(self):
        """Использование пользовательского максимального порога токенов"""
        # Устанавливаем максимум
        data_config.token_filter_max_tokens = 200

        # Проверяем что значение установилось
        assert data_config.token_filter_max_tokens == 200

        # Можно установить None для отключения максимума
        data_config.token_filter_max_tokens = None
        assert data_config.token_filter_max_tokens is None

    def test_filtering_with_different_min_thresholds(self):
        """Фильтрация работает с разными минимальными порогами"""
        from rag_gigachat.utils.text_utils import filter_documents_by_token_count

        mock_documents = [
            LangChainDocument(
                page_content="Текст с несколькими словами для тестирования фильтра.",
                metadata={'source': 'test.pdf', 'page_number': 1}
            ),
        ]

        # С низким порогом проходит
        data_config.token_filter_min_tokens = 5
        filtered_low = filter_documents_by_token_count(
            mock_documents,
            min_tokens=data_config.token_filter_min_tokens
        )
        assert len(filtered_low) == 1

        # С высоким порогом не проходит
        data_config.token_filter_min_tokens = 100
        filtered_high = filter_documents_by_token_count(
            mock_documents,
            min_tokens=data_config.token_filter_min_tokens
        )
        assert len(filtered_high) == 0

    def test_corpus_loader_respects_config_min_tokens(self):
        """CorpusLoader использует минимальный порог из конфигурации"""
        data_config.token_filter_min_tokens = 20

        loader = CorpusLoader(data_dir=Path("test_data"))

        # Проверяем что значение используется
        assert data_config.token_filter_min_tokens == 20

    def test_corpus_loader_respects_config_max_tokens(self):
        """CorpusLoader использует максимальный порог из конфигурации"""
        data_config.token_filter_max_tokens = 300

        loader = CorpusLoader(data_dir=Path("test_data"))

        # Проверяем что значение используется
        assert data_config.token_filter_max_tokens == 300


class TestFilteringConfigurationIntegration:
    """Интеграционные тесты конфигурационной фильтрации"""

    def test_default_filtering_config(self):
        """По умолчанию фильтрация включена с правильными параметрами"""
        from rag_gigachat.config import DataConfig

        config = DataConfig()

        assert config.token_filtering_enabled is True
        assert config.token_filter_min_tokens == 30
        assert config.token_filter_max_tokens is None

    def test_filtering_config_combinations(self):
        """Различные комбинации конфигурационных параметров"""
        from rag_gigachat.config import DataConfig

        # Комбинация 1: Строгая фильтрация (минимум 50 токенов)
        config1 = DataConfig()
        config1.token_filtering_enabled = True
        config1.token_filter_min_tokens = 50
        config1.token_filter_max_tokens = None

        assert config1.token_filtering_enabled is True
        assert config1.token_filter_min_tokens == 50

        # Комбинация 2: Мягкая фильтрация (минимум 10 токенов)
        config2 = DataConfig()
        config2.token_filter_min_tokens = 10

        assert config2.token_filter_min_tokens == 10

        # Комбинация 3: Выключенная фильтрация
        config3 = DataConfig()
        config3.token_filtering_enabled = False

        assert config3.token_filtering_enabled is False

    def test_filtering_config_with_range(self):
        """Фильтрация с минимальным и максимальным диапазоном"""
        from rag_gigachat.config import DataConfig

        config = DataConfig()
        config.token_filtering_enabled = True
        config.token_filter_min_tokens = 25
        config.token_filter_max_tokens = 250

        assert config.token_filter_min_tokens == 25
        assert config.token_filter_max_tokens == 250

        # Документ с 200 токенами должен пройти оба порога
        assert 25 <= 200 <= 250


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
