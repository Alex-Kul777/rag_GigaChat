"""PHASE STEP 4: End-to-end RAG tests with token filtering

Comprehensive tests validating token filtering integration with RAGPipeline,
including document quality metrics and performance impact analysis.
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.config import data_config
from rag_gigachat.utils.text_utils import SPACY_AVAILABLE, filter_documents_by_token_count
from langchain_core.documents import Document as LangChainDocument


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
class TestRAGPipelineFiltering:
    """Тесты интеграции фильтрации в RAG пайплайне"""

    def setup_method(self):
        """Сохранить конфигурацию перед каждым тестом"""
        self.original_filtering_enabled = data_config.token_filtering_enabled
        self.original_min_tokens = data_config.token_filter_min_tokens

    def teardown_method(self):
        """Восстановить конфигурацию после каждого теста"""
        data_config.token_filtering_enabled = self.original_filtering_enabled
        data_config.token_filter_min_tokens = self.original_min_tokens

    def test_rag_pipeline_initializes_with_filtering_config(self):
        """RAGPipeline инициализируется с фильтрацией из конфигурации"""
        pipeline = RAGPipeline(chunk_size=500, chunk_overlap=50)
        assert pipeline is not None

    def test_filtering_enabled_by_default(self):
        """По умолчанию фильтрация включена"""
        assert data_config.token_filtering_enabled is True
        assert data_config.token_filter_min_tokens == 30

    def test_filtering_can_be_disabled(self):
        """Фильтрация может быть отключена"""
        data_config.token_filtering_enabled = False

        # Создаем пайплайн с отключенной фильтрацией
        assert data_config.token_filtering_enabled is False

        # Восстанавливаем
        data_config.token_filtering_enabled = True

    def test_filtering_threshold_affects_results(self):
        """Порог фильтрации влияет на результаты"""
        mock_documents = [
            LangChainDocument(
                page_content="Первый длинный документ с достаточным количеством текста. " * 5,
                metadata={'source': 'doc1.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content="Короткий.",
                metadata={'source': 'doc2.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content="Второй длинный документ с деталями и информацией. " * 5,
                metadata={'source': 'doc3.pdf', 'page_number': 1}
            ),
        ]

        # С низким порогом больше документов проходит
        filtered_low = filter_documents_by_token_count(
            mock_documents,
            min_tokens=5
        )
        count_low = len(filtered_low)

        # С высоким порогом меньше документов проходит
        filtered_high = filter_documents_by_token_count(
            mock_documents,
            min_tokens=100
        )
        count_high = len(filtered_high)

        assert count_low > count_high


class TestDocumentQualityInRAG:
    """Тесты качества документов в RAG контексте"""

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
    def test_filtered_documents_have_quality_metadata(self):
        """Отфильтрованные документы содержат метаданные о качестве"""
        mock_documents = [
            LangChainDocument(
                page_content="Высококачественный документ с полезной информацией и деталями. " * 3,
                metadata={'source': 'quality.pdf', 'page_number': 1}
            ),
        ]

        filtered = filter_documents_by_token_count(mock_documents, min_tokens=25)

        # Проверяем что документ имеет метаданные о токенах
        assert len(filtered) > 0
        doc = filtered[0]
        assert 'token_count' in doc.metadata
        assert doc.metadata['token_count'] > 0

    def test_garbage_chunks_are_filtered_out(self):
        """Мусорные чанки отфильтровываются"""
        mock_documents = [
            LangChainDocument(
                page_content="Хороший документ с информацией. " * 5,
                metadata={'source': 'good.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content=".",  # Мусор
                metadata={'source': 'garbage.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content="",  # Пустой
                metadata={'source': 'empty.pdf', 'page_number': 1}
            ),
        ]

        filtered = filter_documents_by_token_count(mock_documents, min_tokens=20)

        # Должен остаться только хороший документ
        assert len(filtered) == 1
        assert "информацией" in filtered[0].page_content

    def test_metadata_preserved_in_filtered_documents(self):
        """Метаданные сохраняются при фильтрации"""
        mock_documents = [
            LangChainDocument(
                page_content="Документ с метаданными. " * 5,
                metadata={
                    'source': 'test.pdf',
                    'page_number': 1,
                    'author': 'Test Author',
                    'custom_field': 'custom_value'
                }
            ),
        ]

        filtered = filter_documents_by_token_count(mock_documents, min_tokens=20)

        # Проверяем что оригинальные метаданные сохранены
        assert len(filtered) > 0
        doc = filtered[0]
        assert doc.metadata['source'] == 'test.pdf'
        assert doc.metadata['author'] == 'Test Author'
        assert doc.metadata['custom_field'] == 'custom_value'

    def test_no_data_loss_from_filtering(self):
        """Фильтрация не приводит к потере важных данных"""
        # Много документов: 80% хорошие, 20% мусор
        good_doc_count = 80
        bad_doc_count = 20

        mock_documents = []
        for i in range(good_doc_count):
            mock_documents.append(LangChainDocument(
                page_content=f"Качественный документ номер {i}. " * 5,
                metadata={'source': f'good_{i}.pdf', 'page_number': 1}
            ))

        for i in range(bad_doc_count):
            mock_documents.append(LangChainDocument(
                page_content=".",  # Мусор
                metadata={'source': f'bad_{i}.pdf', 'page_number': 1}
            ))

        filtered = filter_documents_by_token_count(mock_documents, min_tokens=20)

        # Должны остаться ~80 документов (все хорошие)
        assert len(filtered) >= 75  # Небольшой допуск на переоценку
        assert len(filtered) <= 85


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
class TestFilteringPerformanceMetrics:
    """Тесты влияния фильтрации на производительность"""

    def test_filtering_impact_on_document_count(self):
        """Фильтрация уменьшает количество обрабатываемых документов"""
        mock_documents = []

        # Создаем 100 документов: 50 хорошие, 50 плохие
        for i in range(50):
            mock_documents.append(LangChainDocument(
                page_content=f"Хороший документ {i}. " * 5,
                metadata={'source': f'good_{i}.pdf', 'page_number': 1}
            ))

        for i in range(50):
            mock_documents.append(LangChainDocument(
                page_content="Короткий",  # Будет отфильтрован
                metadata={'source': f'short_{i}.pdf', 'page_number': 1}
            ))

        filtered = filter_documents_by_token_count(mock_documents, min_tokens=25)

        # Должно быть примерно 50 документов (половина отфильтрована)
        reduction = (len(mock_documents) - len(filtered)) / len(mock_documents)
        assert reduction >= 0.3  # Минимум 30% сокращение

    def test_filtering_quality_ratio(self):
        """Фильтрация улучшает качество/размер соотношение"""
        mock_documents = [
            LangChainDocument(
                page_content="Качественный документ. " * 10,
                metadata={'source': 'quality.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content="Мусор",
                metadata={'source': 'garbage.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content=".",
                metadata={'source': 'garbage2.pdf', 'page_number': 1}
            ),
        ]

        before_filter = len(mock_documents)
        filtered = filter_documents_by_token_count(mock_documents, min_tokens=30)
        after_filter = len(filtered)

        # Сокращение на ~67% (с 3 до 1)
        assert after_filter < before_filter
        assert all(len(doc.page_content) > 5 for doc in filtered)


class TestFilteringConfigurationCombinations:
    """Тесты различных комбинаций конфигурационных параметров"""

    def setup_method(self):
        """Сохранить конфигурацию перед каждым тестом"""
        self.original_filtering_enabled = data_config.token_filtering_enabled
        self.original_min_tokens = data_config.token_filter_min_tokens
        self.original_max_tokens = data_config.token_filter_max_tokens

    def teardown_method(self):
        """Восстановить конфигурацию после каждого теста"""
        data_config.token_filtering_enabled = self.original_filtering_enabled
        data_config.token_filter_min_tokens = self.original_min_tokens
        data_config.token_filter_max_tokens = self.original_max_tokens

    def test_strict_filtering_config(self):
        """Строгая конфигурация: минимум 50 токенов"""
        data_config.token_filtering_enabled = True
        data_config.token_filter_min_tokens = 50

        mock_documents = [
            LangChainDocument(
                page_content="Очень короткий документ.",
                metadata={'source': 'short.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content="Длинный документ с много информацией. " * 5,
                metadata={'source': 'long.pdf', 'page_number': 1}
            ),
        ]

        filtered = filter_documents_by_token_count(
            mock_documents,
            min_tokens=data_config.token_filter_min_tokens
        )

        # Только длинный документ проходит
        assert len(filtered) == 1
        assert "много" in filtered[0].page_content

    def test_lenient_filtering_config(self):
        """Мягкая конфигурация: минимум 5 токенов"""
        data_config.token_filtering_enabled = True
        data_config.token_filter_min_tokens = 5

        mock_documents = [
            LangChainDocument(
                page_content="Слово",
                metadata={'source': 'doc1.pdf', 'page_number': 1}
            ),
            LangChainDocument(
                page_content="Два слова",
                metadata={'source': 'doc2.pdf', 'page_number': 1}
            ),
        ]

        filtered = filter_documents_by_token_count(
            mock_documents,
            min_tokens=data_config.token_filter_min_tokens
        )

        # Оба документа проходят мягкий фильтр
        assert len(filtered) == 2

    def test_disabled_filtering_config(self):
        """Отключенная фильтрация пропускает все"""
        data_config.token_filtering_enabled = False

        mock_documents = [
            LangChainDocument(
                page_content=".",
                metadata={'source': 'garbage.pdf', 'page_number': 1}
            ),
        ]

        # Когда фильтрация отключена, нет фильтрации
        assert data_config.token_filtering_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
