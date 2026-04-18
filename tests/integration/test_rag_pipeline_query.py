"""
Integration test для RAG Pipeline - проверка обработки запросов
Тестирует: загрузку документов, поиск, генерацию ответов
"""
import pytest
import logging
from pathlib import Path
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.models import RetrievalType

# Настройка логирования для тестов
logger = logging.getLogger(__name__)

# Параметры для теста
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data/domain_2_Debug/books"
TEST_QUERY = "Что такое RAG и как оно работает?"
K_RETRIEVE = 5


class TestRAGPipelineQuery:
    """Тесты для проверки RAG pipeline в режиме query"""

    @pytest.fixture
    def pipeline(self):
        """Fixture: инициализация RAGPipeline"""
        logger.info(f"Инициализация RAGPipeline с данными из {TEST_DATA_DIR}")
        pipeline = RAGPipeline(
            retrieval_type=RetrievalType.DENSE,
            embedding_type="huggingface",
            llm_type="local"
        )
        return pipeline

    def test_pipeline_initialization(self, pipeline):
        """Тест 1: Проверка инициализации pipeline"""
        assert pipeline is not None, "Pipeline должен быть инициализирован"
        assert pipeline.vector_store_manager is not None, "VectorStoreManager должен существовать"
        assert pipeline.llm_manager is not None, "LLMManager должен существовать"
        logger.info("✅ Pipeline инициализирован успешно")

    def test_documents_loading(self, pipeline):
        """Тест 2: Проверка загрузки документов"""
        assert TEST_DATA_DIR.exists(), f"Директория {TEST_DATA_DIR} не существует"

        pipeline.load_from_pdf_directory_with_metadata(
            TEST_DATA_DIR,
            recursive=False,
            force_reload=True
        )

        assert pipeline.vector_store_initialized, "Vector store должен быть инициализирован"
        logger.info("✅ Документы загружены успешно")

    def test_query_processing(self, pipeline):
        """Тест 3: Проверка обработки запроса"""
        # Загружаем документы
        pipeline.load_from_pdf_directory_with_metadata(
            TEST_DATA_DIR,
            recursive=False,
            force_reload=True
        )

        # Обрабатываем запрос
        result = pipeline.process_query(TEST_QUERY, k=K_RETRIEVE)

        # Проверяем результат
        assert result is not None, "Результат не должен быть None"
        assert result.answer, "Ответ не должен быть пустым"
        assert len(result.answer) > 10, "Ответ должен содержать достаточно текста"

        logger.info(f"✅ Запрос обработан: {len(result.answer)} символов в ответе")

    def test_document_retrieval(self, pipeline):
        """Тест 4: Проверка поиска документов"""
        pipeline.load_from_pdf_directory_with_metadata(
            TEST_DATA_DIR,
            recursive=False,
            force_reload=True
        )

        result = pipeline.process_query(TEST_QUERY, k=K_RETRIEVE)

        # Проверяем что найдены документы
        assert result.retrieval_results is not None, "Результаты поиска не должны быть None"
        assert len(result.retrieval_results.retrieved_docs) > 0, "Должны быть найдены документы"
        assert len(result.retrieval_results.retrieved_docs) <= K_RETRIEVE, \
            f"Найдено документов должно быть <= {K_RETRIEVE}"

        logger.info(f"✅ Найдено {len(result.retrieval_results.retrieved_docs)} документов")

    def test_answer_adequacy(self, pipeline):
        """Тест 5: Проверка адекватности ответа"""
        pipeline.load_from_pdf_directory_with_metadata(
            TEST_DATA_DIR,
            recursive=False,
            force_reload=True
        )

        result = pipeline.process_query(TEST_QUERY, k=K_RETRIEVE)

        # Проверяем что ответ содержит ожидаемые ключевые слова
        answer_lower = result.answer.lower()

        # Проверяем что это не пустой или тривиальный ответ
        assert len(answer_lower.split()) > 5, "Ответ должен содержать достаточно слов"

        # Проверяем что это не просто контекст
        assert "rag" in answer_lower or "реинжиниринг" in answer_lower or \
               "генер" in answer_lower or "знани" in answer_lower, \
               "Ответ должен содержать релевантное содержание"

        logger.info(f"✅ Ответ адекватен: {answer_lower[:100]}...")

    def test_generation_metrics(self, pipeline):
        """Тест 6: Проверка метрик генерации"""
        pipeline.load_from_pdf_directory_with_metadata(
            TEST_DATA_DIR,
            recursive=False,
            force_reload=True
        )

        result = pipeline.process_query(TEST_QUERY, k=K_RETRIEVE)

        # Проверяем метрики
        assert result.generation_time > 0, "Время генерации должно быть положительным"
        assert result.generation_time < 120, "Время генерации < 120 сек (разумно для CPU/CUDA)"
        assert result.tokens_generated > 0, "Должны быть сгенерированы токены"

        logger.info(
            f"✅ Метрики: время={result.generation_time:.2f}s, "
            f"токены={result.tokens_generated}"
        )

    @pytest.mark.parametrize("query,expected_keywords", [
        ("Что такое RAG?", ["rag", "генер", "документ", "контекст"]),
        ("Как RAG работает?", ["rag", "поиск", "генер", "модель"]),
    ])
    def test_multiple_queries(self, pipeline, query, expected_keywords):
        """Тест 7: Проверка нескольких запросов"""
        pipeline.load_from_pdf_directory_with_metadata(
            TEST_DATA_DIR,
            recursive=False,
            force_reload=True
        )

        result = pipeline.process_query(query, k=K_RETRIEVE)

        assert result.answer, f"Ответ на '{query}' не должен быть пустым"
        assert len(result.retrieval_results.retrieved_docs) > 0, \
            f"Документы должны быть найдены для '{query}'"

        logger.info(f"✅ Запрос '{query}' обработан успешно")
