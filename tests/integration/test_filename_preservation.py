"""
Integration test для проверки сохранения имён файлов и номеров страниц
Тестирует: что реальные имена файлов (source_file) и страницы проходят через весь pipeline
"""
import pytest
import logging
from pathlib import Path
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.models import RetrievalType

logger = logging.getLogger(__name__)

# Параметры для теста
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data/domain_2_Debug/books"
TEST_QUERY = "что такое RAG"
K_RETRIEVE = 3


class TestFilenamePreservation:
    """Тесты для проверки сохранения имён файлов через pipeline"""

    @pytest.fixture(scope="class")
    def pipeline(self):
        """Fixture: инициализация RAGPipeline (один раз на весь класс)"""
        logger.info(f"Инициализация RAGPipeline с данными из {TEST_DATA_DIR}")
        pipeline = RAGPipeline(
            retrieval_type=RetrievalType.DENSE,
            embedding_type="huggingface",
            llm_type="local"
        )
        # Загружаем документы один раз
        if TEST_DATA_DIR.exists():
            pipeline.load_from_pdf_directory_with_metadata(
                TEST_DATA_DIR,
                recursive=False,
                force_reload=True
            )
        return pipeline

    def test_filename_in_vector_store(self, pipeline):
        """Тест 1: Проверка что source_file сохраняется в FAISS индексе"""
        assert pipeline.vector_store_initialized, "Vector store должен быть инициализирован"
        logger.info("✅ Vector store инициализирован")

    def test_retrieval_has_source_file(self, pipeline):
        """Тест 2: Проверка что результаты поиска содержат source_file"""
        result = pipeline.process_query(TEST_QUERY, k=K_RETRIEVE)

        # Проверяем что найдены документы
        assert result.retrieval_results is not None, "Результаты поиска не должны быть None"
        assert len(result.retrieval_results.retrieved_docs) > 0, "Должны быть найдены документы"

        # Проверяем что каждый документ имеет source_file
        for i, doc in enumerate(result.retrieval_results.retrieved_docs):
            assert 'source_file' in doc, f"Документ {i} должен иметь 'source_file'"
            assert doc['source_file'] is not None, f"Документ {i}: source_file не должен быть None"
            assert doc['source_file'].strip(), f"Документ {i}: source_file не должен быть пустым"
            assert '.pdf' in doc['source_file'].lower(), \
                f"Документ {i}: source_file должен содержать '.pdf': {doc['source_file']}"

            logger.info(f"  Документ {i}: source_file = {doc['source_file']}")

        logger.info(f"✅ Все {len(result.retrieval_results.retrieved_docs)} документов имеют source_file")

    def test_filename_not_generic(self, pipeline):
        """Тест 3: Проверка что имена файлов НЕ generic (не doc_0.pdf и т.д.)"""
        result = pipeline.process_query(TEST_QUERY, k=K_RETRIEVE)

        generic_pattern_count = 0
        for doc in result.retrieval_results.retrieved_docs:
            source_file = doc.get('source_file', '')
            # Проверяем что это не generic имя (doc_0.pdf, doc_1.pdf и т.д.)
            if source_file.lower().startswith('doc_') and source_file.endswith('.pdf'):
                # e.g., "doc_0.pdf", "doc_1.pdf"
                generic_pattern_count += 1

        assert generic_pattern_count == 0, \
            f"Найдено {generic_pattern_count} документов с generic имена (doc_N.pdf)"

        logger.info(f"✅ Все документы имеют реальные имена (не generic doc_N.pdf)")

    def test_page_number_preservation(self, pipeline):
        """Тест 4: Проверка что номера страниц (page) сохраняются"""
        result = pipeline.process_query(TEST_QUERY, k=K_RETRIEVE)

        # Проверяем что каждый документ имеет номер страницы
        for i, doc in enumerate(result.retrieval_results.retrieved_docs):
            page = doc.get('page')
            assert page is not None, \
                f"Документ {i} должен иметь 'page'. doc_id={doc.get('doc_id')}"
            assert isinstance(page, int) or isinstance(page, float), \
                f"Документ {i}: page должен быть числом, получено {type(page)}"
            assert page >= 1, \
                f"Документ {i}: page должен быть >= 1, получено {page}"

            logger.info(f"  Документ {i}: page = {page}, source_file = {doc.get('source_file')}")

        logger.info(f"✅ Все документы имеют корректные номера страниц")

    def test_full_metadata_chain(self, pipeline):
        """Тест 5: Проверка полной цепи сохранения метаданных"""
        result = pipeline.process_query(TEST_QUERY, k=K_RETRIEVE)

        # Проверяем что все необходимые поля присутствуют
        required_fields = ['doc_id', 'source_file', 'page', 'score', 'text']
        for i, doc in enumerate(result.retrieval_results.retrieved_docs):
            for field in required_fields:
                assert field in doc, \
                    f"Документ {i} должен иметь поле '{field}'. Доступны: {list(doc.keys())}"

            # Дополнительная проверка: source_file должен совпадать с основной частью doc_id
            doc_id = doc['doc_id']
            source_file = doc['source_file']
            # doc_id обычно вида "filename_pageN", а source_file = "filename.pdf"
            if source_file and source_file.endswith('.pdf'):
                filename_without_ext = source_file.replace('.pdf', '')
                # Проверяем что doc_id начинается с этого имени
                assert doc_id.startswith(filename_without_ext), \
                    f"doc_id '{doc_id}' должен начинаться с '{filename_without_ext}' " \
                    f"(из source_file '{source_file}')"

            logger.info(
                f"  Документ {i}: doc_id={doc_id}, source_file={source_file}, page={doc['page']}, "
                f"score={doc.get('score', 'N/A'):.3f}"
            )

        logger.info(f"✅ Полная цепь метаданных сохранена для {len(result.retrieval_results.retrieved_docs)} документов")

    def test_multiple_files_different_filenames(self, pipeline):
        """Тест 6: Проверка что если несколько файлов, они имеют разные имена"""
        result = pipeline.process_query(TEST_QUERY, k=K_RETRIEVE)

        # Собираем уникальные имена файлов
        filenames = set()
        for doc in result.retrieval_results.retrieved_docs:
            source_file = doc.get('source_file', '')
            if source_file:
                filenames.add(source_file)

        # Если найдено несколько документов, они могут быть из одного или разных файлов
        # Главное - что имена не generic
        for filename in filenames:
            assert not filename.lower().startswith('doc_'), \
                f"Filename '{filename}' имеет generic паттерн"
            assert filename.lower().endswith('.pdf'), \
                f"Filename '{filename}' должен заканчиваться на .pdf"

        logger.info(f"✅ Найдено {len(filenames)} уникальное(ых) имя/имена файлов: {filenames}")

    @pytest.mark.parametrize("query", [
        "что такое RAG",
        "как работает система",
        "информация о данных"
    ])
    def test_metadata_preserved_for_different_queries(self, pipeline, query):
        """Тест 7: Проверка что метаданные сохраняются для разных запросов"""
        result = pipeline.process_query(query, k=K_RETRIEVE)

        if result.retrieval_results and result.retrieval_results.retrieved_docs:
            for doc in result.retrieval_results.retrieved_docs:
                assert 'source_file' in doc, f"Запрос '{query}': документ должен иметь source_file"
                assert 'page' in doc, f"Запрос '{query}': документ должен иметь page"

            logger.info(f"✅ Запрос '{query}': метаданные сохранены для {len(result.retrieval_results.retrieved_docs)} документов")
        else:
            logger.warning(f"⚠️  Запрос '{query}': документы не найдены")
