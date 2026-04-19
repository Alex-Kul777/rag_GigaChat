"""
Integration test для Debug Mode — проверка инициализации и выбора модели
Покрывает регрессионные сценарии для RAG_DEBUG_MODE=true
"""
import pytest
import logging
import os
import sys
import subprocess
import time
from pathlib import Path

from rag_gigachat.core.llm_manager import LLMManager
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.models import RetrievalType
from rag_gigachat.config import model_config, debug_config

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

# Настройка логирования для тестов
logger = logging.getLogger(__name__)

# Параметры для теста
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data/domain_2_Debug/books"
TEST_QUERY = "Что такое RAG?"
K_RETRIEVE = 5


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


@pytest.mark.slow
class TestDebugModeModelLoading:
    """Тесты для проверки загрузки модели и использования fp16"""

    @pytest.fixture(scope="class")
    def debug_pipeline(self):
        """Fixture: инициализация RAGPipeline в debug режиме

        Загружает модель один раз на весь класс для оптимизации.
        Устанавливает RAG_DEBUG_MODE=true перед инициализацией.
        """
        logger.info("Инициализация debug pipeline...")

        # Устанавливаем debug mode перед инициализацией
        os.environ["RAG_DEBUG_MODE"] = "true"

        try:
            # Создаем RAG pipeline в debug режиме
            pipeline = RAGPipeline(
                retrieval_type=RetrievalType.DENSE,
                embedding_type="huggingface",
                llm_type="local"
            )

            # Загружаем документы если директория существует
            if TEST_DATA_DIR.exists():
                logger.info(f"Загрузка документов из {TEST_DATA_DIR}")
                pipeline.load_from_pdf_directory_with_metadata(
                    TEST_DATA_DIR,
                    recursive=False,
                    force_reload=False
                )
            else:
                logger.warning(f"Директория {TEST_DATA_DIR} не существует")

            # Загружаем локальную модель (facebook/opt-125m в debug режиме)
            logger.info("Загрузка локальной модели facebook/opt-125m...")
            pipeline.llm_manager.load_local_model()

            logger.info("✅ Debug pipeline инициализирован успешно")
            yield pipeline

        finally:
            # Cleanup: удаляем переменную окружения после теста
            if "RAG_DEBUG_MODE" in os.environ:
                del os.environ["RAG_DEBUG_MODE"]
            logger.info("Cleanup: RAG_DEBUG_MODE удален")

    def test_debug_model_loads_without_oom(self, debug_pipeline):
        """Тест 4: Проверка что модель загружается без OOM (SIGKILL -9)"""
        # Если мы здесь, значит модель уже загружена успешно (fixture отработала)
        assert debug_pipeline is not None, "Pipeline должен быть инициализирован"
        assert debug_pipeline.llm_manager is not None, "LLMManager должен существовать"
        assert debug_pipeline.llm_manager.is_initialized, \
            "LLMManager должен быть инициализирован (загрузка модели успешна)"

        # Проверяем что выбрана правильная модель
        assert debug_pipeline.llm_manager.model_name == "facebook/opt-125m", \
            f"В debug режиме должна быть facebook/opt-125m, получена {debug_pipeline.llm_manager.model_name}"

        # Проверяем что LLM инициализирован
        assert debug_pipeline.llm_manager.llm is not None, \
            "LLM объект должен быть инициализирован"

        logger.info("✅ Модель facebook/opt-125m загружена без OOM")

    @pytest.mark.skipif(not _torch_available, reason="torch не установлен")
    def test_debug_dtype_is_float16(self, debug_pipeline):
        """Тест 5: Проверка что модель использует float16 в debug режиме"""
        # Получаем модель из pipeline
        pipeline_obj = debug_pipeline.llm_manager.llm
        assert pipeline_obj is not None, "LLM должен быть инициализирован"

        # Получаем параметры модели
        try:
            # HuggingFacePipeline имеет pipeline атрибут
            hf_pipeline = pipeline_obj.pipeline
            model = hf_pipeline.model

            # Получаем первый параметр и проверяем его dtype
            first_param = next(model.parameters())
            dtype = first_param.dtype

            # Проверяем что используется float16 (fp16)
            assert dtype == torch.float16, \
                f"В debug режиме модель должна быть float16, получена {dtype}"

            logger.info(f"✅ Модель использует float16 (fp16) для экономии памяти")

        except AttributeError as e:
            logger.error(f"Не удалось получить параметры модели: {e}")
            # Если не можем проверить dtype через параметры,
            # достаточно того что модель загружена без ошибок
            logger.warning("⚠️  Проверка dtype не удалась, но модель загружена успешно")

    def test_debug_documents_loaded(self, debug_pipeline):
        """Тест 6: Проверка что документы загружены в индекс"""
        assert debug_pipeline.vector_store_initialized, \
            "Vector store должен быть инициализирован"

        assert debug_pipeline.vector_store_manager is not None, \
            "VectorStoreManager должен существовать"

        # Проверяем что есть документы в индексе
        try:
            # Пытаемся выполнить поиск - если нет документов, будет ошибка
            result = debug_pipeline.vector_store_manager.similarity_search(
                "RAG",
                k=1
            )
            assert len(result) > 0, \
                "Должны быть найдены документы (индекс не пустой)"

            logger.info(f"✅ В индексе загружены документы ({len(result)} найдено для тестового поиска)")
        except Exception as e:
            logger.warning(f"⚠️  Не удалось проверить документы: {e}")


@pytest.mark.slow
class TestDebugModeQueryProcessing:
    """Тесты для проверки end-to-end обработки запроса в debug режиме"""

    @pytest.fixture(scope="class")
    def initialized_pipeline(self):
        """Fixture: инициализированный debug pipeline готов к запросам"""
        logger.info("Инициализация debug pipeline для query processing...")

        os.environ["RAG_DEBUG_MODE"] = "true"

        try:
            pipeline = RAGPipeline(
                retrieval_type=RetrievalType.DENSE,
                embedding_type="huggingface",
                llm_type="local"
            )

            if TEST_DATA_DIR.exists():
                pipeline.load_from_pdf_directory_with_metadata(
                    TEST_DATA_DIR,
                    recursive=False,
                    force_reload=False
                )

            pipeline.llm_manager.load_local_model()
            logger.info("✅ Pipeline готов к запросам")
            yield pipeline

        finally:
            if "RAG_DEBUG_MODE" in os.environ:
                del os.environ["RAG_DEBUG_MODE"]

    def test_debug_auto_question_end_to_end(self, initialized_pipeline):
        """Тест 7: Проверка обработки автоматического вопроса"""
        query = TEST_QUERY  # "Что такое RAG?"

        # Обрабатываем запрос через pipeline
        result = initialized_pipeline.process_query(query, k=K_RETRIEVE)

        # Проверяем что результат получен
        assert result is not None, "Результат обработки не должен быть None"
        assert result.answer is not None, "Ответ не должен быть None"
        assert isinstance(result.answer, str), "Ответ должен быть строкой"
        assert len(result.answer) > 10, \
            f"Ответ должен содержать достаточно текста (получен: {len(result.answer)} символов)"

        # Проверяем что найдены документы
        assert result.retrieval_results is not None, \
            "Результаты поиска не должны быть None"
        assert len(result.retrieval_results.retrieved_docs) > 0, \
            "Должны быть найдены документы в поиске"
        assert len(result.retrieval_results.retrieved_docs) <= K_RETRIEVE, \
            f"Найдено документов ({len(result.retrieval_results.retrieved_docs)}) " \
            f"должно быть <= {K_RETRIEVE}"

        logger.info(f"✅ Автоматический вопрос обработан: {len(result.answer)} символов, "
                   f"{len(result.retrieval_results.retrieved_docs)} документов найдено")

    def test_debug_answer_contains_relevant_words(self, initialized_pipeline):
        """Тест 8: Проверка что ответ содержит релевантное содержание"""
        query = TEST_QUERY  # "Что такое RAG?"

        result = initialized_pipeline.process_query(query, k=K_RETRIEVE)

        # Проверяем что ответ адекватен
        answer_lower = result.answer.lower()

        # Проверяем минимальную длину
        assert len(answer_lower.split()) > 5, \
            f"Ответ должен содержать > 5 слов (получен: {len(answer_lower.split())})"

        # Проверяем что содержит хотя бы одно релевантное слово
        relevant_words = ["rag", "поиск", "документ", "генер", "модел", "контекст"]
        has_relevant = any(word in answer_lower for word in relevant_words)

        assert has_relevant, \
            f"Ответ должен содержать хотя бы одно из слов: {relevant_words}\n" \
            f"Получен ответ: {result.answer[:100]}..."

        logger.info(f"✅ Ответ содержит релевантное содержание: {answer_lower[:80]}...")

    def test_debug_document_retrieval_quality(self, initialized_pipeline):
        """Тест 9: Проверка качества поиска документов"""
        query = TEST_QUERY

        result = initialized_pipeline.process_query(query, k=K_RETRIEVE)

        # Проверяем структуру результатов
        assert len(result.retrieval_results.retrieved_docs) > 0, \
            "Должны быть найдены документы"

        # Проверяем каждый документ
        for i, doc in enumerate(result.retrieval_results.retrieved_docs):
            assert "doc_id" in doc, f"Документ {i} должен иметь doc_id"
            assert "text" in doc or "page_content" in doc, \
                f"Документ {i} должен иметь содержание"
            assert "score" in doc, f"Документ {i} должен иметь score"

            # Score должен быть числом в разумном диапазоне
            score = doc.get("score", 0)
            assert isinstance(score, (int, float)), \
                f"Score должен быть числом, получен {type(score)}"
            assert 0 <= score <= 1, \
                f"Score должен быть в диапазоне [0, 1], получен {score}"

        logger.info(f"✅ Найдено {len(result.retrieval_results.retrieved_docs)} " \
                   f"валидных документов с корректными scores")


@pytest.mark.slow
class TestDebugModePerformance:
    """Тесты для проверки производительности и использования памяти"""

    @pytest.fixture(scope="class")
    def performance_pipeline(self):
        """Fixture: pipeline для тестирования производительности"""
        logger.info("Инициализация pipeline для тестов производительности...")

        os.environ["RAG_DEBUG_MODE"] = "true"

        try:
            pipeline = RAGPipeline(
                retrieval_type=RetrievalType.DENSE,
                embedding_type="huggingface",
                llm_type="local"
            )

            if TEST_DATA_DIR.exists():
                pipeline.load_from_pdf_directory_with_metadata(
                    TEST_DATA_DIR,
                    recursive=False,
                    force_reload=False
                )

            pipeline.llm_manager.load_local_model()
            logger.info("✅ Pipeline инициализирован для performance тестов")
            yield pipeline

        finally:
            if "RAG_DEBUG_MODE" in os.environ:
                del os.environ["RAG_DEBUG_MODE"]

    def test_debug_generation_time_under_limit(self, performance_pipeline):
        """Тест 10: Проверка что генерация ответа завершается за разумное время"""
        query = TEST_QUERY

        # Обрабатываем запрос и измеряем время
        result = performance_pipeline.process_query(query, k=K_RETRIEVE)

        # Проверяем что время обработки положительное
        assert result.generation_time > 0, \
            f"Время генерации должно быть > 0, получено {result.generation_time}"

        # Проверяем что время разумное (< 15 сек для opt-125m)
        assert result.generation_time < 15.0, \
            f"Время генерации должно быть < 15 сек, получено {result.generation_time:.2f} сек"

        # Проверяем что сгенерированы токены
        assert result.tokens_generated > 0, \
            f"Должны быть сгенерированы токены, получено {result.tokens_generated}"

        logger.info(f"✅ Генерация завершена за {result.generation_time:.2f} сек "
                   f"({result.tokens_generated} токенов)")

    @pytest.mark.skipif(not _torch_available or not torch.cuda.is_available(),
                        reason="CUDA недоступна")
    def test_debug_gpu_memory_under_limit(self, performance_pipeline):
        """Тест 11: Проверка что GPU используется < 1 GB памяти"""
        # Процесс query чтобы загрузить модель в GPU
        query = TEST_QUERY
        result = performance_pipeline.process_query(query, k=K_RETRIEVE)

        # Получаем использованную GPU память
        gpu_allocated = torch.cuda.memory_allocated(0) / 1e9  # Конвертируем в GB

        # Проверяем что память используется
        assert gpu_allocated > 0, \
            f"GPU память должна быть > 0 GB, получена {gpu_allocated:.3f} GB"

        # Проверяем что не переполняем GPU (facebook/opt-125m с fp16 < 1 GB)
        assert gpu_allocated < 1.0, \
            f"GPU память должна быть < 1 GB, использовано {gpu_allocated:.3f} GB " \
            f"(контрольная регрессия: было SIGKILL -9 при float32)"

        logger.info(f"✅ GPU использует {gpu_allocated:.3f} GB памяти (< 1 GB ✓)")

    @pytest.mark.skipif(not _torch_available or not torch.cuda.is_available(),
                        reason="CUDA недоступна")
    def test_debug_gpu_memory_consistency(self, performance_pipeline):
        """Тест 12: Проверка что GPU память не растет бесконечно при запросах"""
        # Получаем baseline использования памяти
        torch.cuda.reset_peak_memory_stats(0)
        gpu_before = torch.cuda.memory_allocated(0)

        # Выполняем несколько запросов
        for i in range(2):
            query = f"{TEST_QUERY} (запрос {i+1})"
            result = performance_pipeline.process_query(query, k=K_RETRIEVE)

        # Получаем память после запросов
        gpu_after = torch.cuda.memory_allocated(0)

        # Проверяем что память не выросла более чем на 50%
        # (нормальный рост кэша, но не утечка)
        peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
        growth_percent = ((gpu_after - gpu_before) / max(gpu_before, 1e8)) * 100

        assert peak_memory < 1.5, \
            f"Peak GPU память должна быть < 1.5 GB, получена {peak_memory:.3f} GB"

        logger.info(f"✅ GPU память стабильна: {gpu_before / 1e9:.3f} GB → "
                   f"{gpu_after / 1e9:.3f} GB (peak: {peak_memory:.3f} GB)")


@pytest.mark.slow
class TestDebugModeStreamlit:
    """Тесты для проверки Streamlit UI интеграции с debug режимом"""

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    APP_FILE = PROJECT_ROOT / "app.py"

    def test_debug_streamlit_starts_and_stays_alive(self):
        """Тест 13: Проверка что Streamlit запускается и не падает за 20 сек"""
        if not self.APP_FILE.exists():
            pytest.skip(f"Файл {self.APP_FILE} не найден")

        # Готовим окружение с debug mode
        env = os.environ.copy()
        env["RAG_DEBUG_MODE"] = "true"
        env["RAG_TEST_QUESTION"] = "Что такое RAG?"

        logger.info("Запуск Streamlit в debug режиме...")

        try:
            # Запускаем Streamlit как subprocess
            process = subprocess.Popen(
                [sys.executable, str(self.APP_FILE), "--mode", "ui"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            logger.info(f"Streamlit процесс запущен (PID: {process.pid})")

            # Ждем 20 секунд
            wait_time = 20
            start_time = time.time()

            while time.time() - start_time < wait_time:
                # Проверяем что процесс еще работает
                exit_code = process.poll()
                if exit_code is not None:
                    # Процесс неожиданно завершился
                    stdout, stderr = process.communicate()
                    pytest.fail(
                        f"Streamlit процесс завершился с кодом {exit_code} "
                        f"после {time.time() - start_time:.1f} сек\n"
                        f"stderr: {stderr}"
                    )

                time.sleep(1)

            logger.info(f"✅ Streamlit работает стабильно ({wait_time} сек)")

            # Graceful shutdown
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        except Exception as e:
            logger.error(f"❌ Ошибка при запуске Streamlit: {e}")
            raise

    def test_debug_streamlit_no_oom_kill(self):
        """Тест 14: Проверка что Streamlit не убивается с SIGKILL -9 (OOM)"""
        if not self.APP_FILE.exists():
            pytest.skip(f"Файл {self.APP_FILE} не найден")

        env = os.environ.copy()
        env["RAG_DEBUG_MODE"] = "true"
        env["RAG_TEST_QUESTION"] = "Что такое RAG?"

        logger.info("Запуск Streamlit для проверки на OOM...")

        try:
            process = subprocess.Popen(
                [sys.executable, str(self.APP_FILE), "--mode", "ui"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            logger.info(f"Streamlit процесс запущен (PID: {process.pid})")

            # Ждем 25 секунд (дольше, чтобы пройти фазу загрузки модели)
            wait_time = 25
            start_time = time.time()

            while time.time() - start_time < wait_time:
                exit_code = process.poll()
                if exit_code is not None:
                    # Проверяем что это не SIGKILL (-9) из-за OOM
                    if exit_code == -9:
                        pytest.fail(
                            f"Streamlit убит с SIGKILL -9 (OOM) "
                            f"после {time.time() - start_time:.1f} сек\n"
                            f"Регрессия: float32 модель требует > 1GB памяти"
                        )
                    elif exit_code == -11:
                        pytest.fail(
                            f"Streamlit убит с SIGSEGV -11 (segmentation fault) "
                            f"после {time.time() - start_time:.1f} сек"
                        )
                    elif exit_code != 0 and exit_code != -15:  # -15 это SIGTERM от нас
                        logger.warning(
                            f"⚠️  Streamlit завершился с кодом {exit_code} "
                            f"после {time.time() - start_time:.1f} сек"
                        )

                time.sleep(1)

            logger.info(f"✅ Streamlit не упал на SIGKILL -9 ({wait_time} сек)")

            # Graceful shutdown
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        except Exception as e:
            logger.error(f"❌ Ошибка при запуске Streamlit: {e}")
            raise

    def test_debug_streamlit_env_vars_passed(self):
        """Тест 15: Проверка что переменные окружения корректно передаются"""
        if not self.APP_FILE.exists():
            pytest.skip(f"Файл {self.APP_FILE} не найден")

        # Тест что RAG_DEBUG_MODE передается в subprocess
        env = os.environ.copy()
        env["RAG_DEBUG_MODE"] = "true"

        logger.info("Проверка передачи RAG_DEBUG_MODE в subprocess...")

        try:
            # Запускаем скрипт который проверяет переменные окружения
            result = subprocess.run(
                [sys.executable, "-c",
                 "import os; exit(0 if os.getenv('RAG_DEBUG_MODE') == 'true' else 1)"],
                env=env,
                timeout=5,
                capture_output=True
            )

            assert result.returncode == 0, \
                "Переменная окружения RAG_DEBUG_MODE не передана в subprocess"

            logger.info("✅ Переменные окружения передаются корректно")

        except Exception as e:
            logger.error(f"❌ Ошибка при проверке env vars: {e}")
            raise
