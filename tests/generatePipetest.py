#!/usr/bin/env python3
"""
tests/generatePipetest.py - Скрипт для автоматического создания и заполнения тестовой инфраструктуры
Запуск: python tests/generatePipetest.py
"""

import os
import sys
from pathlib import Path

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent
# sys.path.insert больше не нужен - пакет в sys.path автоматически

def create_directory(path):
    """Создание директории если не существует"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"  📁 Создана директория: {path}")

def write_file(filepath, content):
    """Запись содержимого в файл"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  📄 Создан файл: {filepath}")

def main():
    print("=" * 60)
    print("🔧 Генерация тестовой инфраструктуры для RAG GigaChat")
    print("=" * 60)
    
    # 1. Создание структуры директорий
    print("\n📁 Создание структуры директорий...")
    create_directory(PROJECT_ROOT / "tests")
    create_directory(PROJECT_ROOT / "tests/fixtures")
    create_directory(PROJECT_ROOT / "tests/integration")
    create_directory(PROJECT_ROOT / "tests/unit")
    create_directory(PROJECT_ROOT / "htmlcov")
    
    # 2. Создание pytest.ini
    print("\n📝 Создание конфигурационных файлов...")
    
    pytest_ini = """[pytest]
minversion = 7.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=rag_core
    --cov=token_counter
    --cov=evaluator
    --cov=config
    --cov=excel_reporter
    --cov-report=html
    --cov-report=term-missing
    --cov-report=xml
    --cov-fail-under=60
markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (requires API)
    slow: Slow running tests
    asyncio: Asynchronous tests
    mock: Tests with mocks
    smoke: Smoke tests for quick validation
timeout = 300
asyncio_mode = auto
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning
"""
    write_file(PROJECT_ROOT / "pytest.ini", pytest_ini)
    
    # 3. Создание .coveragerc
    coveragerc = """[run]
source = 
    rag_core
    token_counter
    evaluator
    config
    excel_reporter
omit = 
    tests/*
    .venv/*
    */site-packages/*
    */__pycache__/*
    */test_*
    */conftest.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstractmethod
    pass
show_missing = True
precision = 2
fail_under = 60

[html]
directory = htmlcov
"""
    write_file(PROJECT_ROOT / ".coveragerc", coveragerc)
    
    # 4. Создание conftest.py
    conftest = '''"""
tests/conftest.py - Фикстуры для pytest
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Путь к тестовым данным
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_documents():
    """Фикстура с тестовыми документами"""
    return {
        "doc_1": "Нейросети - это математические модели, вдохновленные структурой человеческого мозга. Они состоят из нейронов, которые обрабатывают и передают сигналы.",
        "doc_2": "Машинное обучение - это подполе искусственного интеллекта, которое фокусируется на разработке алгоритмов, способных обучаться на данных.",
        "doc_3": "RAG (Retrieval-Augmented Generation) - это метод, который комбинирует поиск информации с генерацией текста.",
        "doc_4": "FAISS (Facebook AI Similarity Search) - библиотека для эффективного поиска похожих векторов.",
    }


@pytest.fixture
def sample_queries():
    """Фикстура с тестовыми запросами"""
    return [
        {"query": "Что такое нейросети?", "expected_keyword": "нейросети"},
        {"query": "Что такое машинное обучение?", "expected_keyword": "машинное обучение"},
        {"query": "Что такое RAG?", "expected_keyword": "RAG"},
        {"query": "Для чего используется FAISS?", "expected_keyword": "FAISS"},
    ]


@pytest.fixture
def sample_testset():
    """Фикстура с тестовым набором для экспериментов"""
    return {
        "q1": {
            "query": "Что такое нейросети?",
            "relevant_docs": ["doc_1"],
            "reference_answer": "Нейросети - это математические модели."
        },
        "q2": {
            "query": "Что такое RAG?",
            "relevant_docs": ["doc_3"],
            "reference_answer": "RAG комбинирует поиск и генерацию."
        }
    }


@pytest.fixture
def mock_gigachat():
    """Мок для GigaChat API"""
    with patch('langchain_gigachat.chat_models.GigaChat') as mock:
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.content = "Это тестовый ответ от мока GigaChat"
        mock_instance.invoke.return_value = mock_response
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_embeddings():
    """Мок для эмбеддингов"""
    with patch('langchain_gigachat.embeddings.GigaChatEmbeddings') as mock:
        mock_instance = Mock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3] for _ in range(3)]
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Временная директория для кэша"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_vectorstore_dir(tmp_path):
    """Временная директория для векторного хранилища"""
    vectorstore_dir = tmp_path / "vectorstore"
    vectorstore_dir.mkdir()
    return vectorstore_dir
'''
    write_file(PROJECT_ROOT / "tests/conftest.py", conftest)
    
    # 5. Создание тестовых файлов
    print("\n📝 Создание тестов...")
    
    # 5.1 test_token_counter.py
    test_token_counter = '''"""
tests/test_token_counter.py - Тесты для счетчика токенов
"""

import pytest
from rag_gigachat.token_counter import TokenCounter


class TestTokenCounter:
    """Тесты для TokenCounter"""
    
    def test_initialization(self):
        """Тест инициализации счетчика"""
        counter = TokenCounter()
        assert counter.total_tokens == 0
        assert counter.num_requests == 0
        assert counter.prompt_tokens == 0
        assert counter.completion_tokens == 0
    
    def test_add_request(self):
        """Тест добавления запроса"""
        counter = TokenCounter()
        tokens = counter.add_request("Привет мир", "Ответ мир")
        assert tokens > 0
        assert counter.num_requests == 1
        assert counter.total_tokens > 0
    
    def test_count_text_tokens(self):
        """Тест подсчета токенов в тексте"""
        counter = TokenCounter()
        text = "Это тестовый текст для подсчета токенов"
        tokens = counter.count_text_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_get_stats(self):
        """Тест получения статистики"""
        counter = TokenCounter()
        counter.add_request("q1", "a1")
        counter.add_request("q2", "a2")
        
        stats = counter.get_stats()
        assert 'num_requests' in stats
        assert 'total_tokens' in stats
        assert stats['num_requests'] == 2
    
    def test_reset(self):
        """Тест сброса счетчика"""
        counter = TokenCounter()
        counter.add_request("test", "response")
        assert counter.num_requests == 1
        
        counter.reset()
        assert counter.num_requests == 0
        assert counter.total_tokens == 0
    
    def test_estimate_cost(self):
        """Тест оценки стоимости"""
        counter = TokenCounter()
        counter.add_request("test", "response" * 100)
        cost = counter.estimate_cost()
        assert cost >= 0
        assert isinstance(cost, float)
'''
    write_file(PROJECT_ROOT / "tests/test_token_counter.py", test_token_counter)
    
    # 5.2 test_config.py
    test_config = '''"""
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
'''
    write_file(PROJECT_ROOT / "tests/test_config.py", test_config)
    
    # 5.3 test_rag_core.py
    test_rag_core = '''"""
tests/test_rag_core.py - Тесты для RAG пайплайна
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_gigachat.core.rag_pipeline import RAGPipeline, VectorStoreManager, LLMManager


class TestRAGPipeline:
    """Тесты для RAGPipeline"""
    
    def test_initialization(self):
        """Тест инициализации RAGPipeline"""
        pipeline = RAGPipeline()
        assert pipeline is not None
        assert pipeline.vector_store_initialized is False
    
    def test_load_documents_from_dict(self, sample_documents):
        """Тест загрузки документов из словаря"""
        pipeline = RAGPipeline()
        pipeline.load_documents_from_dict(sample_documents)
        assert pipeline.vector_store_initialized is True
    
    def test_get_stats(self, sample_documents):
        """Тест получения статистики"""
        pipeline = RAGPipeline()
        pipeline.load_documents_from_dict(sample_documents)
        
        stats = pipeline.get_stats()
        assert 'vector_store_initialized' in stats
        assert stats['vector_store_initialized'] is True
        assert 'chunk_size' in stats
        assert 'chunk_overlap' in stats
    
    @pytest.mark.mock
    def test_process_query_returns_result(self, sample_documents, mock_gigachat):
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
'''
    write_file(PROJECT_ROOT / "tests/test_rag_core.py", test_rag_core)
    
    # 5.4 test_smoke.py
    test_smoke = '''"""
tests/test_smoke.py - Дымовые тесты для быстрой проверки
"""

import pytest


class TestSmoke:
    """Дымовые тесты"""
    
    def test_import_all_modules(self):
        """Проверка импорта всех основных модулей"""
        from rag_core import RAGPipeline
        from rag_gigachat.token_counter import TokenCounter
        from rag_gigachat.config import model_config, data_config
        from evaluator import WikiEvalEvaluator
        from excel_reporter import ExcelReporter
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
        from rag_core import RAGPipeline
        pipeline = RAGPipeline()
        assert pipeline is not None
        assert pipeline.vector_store_initialized is False
'''
    write_file(PROJECT_ROOT / "tests/test_smoke.py", test_smoke)
    
    # 6. Создание fixtures
    print("\n📝 Создание тестовых данных (fixtures)...")
    
    sample_docs_json = '''{
    "doc_1": "Нейросети - это математические модели, вдохновленные структурой человеческого мозга.",
    "doc_2": "Машинное обучение - это подполе искусственного интеллекта.",
    "doc_3": "RAG (Retrieval-Augmented Generation) - это метод улучшения LLM.",
    "doc_4": "FAISS - библиотека для эффективного поиска похожих векторов."
}'''
    write_file(PROJECT_ROOT / "tests/fixtures/sample_docs.json", sample_docs_json)
    
    sample_queries_json = '''[
    {"query": "Что такое нейросети?", "expected": "нейросети"},
    {"query": "Что такое машинное обучение?", "expected": "машинное обучение"},
    {"query": "Что такое RAG?", "expected": "RAG"}
]'''
    write_file(PROJECT_ROOT / "tests/fixtures/sample_queries.json", sample_queries_json)
    
    # 7. Создание Makefile для тестов
    makefile = '''# Makefile для управления тестами
.PHONY: test test-cov test-unit test-integration test-smoke clean help

help:
	@echo "Доступные команды:"
	@echo "  make test         - Запустить все тесты"
	@echo "  make test-cov     - Запустить тесты с coverage"
	@echo "  make test-unit    - Запустить только unit-тесты"
	@echo "  make test-smoke   - Запустить дымовые тесты"
	@echo "  make clean        - Очистить временные файлы"

test:
	pytest -v

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report: open htmlcov/index.html"

test-unit:
	pytest -m "unit" -v

test-smoke:
	pytest -m "smoke" -v

test-integration:
	pytest -m "integration" -v

clean:
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
'''
    write_file(PROJECT_ROOT / "Makefile", makefile)
    
    # 8. Создание .gitignore дополнений для тестов
    gitignore_content = '''
# Test artifacts
htmlcov/
.coverage
.coverage.*
coverage.xml
.pytest_cache/
.tox/
.mypy_cache/
*.cover
test_results/
'''
    gitignore_path = PROJECT_ROOT / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, 'a') as f:
            f.write(gitignore_content)
        print(f"  📝 Дополнен .gitignore")
    else:
        write_file(gitignore_path, "# Python\n__pycache__/\n.venv/\n" + gitignore_content)
    
    # 9. Вывод инструкции
    print("\n" + "=" * 60)
    print("✅ Генерация тестовой инфраструктуры завершена!")
    print("=" * 60)
    print("\n📋 Что создано:")
    print("  ├── tests/")
    print("  │   ├── conftest.py       - Фикстуры для pytest")
    print("  │   ├── test_token_counter.py")
    print("  │   ├── test_config.py")
    print("  │   ├── test_rag_core.py")
    print("  │   ├── test_smoke.py")
    print("  │   └── fixtures/")
    print("  │       ├── sample_docs.json")
    print("  │       └── sample_queries.json")
    print("  ├── pytest.ini            - Конфигурация pytest")
    print("  ├── .coveragerc           - Конфигурация coverage")
    print("  └── Makefile              - Управление тестами")
    
    print("\n🚀 Запуск тестов:")
    print("  source .venv/bin/activate")
    print("  pytest -v                 # Запустить все тесты")
    print("  make test                 # Использовать Makefile")
    print("  make test-cov             # Запустить с coverage")
    
    print("\n📊 Установка дополнительных пакетов (если нужно):")
    print("  pip install pytest pytest-cov pytest-mock")
    
    print("\n✨ Готово!")

if __name__ == "__main__":
    main()