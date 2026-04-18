"""
Integration test для app.py CLI - проверка режима query
Запускает app.py как subprocess и проверяет вывод
"""
import pytest
import subprocess
import json
import sys
from pathlib import Path

# Параметры для теста
PROJECT_ROOT = Path(__file__).parent.parent.parent
APP_FILE = PROJECT_ROOT / "app.py"
TEST_DATA_DIR = PROJECT_ROOT / "data/domain_2_Debug/books"
TEST_QUERY = "Что такое RAG и как оно работает?"
OUTPUT_FILE = PROJECT_ROOT / ".test_output.json"


@pytest.fixture
def cleanup_output():
    """Cleanup: удаление файла результатов после теста"""
    yield
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()


class TestAppCLI:
    """Тесты для проверки app.py CLI"""

    def test_app_file_exists(self):
        """Тест 1: Проверка что app.py существует"""
        assert APP_FILE.exists(), f"Файл {APP_FILE} должен существовать"

    def test_documents_path_exists(self):
        """Тест 2: Проверка что директория с документами существует"""
        assert TEST_DATA_DIR.exists(), f"Директория {TEST_DATA_DIR} должна существовать"

    def test_app_query_mode_help(self):
        """Тест 3: Проверка справки app.py"""
        result = subprocess.run(
            [sys.executable, str(APP_FILE), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, "app.py --help должна завершиться с кодом 0"
        assert "query" in result.stdout.lower(), "Справка должна содержать режим query"
        assert "--query" in result.stdout, "Справка должна описывать флаг --query"

    def test_app_query_mode_basic(self, cleanup_output):
        """Тест 4: Запуск app.py в режиме query"""
        result = subprocess.run(
            [
                sys.executable,
                str(APP_FILE),
                "--mode", "query",
                "--query", TEST_QUERY,
                "--documents", str(TEST_DATA_DIR),
                "--k", "3"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )

        assert result.returncode == 0, f"app.py должна завершиться с кодом 0. stderr: {result.stderr}"
        assert "ОТВЕТ" in result.stdout, "Выход должен содержать раздел ОТВЕТ"
        assert "НАЙДЕННЫЕ ДОКУМЕНТЫ" in result.stdout, "Выход должен содержать найденные документы"

    def test_app_query_mode_output_to_json(self, cleanup_output):
        """Тест 5: Сохранение результата в JSON"""
        result = subprocess.run(
            [
                sys.executable,
                str(APP_FILE),
                "--mode", "query",
                "--query", TEST_QUERY,
                "--documents", str(TEST_DATA_DIR),
                "--output", str(OUTPUT_FILE),
                "--k", "3"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )

        assert result.returncode == 0, f"app.py должна завершиться успешно. stderr: {result.stderr}"
        assert OUTPUT_FILE.exists(), f"Файл результата {OUTPUT_FILE} должен быть создан"

        # Проверяем содержимое JSON
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert "query" in data, "JSON должен содержать поле query"
        assert "answer" in data, "JSON должен содержать поле answer"
        assert "retrieved_docs" in data, "JSON должен содержать поле retrieved_docs"
        assert len(data["answer"]) > 0, "Ответ не должен быть пустым"

    def test_app_query_answer_content(self, cleanup_output):
        """Тест 6: Проверка содержимого ответа"""
        result = subprocess.run(
            [
                sys.executable,
                str(APP_FILE),
                "--mode", "query",
                "--query", TEST_QUERY,
                "--documents", str(TEST_DATA_DIR),
                "--output", str(OUTPUT_FILE),
                "--k", "5"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )

        assert result.returncode == 0, f"app.py должна завершиться успешно. stderr: {result.stderr}"

        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Проверяем что ответ адекватен
        answer = data["answer"].lower()
        assert len(answer.split()) > 5, "Ответ должен содержать достаточно слов"

        # Проверяем наличие релевантных ключевых слов
        relevant_words = ["rag", "генер", "документ", "контекст", "поиск"]
        has_relevant = any(word in answer for word in relevant_words)
        assert has_relevant, f"Ответ должен содержать хотя бы одно из слов: {relevant_words}"

    def test_app_query_document_retrieval(self, cleanup_output):
        """Тест 7: Проверка поиска документов"""
        result = subprocess.run(
            [
                sys.executable,
                str(APP_FILE),
                "--mode", "query",
                "--query", TEST_QUERY,
                "--documents", str(TEST_DATA_DIR),
                "--output", str(OUTPUT_FILE),
                "--k", "3"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )

        assert result.returncode == 0

        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Проверяем что найдены документы
        assert len(data["retrieved_docs"]) > 0, "Должны быть найдены документы"
        assert len(data["retrieved_docs"]) <= 3, "Найдено документов <= k"

        # Проверяем структуру документов
        for doc in data["retrieved_docs"]:
            assert "doc_id" in doc, "Документ должен иметь doc_id"
            assert "score" in doc, "Документ должен иметь score"
            assert isinstance(doc["score"], (int, float)), "Score должен быть числом"

    def test_app_query_with_different_k_values(self, cleanup_output):
        """Тест 8: Проверка с разными значениями K"""
        for k_value in [1, 3, 5]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(APP_FILE),
                    "--mode", "query",
                    "--query", TEST_QUERY,
                    "--documents", str(TEST_DATA_DIR),
                    "--output", str(OUTPUT_FILE),
                    "--k", str(k_value)
                ],
                capture_output=True,
                text=True,
                timeout=180
            )

            assert result.returncode == 0, f"Должна работать с k={k_value}"

            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert len(data["retrieved_docs"]) <= k_value, \
                f"Найдено документов должно быть <= {k_value}"

    def test_app_query_error_handling_missing_query(self):
        """Тест 9: Обработка ошибки - отсутствует --query"""
        result = subprocess.run(
            [
                sys.executable,
                str(APP_FILE),
                "--mode", "query",
                "--documents", str(TEST_DATA_DIR)
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Должна быть ошибка или справка
        assert result.returncode != 0 or "query" in result.stderr.lower() or \
               "необходимо указать" in result.stderr.lower(), \
               "Должна быть ошибка при отсутствии --query"

    def test_app_performance(self, cleanup_output):
        """Тест 10: Проверка производительности"""
        result = subprocess.run(
            [
                sys.executable,
                str(APP_FILE),
                "--mode", "query",
                "--query", TEST_QUERY,
                "--documents", str(TEST_DATA_DIR),
                "--output", str(OUTPUT_FILE),
                "--k", "3"
            ],
            capture_output=True,
            text=True,
            timeout=180
        )

        assert result.returncode == 0

        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Проверяем что время обработки разумное
        assert "generation_time" in data, "Должно быть поле generation_time"
        assert data["generation_time"] > 0, "Время должно быть > 0"
        assert data["generation_time"] < 120, "Время должно быть < 120 сек"
