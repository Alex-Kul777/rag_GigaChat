"""
validator.py - Валидация входных данных для RAG системы

Проверяет запросы пользователей, документы, файлы и конфигурацию
перед передачей в пайплайн обработки.

Added: 2026-04-14
Author: Claude
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from rag_gigachat.config import model_config, data_config, gigachat_config

try:
    from langchain_gigachat.chat_models import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False

logger = logging.getLogger(__name__)


# Константы валидации
MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 1000
MAX_QUERY_TOKENS_APPROX = 500   # Лимит токенов GigaChat API
MAX_DOCUMENT_CHARS = 50_000_000
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.json', '.csv'}


@dataclass
class ValidationError:
    """Одна ошибка валидации"""
    field: str
    message: str
    code: str


@dataclass
class ValidationResult:
    """Результат валидации входных данных"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, field: str, message: str, code: str = "INVALID") -> None:
        """Добавить ошибку и пометить результат как невалидный"""
        self.errors.append(ValidationError(field=field, message=message, code=code))
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def error_messages(self) -> List[str]:
        """Список текстовых сообщений об ошибках"""
        return [f"[{e.field}] {e.message}" for e in self.errors]

    def __str__(self) -> str:
        if self.is_valid:
            status = "OK"
            if self.warnings:
                status += f" ({len(self.warnings)} предупреждений)"
        else:
            status = f"ОШИБКА: {'; '.join(self.error_messages)}"
        return f"ValidationResult({status})"


class InputValidator:
    """
    Валидатор входных данных RAG системы.

    Проверяет пользовательские запросы, документы, PDF-файлы
    и параметры конфигурации перед запуском пайплайна.
    """

    def validate_query(self, query: str) -> ValidationResult:
        """
        Валидация пользовательского запроса.

        Args:
            query: Текст запроса от пользователя

        Returns:
            ValidationResult с результатом проверки

        Raises:
            TypeError: Если query не является строкой
        """
        logger.debug(f"🔍 DEBUG: Валидация запроса длиной {len(query) if isinstance(query, str) else '?'}")
        result = ValidationResult(is_valid=True)

        if not isinstance(query, str):
            result.add_error("query", "Запрос должен быть строкой", "TYPE_ERROR")
            logger.error(f"Валидация запроса: неверный тип {type(query)}")
            return result

        stripped = query.strip()

        if not stripped:
            result.add_error("query", "Запрос не может быть пустым", "EMPTY")
            return result

        if len(stripped) < MIN_QUERY_LENGTH:
            result.add_error(
                "query",
                f"Запрос слишком короткий (минимум {MIN_QUERY_LENGTH} символа)",
                "TOO_SHORT"
            )

        if len(stripped) > MAX_QUERY_LENGTH:
            result.add_error(
                "query",
                f"Запрос слишком длинный (максимум {MAX_QUERY_LENGTH} символов, получено {len(stripped)})",
                "TOO_LONG"
            )

        # Приблизительная оценка токенов: ~4 символа на токен для русского текста
        approx_tokens = len(stripped) // 4
        if approx_tokens > MAX_QUERY_TOKENS_APPROX:
            result.add_warning(
                f"Запрос может превысить лимит токенов GigaChat API "
                f"(~{approx_tokens} токенов, лимит ~{MAX_QUERY_TOKENS_APPROX})"
            )
            logger.info(f"⚠️ Предупреждение: запрос содержит ~{approx_tokens} токенов")

        if result.is_valid:
            logger.info(f"✅ Запрос прошёл валидацию ({len(stripped)} символов)")
        else:
            logger.error(f"❌ Запрос не прошёл валидацию: {result.error_messages}")

        return result

    def validate_document(self, text: str, doc_id: str = "") -> ValidationResult:
        """
        Валидация текста документа перед добавлением в базу знаний.

        Args:
            text: Текст документа
            doc_id: Идентификатор документа (для логов)

        Returns:
            ValidationResult с результатом проверки
        """
        label = f"document[{doc_id}]" if doc_id else "document"
        logger.debug(f"🔍 DEBUG: Валидация документа '{doc_id}'")
        result = ValidationResult(is_valid=True)

        if not isinstance(text, str):
            result.add_error(label, "Текст документа должен быть строкой", "TYPE_ERROR")
            return result

        if not text.strip():
            result.add_error(label, "Документ не может быть пустым", "EMPTY")
            return result

        if len(text) > MAX_DOCUMENT_CHARS:
            result.add_error(
                label,
                f"Документ слишком большой ({len(text)} символов, максимум {MAX_DOCUMENT_CHARS})",
                "TOO_LARGE"
            )

        if len(text) < 10:
            result.add_warning(f"Документ очень короткий ({len(text)} символов) — возможно, он неполный")

        if result.is_valid:
            logger.info(f"✅ Документ '{doc_id}' прошёл валидацию ({len(text)} символов)")
        else:
            logger.error(f"❌ Документ '{doc_id}' не прошёл валидацию: {result.error_messages}")

        return result

    def validate_file(self, file_path: str | Path) -> ValidationResult:
        """
        Валидация файла перед загрузкой в систему.

        Args:
            file_path: Путь к файлу

        Returns:
            ValidationResult с результатом проверки
        """
        path = Path(file_path)
        logger.debug(f"🔍 DEBUG: Валидация файла '{path}'")
        result = ValidationResult(is_valid=True)

        if not path.exists():
            result.add_error("file", f"Файл не найден: {path}", "NOT_FOUND")
            return result

        if not path.is_file():
            result.add_error("file", f"Путь не является файлом: {path}", "NOT_A_FILE")
            return result

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            result.add_error(
                "file",
                f"Неподдерживаемый формат '{path.suffix}'. Допустимые: {ALLOWED_EXTENSIONS}",
                "UNSUPPORTED_FORMAT"
            )

        try:
            size = path.stat().st_size
            if size == 0:
                result.add_error("file", f"Файл пустой: {path.name}", "EMPTY_FILE")
            elif size > MAX_FILE_SIZE_BYTES:
                result.add_error(
                    "file",
                    f"Файл слишком большой ({size / 1024 / 1024:.1f} МБ, максимум 10 МБ): {path.name}",
                    "TOO_LARGE"
                )
        except OSError as e:
            result.add_error("file", f"Не удалось проверить размер файла: {e}", "IO_ERROR")
            logger.error(f"Ошибка при проверке файла {path}: {e}")

        if result.is_valid:
            logger.info(f"✅ Файл '{path.name}' прошёл валидацию")
        else:
            logger.error(f"❌ Файл '{path.name}' не прошёл валидацию: {result.error_messages}")

        return result

    def validate_retrieval_params(
        self,
        k: int,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ValidationResult:
        """
        Валидация параметров поиска и генерации.

        Args:
            k: Количество документов для поиска
            temperature: Температура генерации (0.0–2.0)
            max_tokens: Максимальное количество токенов ответа

        Returns:
            ValidationResult с результатом проверки
        """
        logger.debug(f"🔍 DEBUG: Валидация параметров: k={k}, temperature={temperature}, max_tokens={max_tokens}")
        result = ValidationResult(is_valid=True)

        if not isinstance(k, int) or k < 1:
            result.add_error("k", f"k должен быть целым числом >= 1, получено: {k}", "INVALID_K")
        elif k > 50:
            result.add_warning(f"Большое значение k={k} может замедлить поиск")

        if temperature is not None:
            if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 2.0):
                result.add_error(
                    "temperature",
                    f"temperature должна быть от 0.0 до 2.0, получено: {temperature}",
                    "INVALID_TEMPERATURE"
                )

        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens < 1:
                result.add_error(
                    "max_tokens",
                    f"max_tokens должен быть целым числом >= 1, получено: {max_tokens}",
                    "INVALID_MAX_TOKENS"
                )
            elif max_tokens > model_config.max_new_tokens:
                result.add_warning(
                    f"max_tokens={max_tokens} превышает значение из конфига ({model_config.max_new_tokens})"
                )

        if result.is_valid:
            logger.info(f"✅ Параметры поиска прошли валидацию")
        else:
            logger.error(f"❌ Параметры поиска не прошли валидацию: {result.error_messages}")

        return result

    def validate_gigachat_config(self) -> ValidationResult:
        """
        Валидация конфигурации GigaChat API (без сетевых запросов).

        Returns:
            ValidationResult с результатом проверки
        """
        logger.debug("🔍 DEBUG: Валидация конфигурации GigaChat")
        result = ValidationResult(is_valid=True)

        if not gigachat_config.api_key:
            result.add_error(
                "GIGACHAT_API_KEY",
                "API-ключ GigaChat не задан. Установите GIGACHAT_API_KEY в .env",
                "MISSING_API_KEY"
            )

        if gigachat_config.timeout < 1:
            result.add_error(
                "timeout",
                f"Таймаут должен быть >= 1 секунды, получено: {gigachat_config.timeout}",
                "INVALID_TIMEOUT"
            )

        if gigachat_config.max_retries < 0:
            result.add_error(
                "max_retries",
                f"max_retries должен быть >= 0, получено: {gigachat_config.max_retries}",
                "INVALID_RETRIES"
            )

        if not GIGACHAT_AVAILABLE:
            result.add_error(
                "library",
                "langchain-gigachat не установлен. Выполните: pip install langchain-gigachat",
                "MISSING_LIBRARY"
            )

        if result.is_valid:
            logger.info("✅ Конфигурация GigaChat прошла валидацию")
        else:
            logger.error(f"❌ Конфигурация GigaChat не прошла валидацию: {result.error_messages}")

        return result

    def check_gigachat_connection(self, probe_text: str = "Привет") -> ValidationResult:
        """
        Проверка реального подключения к GigaChat API (делает тестовый запрос).

        Вызывай после validate_gigachat_config(), чтобы убедиться, что конфиг валиден
        перед выполнением сетевого запроса.

        Args:
            probe_text: Текст для тестового запроса (короткий, чтобы не тратить токены)

        Returns:
            ValidationResult — is_valid=True, если ответ получен успешно
        """
        logger.info("🌐 Проверка подключения к GigaChat API...")
        result = ValidationResult(is_valid=True)

        # Сначала проверяем конфиг без сети
        config_check = self.validate_gigachat_config()
        if not config_check.is_valid:
            result.errors.extend(config_check.errors)
            result.is_valid = False
            logger.error("❌ Проверка подключения прервана: конфигурация невалидна")
            return result

        try:
            llm = GigaChat(
                credentials=gigachat_config.api_key,
                scope=gigachat_config.scope,
                model=gigachat_config.model,
                verify_ssl_certs=gigachat_config.verify_ssl_certs,
                timeout=gigachat_config.timeout,
            )
            response = llm.invoke(probe_text)

            # Проверяем, что ответ содержит текст
            answer = response.content if hasattr(response, "content") else str(response)
            if not answer.strip():
                result.add_error(
                    "response",
                    "GigaChat вернул пустой ответ",
                    "EMPTY_RESPONSE"
                )
            else:
                logger.info(f"✅ GigaChat отвечает. Ответ на пробный запрос: '{answer[:80].strip()}'")
                result.add_warning(f"Тестовый ответ: '{answer[:80].strip()}'")

        except Exception as e:
            error_str = str(e)
            # Разбираем тип ошибки для более понятного сообщения
            if "401" in error_str or "Unauthorized" in error_str or "credentials" in error_str.lower():
                code, msg = "AUTH_ERROR", f"Неверный API-ключ или нет прав доступа: {e}"
            elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
                code, msg = "TIMEOUT", f"Превышен таймаут подключения ({gigachat_config.timeout}с): {e}"
            elif "connection" in error_str.lower() or "network" in error_str.lower():
                code, msg = "NETWORK_ERROR", f"Ошибка сети — проверьте интернет-соединение: {e}"
            else:
                code, msg = "API_ERROR", f"Ошибка GigaChat API: {e}"

            result.add_error("gigachat", msg, code)
            logger.error(f"❌ Подключение к GigaChat не удалось [{code}]: {e}")

        return result

    def validate_batch(self, queries: List[str]) -> Dict[str, Any]:
        """
        Пакетная валидация списка запросов.

        Args:
            queries: Список запросов

        Returns:
            Словарь с результатами: valid_queries, invalid_indices, summary
        """
        logger.info(f"🔍 Пакетная валидация {len(queries)} запросов")
        valid_queries: List[str] = []
        invalid_indices: List[Dict[str, Any]] = []

        for i, query in enumerate(queries):
            result = self.validate_query(query)
            if result.is_valid:
                valid_queries.append(query)
            else:
                invalid_indices.append({"index": i, "query": query, "errors": result.error_messages})

        summary = {
            "total": len(queries),
            "valid": len(valid_queries),
            "invalid": len(invalid_indices),
            "valid_queries": valid_queries,
            "invalid_details": invalid_indices,
        }
        logger.info(
            f"✅ Пакетная валидация завершена: {summary['valid']}/{summary['total']} запросов валидны"
        )
        return summary


# Глобальный экземпляр валидатора для удобного импорта
validator = InputValidator()


if __name__ == "__main__":
    # Пример использования и базовое тестирование
    v = InputValidator()

    print("=== Тест валидации запросов ===")
    test_queries = [
        "Что такое RAG?",
        "",
        "ab",
        "A" * 1500,
        "   ",
        "Как работает векторный поиск в FAISS с использованием эмбеддингов GigaChat?",
    ]
    for q in test_queries:
        res = v.validate_query(q)
        preview = repr(q[:40]) if q else repr(q)
        print(f"  {preview!r:45} -> {res}")

    print("\n=== Тест валидации параметров ===")
    params = [(5, 0.7, 2000), (0, -1.0, None), (100, 1.5, 500)]
    for k, temp, mt in params:
        res = v.validate_retrieval_params(k, temp, mt)
        print(f"  k={k}, temp={temp}, max_tokens={mt} -> {res}")

    print("\n=== Тест конфигурации GigaChat (без сети) ===")
    res = v.validate_gigachat_config()
    print(f"  {res}")
    if res.warnings:
        for w in res.warnings:
            print(f"  ⚠️  {w}")

    print("\n=== Проверка реального подключения к GigaChat ===")
    conn = v.check_gigachat_connection("Скажи 'ОК' одним словом")
    print(f"  {conn}")
    if conn.warnings:
        for w in conn.warnings:
            print(f"  ℹ️  {w}")

    print("\n=== Пакетная валидация ===")
    batch = ["Первый вопрос?", "", "Второй вопрос?", "x", "Третий нормальный вопрос"]
    summary = v.validate_batch(batch)
    print(f"  Всего: {summary['total']}, валидных: {summary['valid']}, невалидных: {summary['invalid']}")
    if summary["invalid_details"]:
        for item in summary["invalid_details"]:
            print(f"    [#{item['index']}] {item['errors']}")
