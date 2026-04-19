"""
logging_utils.py - Расширенное логирование с информацией о модуле/классе/методе

Обеспечивает структурированное логирование для process mining анализа.
Поддерживает как текстовый формат (консоль) так и JSON (файл).
"""
import logging
import json
import sys
from datetime import datetime
from pathlib import Path


class ContextualFormatter(logging.Formatter):
    """Форматирует логи с информацией о модуле, классе и методе"""

    def format(self, record):
        # Добавляем информацию о модуле/методе
        module_path = record.name.replace('rag_gigachat.', '')
        class_name = getattr(record, 'class_name', '')
        method_name = record.funcName or 'module'

        if class_name:
            location = f"{module_path}.{class_name}.{method_name}"
        else:
            location = f"{module_path}.{method_name}"

        # Базовый формат с информацией о коде
        formatted = (
            f"{self.formatTime(record, self.datefmt)} | "
            f"{record.levelname:8s} | "
            f"{location:60s} | "
            f"{record.getMessage()}"
        )

        return formatted


class JSONFormatter(logging.Formatter):
    """Форматирует логи в JSON для структурированного анализа"""

    def format(self, record):
        module_path = record.name.replace('rag_gigachat.', '')

        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.name,
            "module_short": module_path,
            "function": record.funcName,
            "lineno": record.lineno,
            "class": getattr(record, 'class_name', None),
            "stage": getattr(record, 'stage', None),
            "message": record.getMessage(),
            "metrics": getattr(record, 'metrics', {}),
        }

        return json.dumps(log_obj, ensure_ascii=False)


class DualLogHandler:
    """Настраивает двойное логирование: текст в консоль + JSON в файл"""

    @staticmethod
    def setup_logging(
        log_level: str = "DEBUG",
        console_enabled: bool = True,
        file_enabled: bool = True,
        log_file: str = "logs/rag_app.log",
        json_file: str = "logs/rag_app.json"
    ) -> logging.Logger:
        """
        Настраивает логирование с двумя выводами

        Args:
            log_level: уровень логирования (DEBUG, INFO, WARNING, ERROR)
            console_enabled: вывод в консоль
            file_enabled: вывод в файл
            log_file: путь к текстовому лог-файлу
            json_file: путь к JSON лог-файлу

        Returns:
            Настроенный logger
        """
        logger = logging.getLogger('rag_gigachat')
        logger.setLevel(getattr(logging, log_level))

        # Очистить существующие handlers
        logger.handlers = []

        # Консоль (текстовый формат)
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level))
            console_formatter = ContextualFormatter(
                fmt='%(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Файл (JSON формат для анализа)
        if file_enabled:
            log_path = Path(log_file).parent
            log_path.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, log_level))
            file_formatter = ContextualFormatter(
                fmt='%(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # JSON файл для process mining
        if file_enabled:
            json_path = Path(json_file).parent
            json_path.mkdir(parents=True, exist_ok=True)

            json_handler = logging.FileHandler(json_file, encoding='utf-8')
            json_handler.setLevel(getattr(logging, log_level))
            json_formatter = JSONFormatter()
            json_handler.setFormatter(json_formatter)
            logger.addHandler(json_handler)

        return logger


class LogContext:
    """Context manager для логирования этапов с автоматическими START/END маркерами"""

    def __init__(self, logger: logging.Logger, stage_name: str, metrics: dict = None):
        """
        Args:
            logger: логгер для записи
            stage_name: имя этапа (LOAD_DOCS, RETRIEVAL и т.д.)
            metrics: дополнительные метрики для логирования
        """
        self.logger = logger
        self.stage_name = stage_name
        self.metrics = metrics or {}
        self.start_time = None
        self.caller_info = self._get_caller_info()

    def _get_caller_info(self):
        """Получить информацию о вызывающем коде"""
        import inspect
        frame = inspect.currentframe().f_back.f_back
        return {
            'function': frame.f_code.co_name,
            'lineno': frame.f_lineno,
            'filename': frame.f_code.co_filename.split('/')[-1]
        }

    def __enter__(self):
        import time
        self.start_time = time.time()

        self.logger.info(
            f"🧪 [{self.stage_name} START]",
            extra={
                'stage': self.stage_name,
                'action': 'START',
                'metrics': self.metrics
            }
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed_ms = int((time.time() - self.start_time) * 1000)

        if exc_type is None:
            self.logger.info(
                f"✅ [{self.stage_name} END] duration={elapsed_ms}ms",
                extra={
                    'stage': self.stage_name,
                    'action': 'END',
                    'metrics': {'duration_ms': elapsed_ms, **self.metrics}
                }
            )
        else:
            self.logger.error(
                f"❌ [{self.stage_name} ERROR] {exc_type.__name__}: {exc_val}",
                extra={
                    'stage': self.stage_name,
                    'action': 'ERROR',
                    'metrics': {'duration_ms': elapsed_ms, 'error': str(exc_val)}
                }
            )

        return False  # Re-raise exception


def get_logger(name: str, class_name: str = None) -> logging.Logger:
    """
    Получить логгер с поддержкой class_name

    Args:
        name: имя модуля (__name__)
        class_name: имя класса (если вызов из класса)

    Returns:
        Настроенный логгер

    Пример:
        logger = get_logger(__name__, self.__class__.__name__)
        logger.info("Сообщение", extra={'class_name': self.__class__.__name__})
    """
    logger = logging.getLogger(name)

    # Добавить информацию о классе в следующее сообщение
    if class_name:
        logger = _ClassLoggerAdapter(logger, class_name)

    return logger


class _ClassLoggerAdapter(logging.LoggerAdapter):
    """Адаптер логгера для автоматического добавления информации о классе"""

    def __init__(self, logger, class_name):
        super().__init__(logger, {'class_name': class_name})

    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra']['class_name'] = self.extra['class_name']
        return msg, kwargs
