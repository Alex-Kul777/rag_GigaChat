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


from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import uuid


@dataclass
class StageMetrics:
    """Метрики отдельного этапа обработки"""
    stage_name: str
    timestamp_start: str
    timestamp_end: str = ""
    duration_ms: int = 0
    status: str = "PENDING"  # PENDING, OK, ERROR, TIMEOUT
    input_size: int = 0
    output_size: int = 0
    memory_mb: float = 0.0
    error_msg: str = ""
    metrics: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Конвертировать в словарь для логирования"""
        return {
            'stage': self.stage_name,
            'timestamp_start': self.timestamp_start,
            'timestamp_end': self.timestamp_end,
            'duration_ms': self.duration_ms,
            'status': self.status,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'memory_mb': self.memory_mb,
            'error_msg': self.error_msg,
            'custom_metrics': self.metrics,
        }


class PipelineTimer:
    """Таймер для измерения времени выполнения этапов обработки"""

    def __init__(self, logger: logging.Logger):
        """
        Args:
            logger: логгер для записи информации
        """
        self.logger = logger
        self.stages: Dict[str, Dict] = {}
        self.request_id = str(uuid.uuid4())[:8]

    def start_stage(self, stage_name: str, params: dict = None) -> str:
        """
        Начать измерение этапа

        Args:
            stage_name: название этапа (LOAD_DOCS, RETRIEVAL и т.д.)
            params: дополнительные параметры для логирования

        Returns:
            request_id для трассировки
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        self.stages[stage_name] = {
            'start_time': time.time(),
            'timestamp_start': timestamp,
            'params': params or {},
            'status': 'RUNNING'
        }

        # Логируем начало этапа
        self.logger.info(
            f"[{self.request_id}] 🧪 [{stage_name} START]",
            extra={
                'stage': stage_name,
                'action': 'START',
                'request_id': self.request_id,
                'metrics': params or {}
            }
        )

        return self.request_id

    def end_stage(self, stage_name: str, metrics: dict = None, status: str = "OK") -> StageMetrics:
        """
        Завершить измерение этапа

        Args:
            stage_name: название этапа
            metrics: итоговые метрики этапа
            status: статус (OK, ERROR, TIMEOUT)

        Returns:
            StageMetrics объект с информацией об этапе
        """
        if stage_name not in self.stages:
            return None

        stage_info = self.stages[stage_name]
        elapsed_ms = int((time.time() - stage_info['start_time']) * 1000)
        timestamp_end = datetime.utcnow().isoformat() + "Z"

        stage_metrics = StageMetrics(
            stage_name=stage_name,
            timestamp_start=stage_info['timestamp_start'],
            timestamp_end=timestamp_end,
            duration_ms=elapsed_ms,
            status=status,
            metrics={**stage_info['params'], **(metrics or {})}
        )

        # Определяем emoji по статусу
        emoji = {
            'OK': '✅',
            'ERROR': '❌',
            'TIMEOUT': '⏱️',
            'PENDING': '⏳'
        }.get(status, '❓')

        # Логируем завершение этапа
        self.logger.info(
            f"[{self.request_id}] {emoji} [{stage_name} END] duration={elapsed_ms}ms, status={status}",
            extra={
                'stage': stage_name,
                'action': 'END',
                'request_id': self.request_id,
                'metrics': {
                    'duration_ms': elapsed_ms,
                    'status': status,
                    **stage_metrics.metrics
                }
            }
        )

        return stage_metrics

    def get_all_metrics(self) -> List[StageMetrics]:
        """Получить метрики всех завершенных этапов"""
        return [
            stage.get('metrics_obj')
            for stage in self.stages.values()
            if 'metrics_obj' in stage
        ]

    def summary(self) -> Dict[str, int]:
        """
        Получить сводку времени по этапам

        Returns:
            {'STAGE_NAME': duration_ms, ...}
        """
        summary = {}
        total_time = 0

        for stage_name, stage_info in self.stages.items():
            if 'start_time' in stage_info:
                elapsed_ms = int((time.time() - stage_info['start_time']) * 1000)
                summary[stage_name] = elapsed_ms
                total_time += elapsed_ms

        summary['TOTAL'] = total_time
        return summary


class MemoryTracker:
    """Отслеживание использования памяти per-stage"""

    def __init__(self):
        self.memory_stats = {}
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
        except ImportError:
            self.psutil = None
            self.process = None

    def start_stage(self, stage_name: str):
        """Начать отслеживание памяти для этапа"""
        if not self.psutil:
            return

        try:
            self.memory_stats[stage_name] = {
                'start_rss': self.process.memory_info().rss / 1024 / 1024,  # MB
                'start_vms': self.process.memory_info().vms / 1024 / 1024,  # MB
            }
            if hasattr(self.process.memory_info(), 'pfaults'):
                self.memory_stats[stage_name]['start_pfaults'] = self.process.memory_info().pfaults

            try:
                import torch
                if torch.cuda.is_available():
                    self.memory_stats[stage_name]['start_gpu'] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            except:
                pass
        except:
            pass

    def end_stage(self, stage_name: str) -> dict:
        """Завершить отслеживание и получить статистику"""
        if not self.psutil or stage_name not in self.memory_stats:
            return {}

        try:
            end_rss = self.process.memory_info().rss / 1024 / 1024
            end_vms = self.process.memory_info().vms / 1024 / 1024

            stats = self.memory_stats[stage_name]
            result = {
                'rss_mb': end_rss,
                'rss_delta_mb': end_rss - stats['start_rss'],
                'vms_mb': end_vms,
                'vms_delta_mb': end_vms - stats['start_vms'],
            }

            if 'start_gpu' in stats:
                try:
                    import torch
                    if torch.cuda.is_available():
                        end_gpu = torch.cuda.memory_allocated() / 1024 / 1024
                        result['gpu_mb'] = end_gpu
                        result['gpu_delta_mb'] = end_gpu - stats['start_gpu']
                except:
                    pass

            return result
        except:
            return {}


class MetricsExporter:
    """Экспорт метрик в DataFrame и Excel"""

    def __init__(self, metrics_list: List[StageMetrics]):
        self.metrics_list = metrics_list

    def to_dataframe(self):
        """Конвертировать метрики в pandas DataFrame"""
        try:
            import pandas as pd

            data = []
            for m in self.metrics_list:
                row = {
                    'stage': m.stage_name,
                    'timestamp_start': m.timestamp_start,
                    'timestamp_end': m.timestamp_end,
                    'duration_ms': m.duration_ms,
                    'status': m.status,
                    'memory_mb': m.memory_mb,
                }
                row.update(m.metrics)
                data.append(row)

            return pd.DataFrame(data)
        except ImportError:
            raise ImportError("pandas требуется для экспорта в DataFrame. Установите: pip install pandas")

    def to_excel(self, filename: str):
        """Экспортировать метрики в Excel файл"""
        try:
            import pandas as pd

            df = self.to_dataframe()
            df.to_excel(filename, index=False, engine='openpyxl')
            return filename
        except ImportError as e:
            raise ImportError("Требуется pandas и openpyxl. Установите: pip install pandas openpyxl")

    def summary_stats(self) -> dict:
        """Получить сводную статистику"""
        total_time = sum(m.duration_ms for m in self.metrics_list)
        max_memory = max((m.memory_mb for m in self.metrics_list), default=0)

        return {
            'total_duration_ms': total_time,
            'stages_count': len(self.metrics_list),
            'max_memory_mb': max_memory,
            'avg_stage_time_ms': total_time / len(self.metrics_list) if self.metrics_list else 0,
        }


class BottleneckAnalyzer:
    """Анализ узких мест в pipeline"""

    def __init__(self, metrics_list: List[StageMetrics], total_time_ms: int):
        self.metrics_list = metrics_list
        self.total_time_ms = total_time_ms

    def analyze(self) -> dict:
        """Проанализировать и найти bottleneck"""
        if not self.metrics_list:
            return {}

        # Сортируем по времени
        sorted_metrics = sorted(self.metrics_list, key=lambda m: m.duration_ms, reverse=True)

        bottleneck = sorted_metrics[0]
        bottleneck_percent = (bottleneck.duration_ms / self.total_time_ms) * 100 if self.total_time_ms > 0 else 0

        result = {
            'bottleneck_stage': bottleneck.stage_name,
            'bottleneck_duration_ms': bottleneck.duration_ms,
            'bottleneck_percent': bottleneck_percent,
            'recommendation': self._get_recommendation(bottleneck),
            'top_stages': [
                {
                    'stage': m.stage_name,
                    'duration_ms': m.duration_ms,
                    'percent': (m.duration_ms / self.total_time_ms) * 100 if self.total_time_ms > 0 else 0
                }
                for m in sorted_metrics[:3]
            ]
        }

        return result

    def _get_recommendation(self, bottleneck: StageMetrics) -> str:
        """Получить рекомендацию для устранения bottleneck"""
        stage = bottleneck.stage_name

        recommendations = {
            'RETRIEVAL': 'Рассмотрите использование быстрого индекса (IVF) или увеличение nprobe',
            'GENERATION': 'Используйте меньшую модель, уменьшите max_tokens или используйте квантование',
            'CHUNKING': 'Уменьшите chunk_size или используйте параллельную обработку',
            'EMBEDDING': 'Используйте более быструю модель embedding или батчинг',
            'INDEX': 'Используйте более быстрый индекс тип (IVF вместо FLAT)',
            'LOAD_DOCS': 'Используйте кэширование или параллельную загрузку PDF'
        }

        return recommendations.get(stage, 'Оптимизируйте этап обработки')
