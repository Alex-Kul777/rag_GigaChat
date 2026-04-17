"""
debug_context.py - Инструментация для отладки RAG пайплайна
Предоставляет StepTracker и декоратор @trace для сбора метрик выполнения.
"""
import logging
import functools
import time
from typing import Any, Callable, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class StepTracker:
    """Трекер шагов выполнения с метриками времени и ошибок"""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
        self.error = None

    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"🔍 START: {self.step_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

        if exc_type is not None:
            self.error = str(exc_val)
            logger.error(f"❌ ERROR in {self.step_name}: {exc_val} ({self.duration_ms:.1f}ms)")
        else:
            logger.debug(f"✅ END: {self.step_name} ({self.duration_ms:.1f}ms)")

        return False


def trace(func: Callable) -> Callable:
    """Декоратор для трассировки выполнения функции с логированием времени"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        func_name = func.__qualname__
        with StepTracker(func_name):
            return func(*args, **kwargs)
    return wrapper
