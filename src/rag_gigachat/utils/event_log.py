"""
event_log.py - Process Mining event logging и CSV export
Предоставляет ProcessEvent, CaseContext, и emit() для инструментации RAG пайплайна.
"""
import csv
import logging
import time
import contextvars
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import os
import json

logger = logging.getLogger(__name__)

# Канонический словарь activities - защита от drift
CANONICAL_ACTIVITIES = {
    "session.start", "session.end",
    "document.load", "document.ocr", "document.chunk",
    "query.receive", "query.embed", "query.rewrite",
    "retrieval.vector_search", "retrieval.bm25", "retrieval.rerank",
    "context.build", "context.truncate",
    "llm.call", "llm.stream_start", "llm.stream_chunk", "llm.complete",
    "response.render",
    "cache.hit", "cache.miss",
}

# ContextVar для проброса case_id по стеку вызовов
case_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "case_id", default=None
)


@dataclass
class ProcessEvent:
    """Одно событие в event log"""

    case_id: str
    activity: str
    timestamp: str
    resource: str
    duration_ms: float
    status: str
    attributes: str


class CaseContext:
    """Контекст кейса (трассы) с генерацией и управлением case_id"""

    _csv_writer = None
    _csv_file = None
    _logs_dir = Path("logs")

    @classmethod
    def _ensure_logs_dir(cls):
        """Убедиться, что директория логов существует"""
        cls._logs_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def generate_case_id(cls) -> str:
        """Генерирует уникальный case_id формата Q-YYYYMMDDHHMMSS-<6hex>"""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        hex_suffix = uuid.uuid4().hex[:6]
        return f"Q-{timestamp}-{hex_suffix}"

    @classmethod
    def get_or_create_case_id(cls) -> str:
        """Получить текущий case_id или создать новый"""
        current_case_id = case_id_var.get()
        if current_case_id is None:
            current_case_id = cls.generate_case_id()
            case_id_var.set(current_case_id)
        return current_case_id

    @classmethod
    def set_case_id(cls, case_id: str):
        """Установить case_id (для точек входа)"""
        case_id_var.set(case_id)

    @classmethod
    def _get_csv_file(cls):
        """Получить файл CSV для текущей сессии (lazy init)"""
        if cls._csv_file is None:
            cls._ensure_logs_dir()
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            csv_path = cls._logs_dir / f"events_{timestamp}.csv"
            cls._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            cls._csv_writer = csv.DictWriter(
                cls._csv_file,
                fieldnames=[
                    "case_id",
                    "activity",
                    "timestamp",
                    "resource",
                    "duration_ms",
                    "status",
                    "attributes",
                ],
            )
            cls._csv_writer.writeheader()
            logger.info(f"📊 Event log file created: {csv_path}")
        return cls._csv_file, cls._csv_writer

    @classmethod
    def write_event(cls, event: ProcessEvent):
        """Записать событие в CSV"""
        _, writer = cls._get_csv_file()
        writer.writerow(asdict(event))
        cls._csv_file.flush()

    @classmethod
    def close(cls):
        """Закрыть CSV файл"""
        if cls._csv_file is not None:
            cls._csv_file.close()
            cls._csv_file = None
            cls._csv_writer = None


@contextmanager
def emit(
    activity: str,
    resource: str,
    **attributes: Any
):
    """
    Контекстный менеджер для эмита событий в process mining log

    Args:
        activity: имя активности (должно быть в CANONICAL_ACTIVITIES)
        resource: компонент-исполнитель (streamlit, gigachat, faiss, pipeline и т.д.)
        **attributes: произвольные контекстные данные

    Raises:
        ValueError: если activity не в CANONICAL_ACTIVITIES

    Example:
        with emit("retrieval.vector_search", resource="faiss", top_k=5):
            results = index.search(query_vec, k=5)
    """
    if activity not in CANONICAL_ACTIVITIES:
        raise ValueError(
            f"❌ Unknown activity '{activity}'. "
            f"Allowed: {sorted(CANONICAL_ACTIVITIES)}"
        )

    case_id = CaseContext.get_or_create_case_id()
    start_time = time.time()
    start_timestamp = datetime.fromtimestamp(start_time).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3]

    status = "ok"
    error = None

    try:
        yield
    except Exception as e:
        status = "error"
        error = str(e)
        logger.error(f"❌ Error in {activity}: {e}")
        raise
    finally:
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Сериализация attributes
        attrs_json = json.dumps(attributes, default=str)

        event = ProcessEvent(
            case_id=case_id,
            activity=activity,
            timestamp=start_timestamp,
            resource=resource,
            duration_ms=duration_ms,
            status=status,
            attributes=attrs_json,
        )

        CaseContext.write_event(event)
        logger.debug(
            f"📝 Event: {activity} ({resource}) - {duration_ms:.1f}ms - {status}"
        )
