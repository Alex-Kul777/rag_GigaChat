"""
Тесты для event_log.py - Process Mining event logging
"""
import pytest
import tempfile
import csv
from pathlib import Path
from src.rag_gigachat.utils.event_log import (
    emit,
    ProcessEvent,
    CaseContext,
    CANONICAL_ACTIVITIES,
)


@pytest.fixture
def temp_logs_dir(monkeypatch):
    """Временная директория для логов"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        monkeypatch.setattr(CaseContext, "_logs_dir", tmpdir_path)
        CaseContext._csv_file = None
        CaseContext._csv_writer = None
        yield tmpdir_path


def test_emit_valid_activity(temp_logs_dir):
    """Тест: эмит валидной activity записывает событие в CSV"""
    with emit("session.start", resource="streamlit", user="test"):
        pass

    csv_files = list(temp_logs_dir.glob("events_*.csv"))
    assert len(csv_files) == 1

    with open(csv_files[0], "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["activity"] == "session.start"
        assert rows[0]["resource"] == "streamlit"
        assert rows[0]["status"] == "ok"


def test_emit_invalid_activity(temp_logs_dir):
    """Тест: эмит невалидной activity вызывает ValueError"""
    with pytest.raises(ValueError, match="Unknown activity"):
        with emit("invalid.activity", resource="test"):
            pass


def test_emit_with_error(temp_logs_dir):
    """Тест: исключение в блоке с emit помечается status=error"""
    with pytest.raises(RuntimeError):
        with emit("query.receive", resource="pipeline"):
            raise RuntimeError("Test error")

    csv_files = list(temp_logs_dir.glob("events_*.csv"))
    with open(csv_files[0], "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["status"] == "error"


def test_nested_emit_inherits_case_id(temp_logs_dir):
    """Тест: вложенные emit() наследуют case_id"""
    CaseContext.set_case_id("Q-20260417143001-abc123")

    with emit("session.start", resource="streamlit"):
        with emit("query.receive", resource="pipeline", query_len=42):
            pass

    csv_files = list(temp_logs_dir.glob("events_*.csv"))
    with open(csv_files[0], "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["case_id"] == "Q-20260417143001-abc123"
        assert rows[1]["case_id"] == "Q-20260417143001-abc123"


def test_emit_duration_measured(temp_logs_dir):
    """Тест: duration_ms корректно измеряется"""
    import time

    with emit("retrieval.vector_search", resource="faiss"):
        time.sleep(0.01)

    csv_files = list(temp_logs_dir.glob("events_*.csv"))
    with open(csv_files[0], "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        duration_ms = float(rows[0]["duration_ms"])
        assert duration_ms >= 10  # минимум 10ms (sleep 0.01s)


def test_all_canonical_activities_valid():
    """Тест: все활ности из CANONICAL_ACTIVITIES валидны"""
    assert len(CANONICAL_ACTIVITIES) > 0
    assert "session.start" in CANONICAL_ACTIVITIES
    assert "llm.call" in CANONICAL_ACTIVITIES
