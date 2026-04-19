"""
test_logging_utils.py - Unit tests for logging_utils module

Tests for:
- ContextualFormatter
- JSONFormatter
- DualLogHandler
- PipelineTimer
- MemoryTracker
- BottleneckAnalyzer
- MetricsExporter
"""
import pytest
import logging
import json
import tempfile
from pathlib import Path
from datetime import datetime
import sys
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.logging_utils import (
    ContextualFormatter,
    JSONFormatter,
    DualLogHandler,
    PipelineTimer,
    MemoryTracker,
    StageMetrics,
    BottleneckAnalyzer,
    MetricsExporter,
)


class TestContextualFormatter:
    """Test ContextualFormatter adds proper context"""

    def test_format_includes_location(self):
        """Test that formatter includes module.class.method"""
        formatter = ContextualFormatter(fmt='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        record = logging.LogRecord(
            name="rag_gigachat.core.rag_pipeline",
            level=logging.INFO,
            pathname="/home/user/src/rag_gigachat/core/rag_pipeline.py",
            lineno=100,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.funcName = "process_query"

        formatted = formatter.format(record)
        assert "rag_pipeline" in formatted
        assert "process_query" in formatted
        assert "Test message" in formatted

    def test_format_timestamp_included(self):
        """Test that formatter includes timestamp"""
        formatter = ContextualFormatter(fmt='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="/test.py",
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_func"
        formatted = formatter.format(record)
        # Should have time and INFO level and location
        assert "INFO" in formatted
        assert "test_func" in formatted


class TestJSONFormatter:
    """Test JSONFormatter produces valid JSON"""

    def test_format_produces_valid_json(self):
        """Test that formatter produces valid JSON lines"""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="rag_gigachat.core.module",
            level=logging.INFO,
            pathname="/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.stage = "RETRIEVAL"

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["lineno"] == 42
        assert parsed["stage"] == "RETRIEVAL"

    def test_json_includes_required_fields(self):
        """Test JSON formatter includes all required fields"""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="rag_gigachat.test",
            level=logging.INFO,
            pathname="/test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert "timestamp" in parsed
        assert "level" in parsed
        assert "module" in parsed
        assert "message" in parsed
        assert parsed["message"] == "Test"


class TestPipelineTimer:
    """Test PipelineTimer for stage timing"""

    def setup_method(self):
        """Create a test logger for each test"""
        self.logger = logging.getLogger("test_pipeline_timer")
        self.logger.handlers = []
        handler = logging.StreamHandler(io.StringIO())
        self.logger.addHandler(handler)

    def test_start_stage_returns_request_id(self):
        """Test that start_stage generates and returns request ID"""
        timer = PipelineTimer(self.logger)
        request_id = timer.start_stage("RETRIEVAL", params={"k": 5})

        assert request_id is not None
        assert len(request_id) >= 8

    def test_end_stage_calculates_duration(self):
        """Test that end_stage calculates duration"""
        import time

        timer = PipelineTimer(self.logger)
        timer.start_stage("GENERATION", params={})

        time.sleep(0.01)  # 10ms sleep
        metrics = timer.end_stage("GENERATION", metrics={"tokens": 10})

        assert metrics is not None
        assert metrics.duration_ms >= 10
        assert metrics.stage_name == "GENERATION"

    def test_get_summary_returns_timings(self):
        """Test summary returns stage timings"""
        timer = PipelineTimer(self.logger)
        timer.start_stage("TEST_STAGE", params={})
        timer.end_stage("TEST_STAGE", metrics={})

        summary = timer.summary()
        assert isinstance(summary, dict)
        assert "TEST_STAGE" in summary
        assert summary["TEST_STAGE"] >= 0


class TestMemoryTracker:
    """Test MemoryTracker for memory usage tracking"""

    def test_start_stage_captures_initial_memory(self):
        """Test that start_stage captures initial memory"""
        tracker = MemoryTracker()
        tracker.start_stage("BEFORE_LOAD")

        assert "BEFORE_LOAD" in tracker.memory_stats

    def test_end_stage_calculates_memory_delta(self):
        """Test that end_stage calculates memory delta"""
        tracker = MemoryTracker()
        tracker.start_stage("LOAD_DOCS")

        # Allocate some memory
        large_list = list(range(100000))

        stats = tracker.end_stage("LOAD_DOCS")

        if tracker.psutil:  # Only assert if psutil is available
            assert "rss_mb" in stats
            assert "rss_delta_mb" in stats
            assert "vms_mb" in stats


class TestStageMetrics:
    """Test StageMetrics dataclass"""

    def test_stage_metrics_initialization(self):
        """Test StageMetrics dataclass initialization"""
        metrics = StageMetrics(
            stage_name="RETRIEVAL",
            timestamp_start="2026-04-19T10:00:00.000Z",
            timestamp_end="2026-04-19T10:00:01.000Z",
            duration_ms=1000,
            status="OK",
            memory_mb=256.5,
            metrics={"k": 5, "docs_count": 3},
        )

        assert metrics.stage_name == "RETRIEVAL"
        assert metrics.duration_ms == 1000
        assert metrics.status == "OK"
        assert metrics.memory_mb == 256.5

    def test_stage_metrics_to_dict(self):
        """Test StageMetrics.to_dict() method"""
        metrics = StageMetrics(
            stage_name="EMBEDDING",
            timestamp_start="2026-04-19T10:00:00.000Z",
            timestamp_end="2026-04-19T10:00:02.000Z",
            duration_ms=2000,
            status="OK",
            metrics={"model": "sentence-transformers/all-MiniLM-L6-v2"},
        )

        result_dict = metrics.to_dict()
        assert result_dict["stage"] == "EMBEDDING"
        assert result_dict["duration_ms"] == 2000
        assert result_dict["status"] == "OK"


class TestBottleneckAnalyzer:
    """Test BottleneckAnalyzer for performance analysis"""

    def test_analyze_identifies_slowest_stage(self):
        """Test that analyzer finds the slowest stage"""
        metrics_list = [
            StageMetrics(
                stage_name="LOAD_DOCS",
                timestamp_start="2026-04-19T10:00:00.000Z",
                timestamp_end="2026-04-19T10:00:01.000Z",
                duration_ms=100,
                status="OK",
            ),
            StageMetrics(
                stage_name="EMBEDDING",
                timestamp_start="2026-04-19T10:00:01.000Z",
                timestamp_end="2026-04-19T10:00:04.000Z",
                duration_ms=3000,  # Longest
                status="OK",
            ),
            StageMetrics(
                stage_name="RETRIEVAL",
                timestamp_start="2026-04-19T10:00:04.000Z",
                timestamp_end="2026-04-19T10:00:04.500Z",
                duration_ms=500,
                status="OK",
            ),
        ]

        analyzer = BottleneckAnalyzer(metrics_list, total_time_ms=3600)
        result = analyzer.analyze()

        assert result["bottleneck_stage"] == "EMBEDDING"
        assert result["bottleneck_duration_ms"] == 3000

    def test_analyzer_provides_recommendations(self):
        """Test that analyzer provides recommendations"""
        metrics_list = [
            StageMetrics(
                stage_name="GENERATION",
                timestamp_start="2026-04-19T10:00:00.000Z",
                timestamp_end="2026-04-19T10:00:10.000Z",
                duration_ms=10000,
                status="OK",
                metrics={"model": "qwen"},
            ),
        ]

        analyzer = BottleneckAnalyzer(metrics_list, total_time_ms=10000)
        result = analyzer.analyze()

        assert "recommendation" in result
        # Recommendation is in Russian, so just verify it's non-empty
        assert len(result["recommendation"]) > 0


class TestMetricsExporter:
    """Test MetricsExporter for exporting metrics"""

    def test_export_to_dataframe(self):
        """Test exporting metrics to DataFrame"""
        metrics_list = [
            StageMetrics(
                stage_name="LOAD_DOCS",
                timestamp_start="2026-04-19T10:00:00.000Z",
                timestamp_end="2026-04-19T10:00:01.000Z",
                duration_ms=1000,
                status="OK",
                metrics={"key": "value", "count": 5},
            ),
        ]

        exporter = MetricsExporter(metrics_list)
        df = exporter.to_dataframe()

        assert not df.empty
        assert "stage" in df.columns
        assert "duration_ms" in df.columns

    def test_export_to_excel(self):
        """Test exporting metrics to Excel file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_list = [
                StageMetrics(
                    stage_name="TEST_STAGE",
                    timestamp_start="2026-04-19T10:00:00.000Z",
                    timestamp_end="2026-04-19T10:00:01.000Z",
                    duration_ms=1000,
                    status="OK",
                ),
            ]

            exporter = MetricsExporter(metrics_list)
            output_file = Path(tmpdir) / "metrics.xlsx"

            exporter.to_excel(str(output_file))

            assert output_file.exists()
            assert output_file.stat().st_size > 0

    def test_summary_stats(self):
        """Test summary statistics calculation"""
        metrics_list = [
            StageMetrics(
                stage_name="A",
                timestamp_start="2026-04-19T10:00:00.000Z",
                duration_ms=1000,
                status="OK",
            ),
            StageMetrics(
                stage_name="B",
                timestamp_start="2026-04-19T10:00:01.000Z",
                duration_ms=2000,
                status="OK",
            ),
        ]

        exporter = MetricsExporter(metrics_list)
        stats = exporter.summary_stats()

        assert stats["total_duration_ms"] == 3000
        assert stats["stages_count"] == 2
        assert stats["avg_stage_time_ms"] == 1500


class TestIntegration:
    """Integration tests for logging system"""

    def test_full_pipeline_logging_flow(self):
        """Test complete logging flow with all components"""
        # Set up logging
        logger = logging.getLogger("test_pipeline")
        logger.handlers = []
        handler = logging.StreamHandler(io.StringIO())
        formatter = ContextualFormatter(fmt='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # Simulate pipeline execution
        timer = PipelineTimer(logger)

        request_id = timer.start_stage("PIPELINE", params={"query": "test"})
        assert request_id is not None

        # Simulate sub-stages
        for stage in ["LOAD", "EMBEDDING", "RETRIEVAL"]:
            timer.start_stage(stage, params={})
            timer.end_stage(stage, metrics={"result": "ok"})

        timer.end_stage("PIPELINE", metrics={})

        # Verify stages were tracked
        summary = timer.summary()
        assert "LOAD" in summary
        assert "EMBEDDING" in summary
        assert "RETRIEVAL" in summary
        assert "PIPELINE" in summary
        assert "TOTAL" in summary

    def test_dual_log_handler_setup(self):
        """Test DualLogHandler sets up logging correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "app.log"
            json_file = Path(tmpdir) / "app.json"

            logger = DualLogHandler.setup_logging(
                log_level="DEBUG",
                log_file=str(log_file),
                json_file=str(json_file)
            )

            logger.info("Test message", extra={"stage": "TEST"})

            assert log_file.exists()
            assert json_file.exists()

            # Check JSON file has valid JSON
            with open(json_file) as f:
                lines = f.readlines()
                if lines:
                    parsed = json.loads(lines[0])
                    assert parsed["message"] == "Test message"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
