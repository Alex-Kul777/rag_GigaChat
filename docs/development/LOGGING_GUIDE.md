# RAG GigaChat Logging System Guide

## Overview

The logging system provides comprehensive process mining capabilities for the RAG pipeline. It tracks:
- **Timing**: Duration of each pipeline stage
- **Memory**: RAM and GPU usage per-stage
- **Context**: Module/class/method information for every log entry
- **Tracing**: Request IDs for full query path visibility
- **Metrics**: Custom metrics per-stage (e.g., token count, document count)

## Quick Start

### Basic Logging Setup

```python
from rag_gigachat.logging_utils import DualLogHandler

# Set up dual logging (console + JSON file)
logger = DualLogHandler.setup_logging(
    log_level="DEBUG",
    log_file="logs/app.log",
    json_file="logs/app.json"
)

logger.info("Application started")
```

### Pipeline Stage Timing

```python
from rag_gigachat.logging_utils import PipelineTimer

timer = PipelineTimer(logger)

# Start a stage
request_id = timer.start_stage("RETRIEVAL", params={"k": 5})

# ... do work ...

# End the stage
metrics = timer.end_stage("RETRIEVAL", metrics={"docs_found": 3})

# Get summary
summary = timer.summary()
print(summary)  # {'RETRIEVAL': 150, 'TOTAL': 150}
```

### Memory Tracking

```python
from rag_gigachat.logging_utils import MemoryTracker

tracker = MemoryTracker()

tracker.start_stage("MODEL_LOAD")
# ... load model ...
mem_stats = tracker.end_stage("MODEL_LOAD")

print(mem_stats)
# {'rss_mb': 2048.5, 'rss_delta_mb': 512.3, 'vms_mb': 3072.1, 'vms_delta_mb': 1024.0}
```

## Components

### ContextualFormatter

Adds module, class, and method information to console logs.

**Output:**
```
2026-04-19 15:05:11 | INFO     | core.rag_pipeline.process_query            | Processing query
```

### JSONFormatter

Formats logs as JSON for process mining analysis.

**Output:**
```json
{
  "timestamp": "2026-04-19T15:05:11.123456Z",
  "level": "INFO",
  "module": "rag_gigachat.core.rag_pipeline",
  "module_short": "core.rag_pipeline",
  "function": "process_query",
  "lineno": 250,
  "class": "RAGPipeline",
  "stage": "RETRIEVAL",
  "message": "Found 3 documents",
  "metrics": {"k": 5, "docs_found": 3}
}
```

### PipelineTimer

Manages stage timing and request ID generation.

**Features:**
- Generates unique request IDs for tracing
- Calculates duration for each stage
- Tracks custom metrics per-stage
- Provides summary of all stage timings

**Example:**
```python
timer = PipelineTimer(logger)
request_id = timer.start_stage("EMBEDDING", params={"model": "e5-small"})
# ... do work ...
metrics = timer.end_stage("EMBEDDING", metrics={"tokens": 1024, "vectors": 32})
summary = timer.summary()
# Returns: {'EMBEDDING': 234, 'TOTAL': 234}
```

### MemoryTracker

Tracks RAM and GPU memory usage per-stage.

**Tracked Metrics:**
- `rss_mb`: Resident Set Size (actual RAM used)
- `rss_delta_mb`: Change in RAM since stage start
- `vms_mb`: Virtual Memory Size
- `gpu_mb`: GPU memory (if CUDA available)
- `gpu_delta_mb`: GPU memory change

### StageMetrics

Dataclass storing metrics for a single pipeline stage.

**Fields:**
- `stage_name`: Name of the stage
- `timestamp_start`: ISO 8601 start timestamp
- `timestamp_end`: ISO 8601 end timestamp
- `duration_ms`: Stage duration in milliseconds
- `status`: OK, ERROR, TIMEOUT, or PENDING
- `memory_mb`: Memory used by this stage
- `metrics`: Custom metrics dictionary

### BottleneckAnalyzer

Identifies performance bottlenecks and provides optimization recommendations.

**Example:**
```python
from rag_gigachat.logging_utils import BottleneckAnalyzer

analyzer = BottleneckAnalyzer(metrics_list, total_time_ms=5000)
result = analyzer.analyze()

print(result)
# {
#   'bottleneck_stage': 'GENERATION',
#   'bottleneck_duration_ms': 3000,
#   'bottleneck_percent': 60.0,
#   'recommendation': 'Используйте меньшую модель...',
#   'top_stages': [...]
# }
```

### MetricsExporter

Exports metrics to DataFrame or Excel for analysis.

**Example:**
```python
from rag_gigachat.logging_utils import MetricsExporter

exporter = MetricsExporter(metrics_list)

# Export to DataFrame
df = exporter.to_dataframe()

# Export to Excel
exporter.to_excel("metrics_report.xlsx")

# Get summary statistics
stats = exporter.summary_stats()
# {'total_duration_ms': 5000, 'stages_count': 5, 'max_memory_mb': 2048.5, ...}
```

## Integration with RAG Pipeline

The logging system is integrated into the RAG pipeline:

```python
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.logging_utils import DualLogHandler

# Set up logging
logger = DualLogHandler.setup_logging()

# Create pipeline (automatically uses configured logger)
pipeline = RAGPipeline()

# Process query - logs all stages automatically
result = pipeline.process_query("What is RAG?")
```

**Automatically Logged Stages:**
- PIPELINE (entire query execution)
- LOAD_DOCS (document loading from PDF)
- CHUNKING (text chunking)
- EMBEDDING (document embedding)
- INDEX (vector index building/loading)
- RETRIEVAL (document retrieval)
- GENERATION (response generation)

## Using Logs for Analysis

### Reading JSON Logs

```python
import json
import pandas as pd

# Read JSON logs
logs = []
with open("logs/app.json") as f:
    for line in f:
        logs.append(json.loads(line))

# Convert to DataFrame for analysis
df = pd.DataFrame(logs)

# Filter by stage
retrieval_logs = df[df['stage'] == 'RETRIEVAL']

# Filter by request ID
request_logs = df[df.get('request_id') == 'req-12345']
```

### Using the Metrics Dashboard

```bash
# Launch Streamlit dashboard to visualize metrics
streamlit run src/rag_gigachat/ui/metrics_dashboard.py
```

The dashboard provides:
- **Timeline View**: Visual timeline of all stages
- **Duration View**: Bar chart comparing stage durations
- **Distribution View**: Pie chart of time distribution
- **Details View**: Filterable table of all logs with Excel export

### Using the Benchmarking System

```python
from rag_gigachat.utils.benchmarking import BenchmarkRun, BenchmarkComparator

# Load benchmark runs
run1 = BenchmarkRun("baseline", "logs/baseline.json")
run2 = BenchmarkRun("optimized", "logs/optimized.json")

# Compare
comparator = BenchmarkComparator([run1, run2])

# Generate report
print(comparator.generate_report())

# Find regressions
regressions = comparator.find_regressions(baseline_run="baseline", threshold_percent=10)
```

## Configuration

### Log Levels

```python
# DEBUG: Detailed debugging information
# INFO: General informational messages
# WARNING: Warning messages
# ERROR: Error messages

logger = DualLogHandler.setup_logging(
    log_level="DEBUG"  # Set to INFO, WARNING, ERROR, or DEBUG
)
```

### Log Files

By default:
- Text logs: `logs/rag_app.log`
- JSON logs: `logs/rag_app.json`

Configure custom paths:

```python
logger = DualLogHandler.setup_logging(
    log_file="custom_logs/app.log",
    json_file="custom_logs/app.json"
)
```

### Disable Components

```python
# Disable console output
logger = DualLogHandler.setup_logging(console_enabled=False)

# Disable file output
logger = DualLogHandler.setup_logging(file_enabled=False)
```

## Best Practices

### 1. Use Request IDs for Tracing

```python
timer = PipelineTimer(logger)
request_id = timer.start_stage("PIPELINE", params={"query": user_query})

# All logs within this request should include request_id
logger.info("Processing step", extra={"request_id": request_id})
```

### 2. Include Relevant Metrics

```python
# Good: Include metrics that help understand performance
timer.end_stage("RETRIEVAL", metrics={"k": 5, "docs_found": 3, "time_ms": 150})

# Bad: Missing useful metrics
timer.end_stage("RETRIEVAL", metrics={})
```

### 3. Catch Errors Properly

```python
try:
    result = pipeline.process_query(query)
    timer.end_stage("GENERATION", metrics=result, status="OK")
except Exception as e:
    timer.end_stage("GENERATION", metrics={"error": str(e)}, status="ERROR")
    raise
```

### 4. Regular Log Cleanup

```python
# Archive old logs
import gzip
import shutil

with open('logs/app.json', 'rb') as f_in:
    with gzip.open('logs/app.json.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
```

## Troubleshooting

### JSON Logs Not Created

```python
# Verify directory exists
from pathlib import Path
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Re-initialize logging
logger = DualLogHandler.setup_logging()
```

### Memory Tracking Not Working

Ensure `psutil` is installed:

```bash
pip install psutil
```

### GPU Memory Not Tracked

Ensure `torch` is installed:

```bash
pip install torch
```

## Performance Impact

The logging system has minimal performance impact:
- Logging adds ~1-2ms per stage (mostly I/O)
- Memory tracking adds negligible overhead (uses psutil)
- JSON formatting is lazy (only on write)

For production with high throughput, consider:
- Setting log level to WARNING (reduce verbosity)
- Rotating log files (use `RotatingFileHandler`)
- Async logging (use QueueHandler)

## Examples

### Complete Pipeline Logging Example

```python
from rag_gigachat.logging_utils import (
    DualLogHandler, PipelineTimer, MemoryTracker,
    MetricsExporter, BottleneckAnalyzer
)

# Set up logging
logger = DualLogHandler.setup_logging(log_level="DEBUG")
timer = PipelineTimer(logger)
memory_tracker = MemoryTracker()

# Simulate pipeline
stages = ["LOAD_DOCS", "EMBEDDING", "RETRIEVAL", "GENERATION"]

for stage in stages:
    request_id = timer.start_stage(stage, params={})
    memory_tracker.start_stage(stage)
    
    # Do work...
    import time
    time.sleep(0.1)
    
    mem_stats = memory_tracker.end_stage(stage)
    timer.end_stage(stage, metrics=mem_stats)

# Analyze results
summary = timer.summary()
print("Stage Timings:", summary)

# Export metrics
metrics_list = [...]  # Get from pipeline
exporter = MetricsExporter(metrics_list)
df = exporter.to_dataframe()
print("\nMetrics DataFrame:")
print(df)

# Find bottleneck
analyzer = BottleneckAnalyzer(metrics_list, total_time_ms=summary["TOTAL"])
bottleneck = analyzer.analyze()
print("\nBottleneck:", bottleneck["bottleneck_stage"])
print("Recommendation:", bottleneck["recommendation"])
```

## See Also

- [Metrics Dashboard Guide](METRICS_DASHBOARD.md) - Using the Streamlit dashboard
- [Benchmarking Guide](BENCHMARKING.md) - Comparing multiple runs
- [Debug Mode Guide](DEBUG_MODE_ACTIVATION.md) - Running in debug mode with fast model
