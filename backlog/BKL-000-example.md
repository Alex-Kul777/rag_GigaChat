---
id: BKL-000
title: "Example: OOM on large PDF processing with OCR enabled"
priority: high
severity: major
status: open
created: 2026-04-15
updated: 2026-04-17
affected_files:
  - src/rag_gigachat/data/data_loader.py
  - src/rag_gigachat/config.py
tags: [ocr, memory, pdf-loading, production]
linked_logs: logs/session_20260415_1430.log
linked_events: logs/events_20260415_1430.csv
process_mining_evidence:
  variant: "V2 (rare)"
  bottleneck_activity: "document.ocr"
  anomaly_type: "high_error_rate"
  affected_cases: 2
safety_checks:
  - pytest tests/unit/test_data_loader.py -k ocr
  - pytest tests/unit/test_data_loader.py -k large_pdf
rollback: "git reset --hard HEAD"
estimated_effort: 1h
---

## Проблема

**Symptom**: When loading large PDFs (>100 pages) with OCR enabled, the application crashes with OOM (Out of Memory) error.

```
CRITICAL in document.ocr: RuntimeError: CUDA out of memory. 
Tried to allocate 2.5 GB on CUDA:0 (12.0 GB total)
```

**When**: During PDF upload via Streamlit UI with `ocr_enabled=true` in config

**Impact**: Users cannot process large, scanned documents → Critical feature broken for document-heavy workflows

**Example**: Loading `data/samples/100page_scan.pdf` (45MB) causes immediate crash

## Гипотеза причины

1. OCR pipeline loads entire PDF into CUDA memory at once
2. No batching or page-level streaming
3. Config allows PDFs up to 50MB (`pdf_max_doc_size`) but OCR not validated
4. `ocr_max_file_size_mb=50` in config allows too-large files to proceed

Current flow:
```
PDF load → Docling OCR (full file into GPU) → BOOM
```

Better approach:
```
PDF load → Batch pages (N at a time) → OCR batch → Stream results
```

## Предлагаемое исправление

### 1. File: `src/rag_gigachat/config.py`
- Reduce `ocr_max_file_size_mb` from 50 to 30
- Add `ocr_batch_size: int = 10` (process 10 pages at a time)
- Add `ocr_device: str = "cpu"` (fallback to CPU for large files)

```python
@dataclass
class DataConfig:
    # ... existing ...
    ocr_enabled: bool = True
    ocr_max_file_size_mb: int = 30        # ← changed from 50
    ocr_batch_size: int = 10              # ← NEW
    ocr_device: str = "cpu"               # ← NEW
    ocr_min_chars_per_page: int = 50
```

### 2. File: `src/rag_gigachat/data/data_loader.py`
- Implement `_batch_ocr_pages()` method to process pages in batches
- Add fallback logic: if OCR with CUDA fails, retry with CPU

```python
def _batch_ocr_pages(self, pages: List[bytes], batch_size: int = 10) -> List[str]:
    """Process OCR in batches to avoid CUDA OOM"""
    results = []
    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]
        try:
            batch_results = self.ocr_pipeline.process_batch(
                batch, device="cuda"
            )
            results.extend(batch_results)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"CUDA OOM, falling back to CPU")
                batch_results = self.ocr_pipeline.process_batch(
                    batch, device="cpu"
                )
                results.extend(batch_results)
            else:
                raise
    return results
```

### 3. File: `src/rag_gigachat/data/data_loader.py` (validation)
- Add pre-flight check: skip OCR if file size > `ocr_max_file_size_mb`

```python
if pdf_size_mb > data_config.ocr_max_file_size_mb:
    logger.warning(
        f"PDF {pdf_size_mb}MB exceeds OCR limit "
        f"({data_config.ocr_max_file_size_mb}MB), skipping OCR"
    )
    ocr_enabled = False
```

## Критерии приёмки

- [ ] Load `data/samples/100page_scan.pdf` without CUDA OOM error
- [ ] `pytest tests/unit/test_data_loader.py -k ocr` — all green
- [ ] `pytest tests/unit/test_data_loader.py -k large_pdf` — all green
- [ ] Manual: Test with 50-page PDF (OCR enabled) → should complete in <30s on GPU
- [ ] Manual: Test with 150-page PDF (>50MB) → should gracefully skip OCR with warning

## Попытки

<!-- Orchestrator fills this -->
