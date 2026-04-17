---
id: BKL-NNN
title: "One-line summary of the issue (max 80 chars)"
priority: medium              # high | medium | low
severity: major               # critical | major | minor
status: open                  # open | in-progress | blocked | done | rejected
created: 2026-04-17
updated: 2026-04-17
affected_files:
  - src/rag_gigachat/core/rag_pipeline.py
  - src/rag_gigachat/data/data_loader.py
tags: [retrieval, performance, vector-search]
linked_logs:
linked_events:
process_mining_evidence:
  variant:
  bottleneck_activity:
  anomaly_type:
  affected_cases:
safety_checks:
  - pytest tests/unit/test_rag_pipeline.py
  - pytest tests/unit/test_data_loader.py
rollback: "git reset --hard HEAD"
estimated_effort: 30min       # 15min | 30min | 1h | 2h+
---

## Проблема

Describe the issue:
- What is the symptom?
- When does it occur?
- Impact on users/performance?

Example log excerpt:
```
ERROR in retrieval.vector_search: Connection timeout after 5s
```

## Гипотеза причины

(Optional) Why do you think this happens?
- Is it a timing issue?
- Configuration problem?
- Logic error?

## Предлагаемое исправление

Concrete steps to fix:
1. File: `src/rag_gigachat/core/vector_store.py`
   - Change: Increase timeout from 5s to 10s
   - Reason: FAISS search on large indices needs more time

2. File: `src/rag_gigachat/data/data_loader.py`
   - Change: Add retry logic for transient failures
   - Diff sketch:
   ```python
   for attempt in range(3):
       try:
           results = faiss_index.search(...)
           break
       except TimeoutError:
           if attempt < 2:
               time.sleep(1)
           else:
               raise
   ```

## Критерии приёмки

- [ ] Logs don't contain "Connection timeout" errors
- [ ] `pytest tests/unit/test_rag_pipeline.py` — all green
- [ ] `pytest tests/unit/test_data_loader.py` — all green
- [ ] (Optional) Manual: Load large PDF (50+ pages) with vector search

## Попытки

<!-- Orchestrator fills this section -->
<!-- Example:
- **Attempt 1** (2026-04-17 14:30): ✅ PASS - Fixed in commit abc123d
-->
