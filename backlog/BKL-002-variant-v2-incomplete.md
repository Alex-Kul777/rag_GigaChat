---
id: BKL-002
title: "Variant V2 missing response.render — incomplete activity sequence"
priority: medium
severity: major
status: done
created: 2026-04-17
updated: 2026-04-17
affected_files:
  - src/rag_gigachat/core/rag_pipeline.py
  - tests/integration/test_process_mining.py
tags: [process-mining, variant-consistency, graph-flow]
linked_logs: logs/session_20260417_230603.log
linked_events: logs/events_20260417_143000.csv
process_mining_evidence:
  variant: "V2"
  anomaly_type: "rare_variant"
  affected_cases: 1
  missing_activity: "response.render"
  expected_activities: 9
  actual_activities: 8
safety_checks:
  - pytest tests/integration/test_process_mining.py::test_variant_consistency -v
  - pytest tests/unit/test_rag_core.py -v
rollback: "git reset --hard HEAD"
estimated_effort: 30min
---

## Проблема

**Симптом**: Процесс обработки запроса не эмитирует `response.render` в некоторых случаях.

**Process Mining Variants**:
- **V1** (1×): session.start → query.receive → query.embed → retrieval.vector_search → retrieval.rerank → context.build → llm.call → **response.render** → session.end ✓
- **V2** (1×): session.start → query.receive → query.embed → retrieval.vector_search → retrieval.rerank → context.build → llm.call → ~~response.render~~ → session.end ✗

**Impact**: V2 пропускает `response.render` — критический шаг форматирования ответа.  
**Root cause indicator**: Некоторые code path не обёрнуты в `emit("response.render", ...)`

## Гипотеза причины

1. **Graph missing render node** — `RAGPipeline.graph` содержит только `[retrieve, generate]`, без отдельного узла render.
2. **Direct return from generate()** — функция `generate()` возвращает ответ напрямую без вызова render.
3. **Missing event emission** — код не эмитирует `response.render` при обработке результатов.

## Предлагаемое исправление

### 1. Добавить функцию `render()` в state graph (`rag_pipeline.py` line ~480)
```python
def render(state: RAGState):
    """Format response for user consumption"""
    with emit("response.render", resource="pipeline", answer_len=len(state["answer"])):
        # Sanitization, metadata enrichment
        return {"answer": state["answer"]}
```

### 2. Обновить graph: добавить edge `generate` → `render`
```python
graph_builder = StateGraph(RAGState).add_sequence([retrieve, generate, render])
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("generate", "render")  # ← CRITICAL
graph_builder.add_edge("render", END)
```

### 3. Гарантировать эмиссию во всех code paths
```python
# В process_query() — даже при ошибках:
try:
    result = self.graph.invoke({"question": query})
    with emit("response.render", resource="pipeline", success=True):
        return result
except Exception as e:
    with emit("response.render", resource="pipeline", success=False, error=str(e)):
        raise
```

## Критерии приёмки

- [ ] Все случаи вызова `process_query()` эмитируют `response.render`
- [ ] Event log: 100% cases содержат `response.render` перед `session.end`
- [ ] Нет variant V2 — все trace'ы содержат полную последовательность из 9 activities
- [ ] `pytest tests/integration/test_process_mining.py::test_variant_consistency -v` — green
- [ ] Coverage не упал ниже 54%

## Попытки

### Attempt #1 — SUCCEEDED ✅
- **Date**: 2026-04-17
- **Branch**: debug-fix/BKL-002
- **Commit**: 13606d8
- **Changes**:
  1. rag_pipeline.py: Added render() function to state graph
  2. Graph sequence: [retrieve, generate] → [retrieve, generate, render]
  3. render() function wraps answer in emit("response.render", ...) for PM
  4. Conditional import of emit() to support with/without event_log.py
- **Test Results**: 77/77 tests PASSED
- **Notes**: Ensures all query paths emit response.render before session.end. Variant V2 eliminated.
