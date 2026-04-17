---
id: BKL-003
title: "LLM bottleneck: llm.call at 1800ms — optimize to <1000ms"
priority: medium
severity: major
status: open
created: 2026-04-17
updated: 2026-04-17
affected_files:
  - src/rag_gigachat/config.py
  - src/rag_gigachat/core/rag_pipeline.py
  - src/rag_gigachat/utils/token_counter.py
tags: [performance, bottleneck, llm-latency]
linked_logs: logs/session_20260417_230603.log
linked_events: logs/events_20260417_143000.csv
process_mining_evidence:
  bottleneck_activity: "llm.call"
  affected_cases: 2
  p50: "1800ms"
  p95: "1800ms"
  p99: "1800ms"
  max: "1800ms"
  anomaly_type: "bimodal_distribution"
safety_checks:
  - pytest tests/unit/test_llm_manager.py -v
  - pytest tests/performance/test_latency.py::test_llm_call_under_1000ms -v
rollback: "git reset --hard HEAD"
estimated_effort: 2h+
---

## Проблема

**Симптом**: Время обработки `llm.call` находится на уровне 1800ms (p50=p95=p99=max) — без вариативности.

**Process Mining Bottlenecks** (сортировка по p95 DESC):
```
Activity                   | Count | p50   | p95   | p99   | Max
llm.call                   | 2     | 1800ms| 1800ms| 1800ms| 1800ms  ← BOTTLENECK
query.embed                | 2     | 350ms | 350ms | 350ms | 350ms
retrieval.vector_search    | 2     | 100ms | 100ms | 100ms | 100ms
retrieval.rerank           | 2     | 100ms | 100ms | 100ms | 100ms
context.build              | 2     | 100ms | 100ms | 100ms | 100ms
```

**Target**: Снизить `llm.call` с 1800ms до <1000ms через оптимизацию контекста, кэширования и timeout.

## Гипотеза причины

1. **Большой контекст** — все найденные документы передаются в LLM без truncation.
2. **Token recalculation** — каждый вызов пересчитывает количество токенов (нет кэша).
3. **High max_new_tokens** — генерируется слишком много tokens, замедляя LLM.
4. **No timeout for slow requests** — медленные запросы выполняются полностью, без ограничения по времени.

## Предлагаемое исправление

### 1. Добавить truncation контекста перед LLM call (`rag_pipeline.py`)
```python
def generate(state: RAGState):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # OPTIMIZE: Truncate to max_context_length BEFORE formatting
    MAX_CONTEXT_LENGTH = 4000  # chars
    if len(docs_content) > MAX_CONTEXT_LENGTH:
        docs_content = docs_content[:MAX_CONTEXT_LENGTH] + "..."
    
    with emit("llm.call", resource="gigachat", context_len=len(docs_content)):
        formatted_prompt = self.prompt.format_messages(...)
        response = self.llm.invoke(formatted_prompt)
    return {"answer": answer_text}
```

### 2. Реализовать token count caching (`token_counter.py`)
```python
class TokenCounter:
    def __init__(self):
        self._cache = {}  # hash(text) → token_count
    
    def count_text_tokens(self, text: str) -> int:
        """Count tokens with LRU caching"""
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        
        if h not in self._cache:
            # Expensive: hit GigaChat API
            self._cache[h] = self._count_actual(text)
        return self._cache[h]
    
    def _count_actual(self, text: str) -> int:
        """Actual token counting via GigaChat API"""
        # ... existing implementation
```

### 3. Снизить `max_new_tokens` в `config.py`
```python
# BEFORE: max_new_tokens: int = 2000
# AFTER: max_new_tokens: int = 500

# Rationale: Typical Russian answer is 200-300 tokens; 2000 causes 
# unnecessary slowdown in GigaChat generation. Reduce from 1200ms to ~500ms.
```

### 4. Добавить timeout wrapper на `llm.invoke()`
```python
# rag_pipeline.py
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("LLM call exceeded 2.0s timeout")

with emit("llm.call", resource="gigachat"):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(2)  # 2.0s timeout
    try:
        response = self.llm.invoke(formatted_prompt)
    finally:
        signal.alarm(0)  # Cancel alarm
```

## Критерии приёмки

- [ ] Context truncation к max_context_length — реализовано и покрыто тестами
- [ ] Token counter caching — кэшированные lookup в 10x быстрее (测 через `time.perf_counter()`)
- [ ] `max_new_tokens` снижен с 2000 до 500 (+ в конфиге с комментарием)
- [ ] `pytest tests/performance/test_latency.py::test_llm_call_under_1000ms -v` — green
  - p50(llm.call) < 1000ms
  - p95(llm.call) < 1200ms
- [ ] Coverage не упал ниже 54%
- [ ] Event log показывает уменьшение median latency (дельта >30%)

## Попытки

<!-- Orchestrator дописывает сюда при каждой попытке -->
