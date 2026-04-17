---
id: BKL-001
title: "LLM timeout in llm.call — 50% error rate with GigaChat"
priority: high
severity: critical
status: in-progress
created: 2026-04-17
updated: 2026-04-17
affected_files:
  - src/rag_gigachat/core/llm_manager.py
  - src/rag_gigachat/config.py
  - src/rag_gigachat/core/rag_pipeline.py
tags: [llm, timeout, reliability, gigachat]
linked_logs: logs/session_20260417_230603.log
linked_events: logs/events_20260417_143000.csv
process_mining_evidence:
  variant: "Q-20260417143002-def"
  bottleneck_activity: "llm.call"
  anomaly_type: "high_error_rate"
  affected_cases: 1
  error_rate: "50% (1/2 calls)"
safety_checks:
  - pytest tests/unit/test_llm_manager.py -v
  - pytest tests/unit/test_rag_core.py::test_query_with_timeout -v
rollback: "git reset --hard HEAD"
estimated_effort: 1h
---

## Проблема

**Симптом**: LLM timeout при обработке запроса длиной 50 символов.  
**Case ID**: Q-20260417143002-def  
**Timestamp**: 2026-04-17T14:30:12.500  
**Duration**: 1800ms (вместо ожидаемых 1000ms)  
**Error**: `{"error":"timeout"}`  

**Выдержка из event log**:
```
Q-20260417143002-def,llm.call,2026-04-17T14:30:12.500,gigachat,1800,error,"{""error"":""timeout""}"
```

**Успешный запрос для сравнения** (Q-20260417143001-abc):
- Query length: 42 символа → llm.call: 1000ms ✓
- Error case: 50 символов → llm.call: 1800ms ✗

**Доля ошибок**: 50% (1 ошибка из 2 вызовов).

## Гипотеза причины

1. **Timeout threshold слишком низкий** — установлен на ~1000-1200ms, но медленнее запросы требуют 1800ms.
2. **Нет retry logic с exponential backoff** — при первом таймауте отказываем пользователю, вместо повтора.
3. **No token budget enforcement** — система не предсказывает, когда запрос будет слишком долгим для timeout.
4. **Incomplete error handling** — при timeout, `response.render` не вызывается → пользователь видит silent failure.

## Предлагаемое исправление

### 1. Добавить model-aware timeout в `config.py`
```python
# Вместо жёсткого timeout=60s
llm_timeout_seconds: int = 90  # +50% буфер для GigaChat
llm_timeout_multiplier: float = 1.5  # Для медленных моделей
```

### 2. Реализовать retry wrapper с exponential backoff в `llm_manager.py`
```python
def _invoke_with_retry(self, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            with emit("llm.call", resource="gigachat", attempt=attempt+1):
                return self.llm.invoke(prompt)
        except TimeoutError:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1.0 + random(0, 1)
                time.sleep(wait_time)
            else:
                raise
```

### 3. Обернуть `llm.invoke()` в `rag_pipeline.py` с event logging
```python
with emit("llm.call", resource="gigachat", query_len=len(query), tokens=token_count):
    response = self.llm.invoke(formatted_prompt)
```

### 4. Гарантировать `response.render` даже при ошибке (см. BKL-002)

## Критерии приёмки

- [ ] `llm.call` с retry — максимум 3 попытки с exponential backoff (1s → 2s → 4s)
- [ ] Event log содержит все `llm.call` события (включая failed + retry)
- [ ] `pytest tests/unit/test_llm_manager.py -v` — green
- [ ] Timeout error rate упал с 50% до <5% на реальных запросах (50+ символов)
- [ ] Coverage не упал ниже 54%

## Попытки

<!-- Orchestrator дописывает сюда при каждой попытке -->
