# Цикл полуавтоматической отладки RAG GigaChat

> **Назначение файла**: инструкция для Claude Code по созданию и эксплуатации цикла отладки на базе встроенного `Agent` tool, `pytest` safety-gates и строго структурированного Backlog.
> **Целевая аудитория**: Claude (как исполнитель) + владелец проекта (как пользователь).
> **Язык**: русский (идентификаторы — английские).

---

## 1. Цель и роли

Цель цикла — итеративно находить и устранять дефекты RAG-пайплайна с минимальными усилиями со стороны пользователя и максимальной гарантией безопасности изменений.

| Роль | Исполнитель | Ответственность |
|------|-------------|------------------|
| **Debug Runner** | Python-скрипт `scripts/debug/run_debug.py` | Запуск Streamlit с усиленным логированием; эмит структурированного event log (CSV) через `emit()` + ротация файлов |
| **Process Miner** | `scripts/debug/mine_process.py` (pure-Python) | Агрегирует `events_*.csv` в варианты трасс, bottleneck-таблицу (p50/p95/p99), выявляет отклонения |
| **Log Analyzer** | Claude (через `Agent(subagent_type=Explore)` + `Agent(subagent_type=Plan)`) | Читает `last_session_summary.md` + `events_*.csv`, использует PM-сигналы (редкие варианты, bottlenecks) для формирования BKL-записей |
| **Orchestrator** | Claude (основной поток) | Берёт приоритетную BKL-запись, создаёт ветку, применяет фикс, прогоняет pytest, коммитит или откатывает |
| **User (владелец проекта)** | Человек | Запускает скрипты, выдаёт триггер-фразы, подтверждает изменения в critical-файлах, делает `git push` |

Пользователь **никогда** не копирует логи вручную в чат — Claude читает их сам через `Read`.

---

## 2. Архитектура цикла

```
  ┌─────────────────────────────────────────────────────────────┐
  │                     ОДНА ИТЕРАЦИЯ ЦИКЛА                     │
  └─────────────────────────────────────────────────────────────┘

  [User]              [Debug Runner]           [Log Analyzer]
    │                       │                         │
    │ ./run_debug.py        │                         │
    ├──────────────────────▶│                         │
    │                       │  logs/session_*.log     │
    │                       │  logs/events_*.csv      │
    │                       │─────┐  (emit events)    │
    │  (действия в UI)      │     │                   │
    ├──────────────────────▶│     ▼                   │
    │    Ctrl+C             │                         │
    │                       │  mine_process.py        │
    │                       │─────┐ (variants +       │
    │                       │     │  bottlenecks +    │
    │                       │     ▼  errors)          │
    │               logs/last_session_summary.md      │
    │                                                 │
    │   #analyze-logs                                 │
    ├────────────────────────────────────────────────▶│
    │                                                 │ Agent(Explore)
    │                                                 │   ∥
    │                                                 │ Agent(Plan)
    │                                     backlog/BKL-NNN-*.md
    │                                                 │
    │   #status                                       │
    ├────────────────────────────────────────────────▶│ → INDEX.md
    │                                                 │
    │   #apply-fix BKL-NNN     [Orchestrator]         │
    ├────────────────────────────────────────────────▶│
    │                           git checkout -b       │
    │                           → правка              │
    │                           → pytest              │
    │                      ┌───────┴────────┐         │
    │                   ✅ pass           ❌ fail     │
    │                  commit           reset --hard  │
    │                  status:done      status:blocked│
    │                                                 │
    │   git push (вручную, после ревью)               │
    ◀─────────────────────────────────────────────────┘
```

---

## 3. Безопасность и гарантии

### Pre-conditions (проверяются Orchestrator'ом перед фиксом)
1. Рабочее дерево чистое (`git status --porcelain` пуст) или изменения явно относятся к задаче.
2. BKL-запись существует, валидна по схеме и имеет `status: open`.
3. `safety_checks` из BKL-записи — непустой список существующих тестов.

### Post-conditions (проверяются после фикса)
1. Все тесты из `safety_checks` прошли.
2. Никакие другие unit-тесты не регрессировали (`pytest tests/unit/ -q`).
3. Коммит создан с сообщением `fix: <title> (refs BKL-NNN)`.

### Границы автоматики
- ❌ Orchestrator **никогда** не делает `git push` — это остаётся за пользователем после ревью.
- ❌ Orchestrator **никогда** не правит critical-файлы (`CLAUDE.md`, `config.py`, `.env`, `requirements.txt`, `.gitignore`) без явного подтверждения пользователя.
- ❌ Orchestrator не удаляет файлы без подтверждения (политика проекта).
- ✅ Orchestrator создаёт/удаляет собственные feature-ветки `debug-fix/BKL-NNN`.
- ✅ Orchestrator делает локальные коммиты в feature-ветке.
- ✅ При провале pytest — автоматический `git reset --hard HEAD` + удаление feature-ветки + пометка `status: blocked` в BKL с указанием причины.

### Rollback-протокол
```bash
# При падении pytest:
git reset --hard HEAD
git checkout main
git branch -D debug-fix/BKL-NNN
# BKL-запись получает status: blocked + запись в секции "Попытки".
```

---

## 4. Файловая структура после внедрения

```
rag_GigaChat/
├── prompts/
│   └── generateImprovementsProcessMining.md   ← этот файл (после переименования)
├── scripts/debug/
│   ├── run_debug.py                           ← Debug Runner
│   ├── mine_process.py                        ← Process Miner (variants/bottlenecks/errors)
│   └── backlog_index.py                       ← генератор INDEX.md
├── src/rag_gigachat/utils/
│   ├── debug_context.py                       ← StepTracker, @trace (raw logs)
│   └── event_log.py                           ← ProcessEvent, CaseContext, emit()
├── backlog/
│   ├── README.md                              ← описание схемы и правил
│   ├── template.md                            ← шаблон записи
│   ├── INDEX.md                               ← автогенерируемый индекс
│   └── BKL-000-example.md                     ← пример заполнения
├── tests/debug/
│   └── test_backlog_schema.py                 ← валидатор YAML frontmatter
├── .claude/hooks/
│   └── post_edit_pytest.sh                    ← хук PostToolUse
├── logs/                                       ← сессии и сводки (gitignored)
└── .env.example                                ← +RAG_DEBUG, +RAG_LOG_LEVEL
```

---

## 5. Схема данных

### 5.1 Backlog entry (`backlog/BKL-NNN-<slug>.md`)

```yaml
---
id: BKL-001                       # обязательно, уникально, формат BKL-\d{3}
title: "Краткое описание"         # обязательно, ≤80 символов
priority: high                    # high | medium | low
severity: major                   # critical | major | minor
status: open                      # open | in-progress | blocked | done | rejected
created: 2026-04-17               # ISO date
updated: 2026-04-17               # ISO date
affected_files:                   # список путей относительно корня репо
  - src/rag_gigachat/data/pdf_loader.py
tags: [ocr, pdf, data-loading]    # свободные метки
linked_logs: logs/session_20260417_1430.log#L120-L145   # опционально
linked_events: logs/events_20260417_1430.csv           # опционально (PM-источник)
process_mining_evidence:                                # опционально, заполняет Log Analyzer
  variant: "V3"                                         # идентификатор варианта
  bottleneck_activity: "retrieval.vector_search"        # если причина — bottleneck
  anomaly_type: "bimodal_distribution"                  # rare_variant | bimodal_distribution | high_error_rate | null
  affected_cases: 3
safety_checks:                    # обязательно, минимум 1 элемент
  - pytest tests/unit/test_pdf_loader.py
rollback: "git reset --hard HEAD"
estimated_effort: 30min           # 15min | 30min | 1h | 2h+
---

## Проблема
Симптомы + выдержки из логов.

## Гипотеза причины
Почему это происходит (опционально, если неочевидно).

## Предлагаемое исправление
Конкретные файлы/функции/diff-набросок.

## Критерии приёмки
- [ ] Лог не содержит ошибки X
- [ ] `pytest tests/unit/test_pdf_loader.py` — green
- [ ] (опционально) ручная проверка: загрузка `data/samples/test_rotated.pdf`

## Попытки
<!-- Orchestrator дописывает сюда при каждой попытке -->
```

### 5.2 Правила приоритизации

| Значение | Priority (блокирует работу?) | Severity (степень некорректности?) |
|----------|------------------------------|-------------------------------------|
| **high / critical** | Критичный путь сломан; потеря данных; security | Падение приложения; crash |
| **medium / major** | Деградация UX; perf regression >20% | Функция работает некорректно |
| **low / minor** | Косметика; мелкие warnings в логах | Неудобно, но допустимо |

**Orchestrator берёт задачи в порядке**: `priority DESC, severity DESC, created ASC`.

### 5.3 Event log (`logs/events_YYYYMMDD_HHMMSS.csv`) — process mining primary source

Каждая строка — одно событие пайплайна. Формат совместим с любым PM-инструментом (Disco, ProM, pm4py).

| Колонка | Тип | Описание |
|---------|-----|-----------|
| `case_id` | `str` | Сквозной ID одного прогона (query, загрузка документа). Формат: `Q-YYYYMMDDHHMMSS-<6hex>` |
| `activity` | `str` | Имя шага из канонического словаря (§5.4). Падает, если вне словаря |
| `timestamp` | ISO-8601 | `YYYY-MM-DDTHH:MM:SS.mmm` — начало события |
| `resource` | `str` | Компонент-исполнитель: `streamlit` / `gigachat` / `faiss` / `bm25` / `pipeline` / `loader` |
| `duration_ms` | `float` | Длительность шага |
| `status` | enum | `ok` / `warn` / `error` |
| `attributes` | JSON-string | Произвольные контекстные данные: `{"query_len": 42, "top_k": 5, "tokens": 1250}` |

Пример:
```csv
case_id,activity,timestamp,resource,duration_ms,status,attributes
Q-20260417143001-a3f,session.start,2026-04-17T14:30:01.120,streamlit,0,ok,"{""user"":""anon""}"
Q-20260417143001-a3f,query.embed,2026-04-17T14:30:01.472,gigachat,340,ok,"{""dim"":1024}"
Q-20260417143001-a3f,retrieval.vector_search,2026-04-17T14:30:01.517,faiss,45,ok,"{""top_k"":5}"
Q-20260417143001-a3f,llm.call,2026-04-17T14:30:03.635,gigachat,2100,ok,"{""tokens"":1250}"
```

Эмиссия событий — через контекстный менеджер `emit()`:
```python
with emit("retrieval.vector_search", resource="faiss", top_k=5):
    results = self.index.search(query_vec, k=5)
```
`case_id` распространяется через `contextvars.ContextVar` — устанавливается в точке входа (`RAGPipeline.query()`, `DocumentLoader.load()`).

### 5.4 Канонический словарь activities

Фиксированный список из ~20 шагов. Валидатор в `event_log.py` отклоняет любое имя вне словаря (защита от drift при рефакторинге).

```python
CANONICAL_ACTIVITIES = {
    # жизненный цикл
    "session.start", "session.end",
    # загрузка документов
    "document.load", "document.ocr", "document.chunk",
    # обработка запроса
    "query.receive", "query.embed", "query.rewrite",
    # поиск
    "retrieval.vector_search", "retrieval.bm25", "retrieval.rerank",
    # построение контекста
    "context.build", "context.truncate",
    # LLM
    "llm.call", "llm.stream_start", "llm.stream_chunk", "llm.complete",
    # ответ
    "response.render",
    # кэш
    "cache.hit", "cache.miss",
}
```

**В MVP инструментируем 10 точек** (остальные — по мере надобности):
1. `session.start` / `session.end` (пара, 2 эмита)
2. `query.receive`
3. `query.embed`
4. `retrieval.vector_search`
5. `retrieval.rerank`
6. `context.build`
7. `llm.call`
8. `response.render`
9. `document.load`
10. `document.ocr`

### 5.5 Отчёт Process Miner (`logs/last_session_summary.md`)

Генерируется `mine_process.py` и содержит 4 раздела:

1. **Variants** — уникальные трассы (последовательности activities) с частотой.
   Пример: `V1 (17×): session.start → query.receive → query.embed → retrieval.vector_search → retrieval.rerank → context.build → llm.call → response.render → session.end`
2. **Bottlenecks** — таблица p50/p95/p99/max по каждой activity, сортировка по p95 DESC.
3. **Errors with trace context** — ошибки с привязкой к `case_id` и предшествующим шагам.
4. **Anomalies** — pure-Python эвристики:
   - Редкие варианты (≤2 случая при N≥10 всего)
   - Bimodal distribution (p95/p50 > 3) — сигнал retry/timeout
   - Activities с `status=error` > 5% случаев

---

## 6. Режимы работы Claude

Claude переключается между ролями по **триггер-фразам**. Если триггера нет — Claude работает в обычном режиме.

| Триггер | Роль | Действия Claude |
|---------|------|------------------|
| `#analyze-logs [path?]` | Log Analyzer | 1. `Read logs/last_session_summary.md` + последний `logs/events_*.csv` (PM-сигналы: варианты, bottlenecks, аномалии). 2. Параллельно: `Agent(subagent_type=Explore)` по сырому логу для поиска аномалий + `Agent(subagent_type=Plan)` для плана фикса с учётом PM-контекста. 3. Создать BKL-записи (поле `process_mining_evidence` заполнять по возможности). 4. Обновить `backlog/INDEX.md`. |
| `#status` | Status Reporter | Запустить `python scripts/debug/backlog_index.py` и вывести сводку открытых задач, отсортированных по priority/severity. |
| `#apply-fix BKL-NNN` | Orchestrator | См. алгоритм ниже. |
| `#reject BKL-NNN <reason>` | Cleanup | Установить `status: rejected`, добавить причину в секцию «Попытки», обновить `updated`. |

### Алгоритм Orchestrator
1. **Прочитать** `backlog/BKL-NNN-*.md`; проверить валидность schema и `status: open`.
2. **Проверить** чистоту рабочего дерева (`git status --porcelain`). Если не чисто — спросить пользователя.
3. **Создать ветку**: `git checkout -b debug-fix/BKL-NNN`.
4. **Установить** в BKL `status: in-progress`, `updated: <today>`.
5. **Применить** исправление (`Edit`/`Write`). Critical-файлы — только с подтверждением пользователя.
6. **Запустить** все команды из `safety_checks` + общий `pytest tests/unit/ -q`.
7. **Развилка**:
   - ✅ **pass**: `git add <changed>` → `git commit -m "fix: <title> (refs BKL-NNN)"` → в BKL: `status: done`, запись в «Попытки».
   - ❌ **fail**: `git reset --hard HEAD` → `git checkout main` → `git branch -D debug-fix/BKL-NNN` → в BKL: `status: blocked` + причина в «Попытки».
8. **НЕ делать** `git push` — выйти и дать пользователю возможность ревью.

---

## 7. Команды для пользователя

```bash
# 1. Запуск debug-сессии
python scripts/debug/run_debug.py

# (в другом терминале или после Ctrl+C) запускаем process mining
# читает последние logs/session_*.log + logs/events_*.csv,
# пишет logs/last_session_summary.md (variants / bottlenecks / errors / anomalies)
python scripts/debug/mine_process.py

# 2. Анализ — Claude сам прочитает logs/last_session_summary.md
#    В чате:
#    #analyze-logs

# 3. Просмотр задач
#    #status

# 4. Применение фикса
#    #apply-fix BKL-001

# 5. Ревью и пуш (вручную)
git log --oneline -5
git diff main..debug-fix/BKL-001
git checkout main && git merge --no-ff debug-fix/BKL-001
git push origin main
```

---

## 8. Definition of Done (одна итерация)

Итерация считается успешно завершённой, когда:
- [ ] BKL-запись имеет `status: done`.
- [ ] Локальный коммит создан в `debug-fix/BKL-NNN`.
- [ ] `pytest tests/unit/ -q` — green.
- [ ] Coverage не упал ниже 54% (порог из `CLAUDE.md`).
- [ ] Пользователь провёл ревью и сделал merge/push (вне автоматики).

---

## 9. Будущие улучшения (out-of-scope MVP)

- **`scripts/debug/scenarios/*.yaml`** — декларативные сценарии воспроизведения (`load_pdf_ocr.yaml`, `query_russian_docs.yaml`, `long_context_regression.yaml`). Исполнитель: `scripts/debug/apply_scenarios.py` через streamlit-testing-framework. Позволит headless-прогоны вместо ручных кликов.
- **Conformance checking** — `logs/expected_traces.yaml` с happy path и допустимыми вариантами; `mine_process.py` подсвечивает skipped/extra/reordered шаги относительно модели.
- **pm4py интеграция** — подключение библиотеки для α-алгоритма, inductive miner, визуализации моделей в graphviz. CSV-формат event log уже pm4py-совместим.
- **XES export** — конвертация `events_*.csv` → `events_*.xes` для загрузки в Disco/ProM.
- **Object-centric event log (OCEL 2.0)** — следующее поколение event log с множественными объектами (query × document × chunk).
- **Автопрогон хуком ErrorCount** — если в сессии >N ERROR, `#analyze-logs` запускается автоматически.
- **Retry-политика** — если фикс провалился 2 раза подряд, BKL-запись переводится в `status: blocked` с эскалацией человеку.
- **Metrics dashboard** — агрегация MTTR/MTBF по тегам в `backlog/` + process-level метрики (variant stability, bottleneck drift).

---

## 10. План работ (чеклист внедрения)

> Выполнять по порядку. На каждом этапе — подтверждение пользователя. Правка critical-файлов (`config.py`, `.env.example`) — с отдельным подтверждением.

### Этап 0. Подготовка (15 мин)
- [ ] Переименовать `prompts/generateImpuvementsProcessMining.md` → `prompts/generateImprovementsProcessMining.md`
- [ ] Создать ветку `feature/debug-loop-v2`
- [ ] Создать директории: `scripts/debug/`, `backlog/`, `logs/`, `tests/debug/`, `.claude/hooks/`
- [ ] Добавить `logs/` и `backlog/INDEX.md` в `.gitignore` (INDEX — автогенерация)

### Этап 1. Инфраструктура логирования (30 мин)
- [ ] Создать `src/rag_gigachat/utils/debug_context.py` с `StepTracker` и декоратором `@trace` на базе существующего `logging.getLogger(__name__)` (без нового `basicConfig`)
- [ ] Добавить в `src/rag_gigachat/config.py` поля `RAG_DEBUG: bool` и `RAG_LOG_LEVEL: str` **⚠ critical file — спросить подтверждение**
- [ ] Обновить `.env.example` новыми переменными **⚠ спросить подтверждение**
- [ ] Точечно добавить `@trace` на ключевые методы `RAGPipeline.query`, `VectorStore.search`, `LLMManager.generate`

### Этап 1.5. Process Mining infrastructure (60 мин)
- [ ] `src/rag_gigachat/utils/event_log.py`:
  - `ProcessEvent` (dataclass с 7 полями из §5.3)
  - `CaseContext` на базе `contextvars.ContextVar` (генерация `case_id` + проброс по стеку)
  - `emit(activity, resource, **attrs)` — контекстный менеджер: стартует таймер, пишет событие в CSV при выходе, `status=error` при исключении
  - `CANONICAL_ACTIVITIES` (§5.4) + валидатор — `ValueError` при попытке эмита неизвестной activity
  - CSV-writer с ротацией по сессиям (`logs/events_YYYYMMDD_HHMMSS.csv`), thread-safe append
- [ ] Инструментация **10 точек** (см. §5.4 MVP-список): обернуть существующие методы контекстным менеджером `emit()`
  - `session.start` / `session.end` — в точке входа Streamlit (`streamlit_app.py`)
  - `query.receive`, `query.embed`, `retrieval.vector_search`, `retrieval.rerank`, `context.build`, `llm.call`, `response.render` — в `RAGPipeline`
  - `document.load`, `document.ocr` — в `DocumentLoader`
- [ ] Unit-тест `tests/debug/test_event_log.py`:
  - эмит валидной activity → запись появляется в CSV
  - эмит невалидной activity → `ValueError`
  - вложенные `emit()` наследуют `case_id`
  - исключение в блоке → `status=error` и duration измерена

### Этап 2. Python-скрипты отладки (45 мин)
- [ ] `scripts/debug/run_debug.py` — запуск Streamlit через `subprocess` с `.venv/bin/streamlit`, ротация логов `logs/session_YYYYMMDD_HHMMSS.log`, graceful shutdown по SIGINT
- [ ] `scripts/debug/mine_process.py` (**заменяет `collect_logs.py`**) — pure-Python process mining:
  - Читает последний `logs/events_*.csv` + `logs/session_*.log`
  - **Variants**: группировка по `case_id` → последовательность activities → подсчёт частот, топ-5 + редкие (≤2 случая при N≥10)
  - **Bottlenecks**: per-activity агрегация (count, p50, p95, p99, max), сортировка по p95 DESC
  - **Errors**: выборка строк `status=error` + 5 строк контекста из сырого лога по timestamp
  - **Anomalies**: bimodal distribution (p95/p50 > 3), high error rate (>5%), rare variants
  - Вывод — в `logs/last_session_summary.md` с 4 разделами (§5.5)
- [ ] `scripts/debug/backlog_index.py` — генератор `backlog/INDEX.md` с сортировкой по priority/severity, группировкой по status

### Этап 3. Backlog и схемы (20 мин)
- [ ] `backlog/README.md` — описание полей, правил приоритизации, жизненного цикла записи
- [ ] `backlog/template.md` — шаблон записи (см. §5.1)
- [ ] `backlog/BKL-000-example.md` — пример на **реальной** проблеме, найденной через `grep -rn "TODO\|FIXME\|XXX\|HACK" src/` в кодовой базе
- [ ] Начальный `backlog/INDEX.md` (заглушка, будет перегенерирован)

### Этап 4. ~~Сценарии `scenarios.yaml`~~ — перенесено в §9 «Будущие улучшения» (out-of-scope MVP)

### Этап 5. Safety gates через pytest (30 мин)
- [ ] `tests/debug/test_backlog_schema.py` — валидация YAML frontmatter всех `backlog/BKL-*.md` через `pydantic` (уже есть через LangChain)
- [ ] `.claude/hooks/post_edit_pytest.sh` — автопрогон `pytest tests/unit/ -q` после `Edit`/`Write` в `src/`
- [ ] Регистрация хука через skill `update-config` (`PostToolUse` matcher на `Edit|Write`)
- [ ] Задокументировать отключение хука: `RAG_SKIP_HOOK=1`

### Этап 6. Интеграция ролей Claude (25 мин)
- [ ] Прописать в этом файле 4 триггер-фразы (§6) — **готово**
- [ ] Алгоритм Orchestrator (§6) — **готово**
- [ ] Проверить, что Log Analyzer действительно запускает `Agent(Explore)` и `Agent(Plan)` в одном турне (параллельно, не последовательно)
- [ ] Sanity-check границ (§3): нет `git push`, нет правки critical-файлов без подтверждения

### Этап 7. Документация (15 мин)
- [ ] Обновить `README_RU.md`: новый раздел «Цикл отладки» со ссылкой на этот файл
- [ ] Обновить `CHANGELOG.md`: запись о v1.9.0 (debug loop)

### Этап 8. Smoke-проверка (15 мин)
- [ ] `python scripts/debug/run_debug.py` → короткая сессия с тестовым PDF → Ctrl+C
- [ ] Убедиться, что `logs/events_*.csv` содержит события из 10 инструментированных точек и все `case_id` валидны
- [ ] `python scripts/debug/mine_process.py` → проверить, что `last_session_summary.md` содержит 4 раздела (variants / bottlenecks / errors / anomalies)
- [ ] `#analyze-logs` → убедиться, что Log Analyzer использует PM-сигналы и заполняет `process_mining_evidence` в BKL-записи
- [ ] `#apply-fix BKL-NNN` на синтетической записи → проверить что:
  - при зелёном pytest создаётся коммит в feature-ветке
  - при умышленно сломанном фиксе происходит `git reset --hard` и ветка удаляется
- [ ] `#status` → корректная сортировка по приоритету

**Суммарная оценка: ~4 часа чистой работы** (было ~3 ч + 60 мин Этап 1.5).

---

## Приложение A. Дефолты, принятые при составлении плана

Если пользователь не скажет иначе, используются:
- **Валидация schema**: `pydantic` (уже в зависимостях через LangChain)
- **Хук pytest**: `pytest tests/unit/ -q` (быстрый, ~5-10 сек); полный прогон — только в Orchestrator
- **Пример BKL-000**: на реальной проблеме из кодовой базы (поиск через `grep -rn "TODO\|FIXME\|XXX\|HACK" src/`)
- **Наименование веток**: `debug-fix/BKL-NNN` (snake-case после префикса)
- **Формат коммита**: `fix: <title> (refs BKL-NNN)` — совместимо с соглашением из `CLAUDE.md`
- **Process mining стек**: pure-Python в MVP (без `pm4py`); event log в CSV (pm4py-совместим); conformance checking — в §9 «Будущие улучшения»
- **Инструментация в MVP**: 10 точек из канонического словаря (§5.4), остальные ~10 activities — добавляются по мере необходимости
- **Case ID**: генерируется в точке входа (`RAGPipeline.query()`, `DocumentLoader.load()`), проброс через `contextvars.ContextVar`
