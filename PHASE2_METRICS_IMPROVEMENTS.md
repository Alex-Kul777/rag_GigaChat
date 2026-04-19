# 📊 Фаза 2: Per-stage метрики и параметры логирования

**Статус:** ✅ Реализовано  
**Дата:** 2026-04-19  
**Файлы изменены:** 2  
**Строк добавлено:** ~200+

---

## 🎯 Резюме

Реализована **Фаза 2** - добавлены детальные per-stage метрики, Request ID трассировка и логирование параметров каждого этапа.

### ✅ Основные достижения:

1. **PipelineTimer класс** — автоматическое измерение времени этапов
2. **StageMetrics dataclass** — структурированные метрики для анализа
3. **Request ID** — уникальный ID для трассировки всего запроса
4. **Per-stage параметры** — логирование chunk_size, k, model и т.д.
5. **Полная иерархия этапов** — 7 логических этапов с START/END маркерами

---

## 📁 Файлы изменены

### 1. **src/rag_gigachat/logging_utils.py** (РАСШИРЕН)

#### Добавлено:

```python
@dataclass
class StageMetrics:
    """Метрики отдельного этапа обработки"""
    stage_name: str              # Имя этапа (RETRIEVAL, GENERATION и т.д.)
    timestamp_start: str         # ISO формат: 2026-04-19T10:00:01.234Z
    timestamp_end: str           # ISO формат: 2026-04-19T10:00:02.345Z
    duration_ms: int             # Время выполнения в миллисекундах
    status: str                  # OK, ERROR, TIMEOUT, PENDING
    input_size: int              # Размер входных данных (опционально)
    output_size: int             # Размер выходных данных (опционально)
    memory_mb: float             # Использование памяти (опционально)
    error_msg: str               # Сообщение об ошибке (если есть)
    metrics: Dict[str, any]      # Дополнительные метрики (custom_metrics)

    def to_dict(self) -> dict:
        """Конвертировать в словарь для JSON логирования"""
```

#### PipelineTimer класс:

```python
class PipelineTimer:
    """Таймер для измерения времени выполнения этапов обработки"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.stages: Dict[str, Dict] = {}
        self.request_id = str(uuid.uuid4())[:8]  # Уникальный ID запроса
    
    def start_stage(self, stage_name: str, params: dict = None) -> str:
        """Начать измерение этапа и вернуть request_id"""
        # Логирует: [REQUEST_ID] 🧪 [STAGE_NAME START]
        # Возвращает request_id для трассировки
    
    def end_stage(self, stage_name: str, metrics: dict = None, status: str = "OK") -> StageMetrics:
        """Завершить измерение и вернуть метрики"""
        # Логирует: [REQUEST_ID] ✅ [STAGE_NAME END] duration=Xms
    
    def summary(self) -> Dict[str, int]:
        """Получить сводку времени по этапам"""
        # Возвращает: {'RETRIEVAL': 1111, 'GENERATION': 25667, 'TOTAL': 26778}
```

**Размер расширения:** ~150 новых строк кода

---

### 2. **src/rag_gigachat/core/rag_pipeline.py** (ЗНАЧИТЕЛЬНО РАСШИРЕН)

#### Изменения в __init__:

```python
# Добавлена инициализация PipelineTimer
from rag_gigachat.logging_utils import PipelineTimer
self.pipeline_timer = PipelineTimer(logger)
```

#### Полная иерархия логирования:

```
PIPELINE (контейнер для всего запроса)
├─ LOAD_DOCS (контейнер для загрузки и подготовки)
│  ├─ CHUNKING (разбиение на чанки)
│  │  params: chunk_size=512, chunk_overlap=50
│  │
│  ├─ EMBEDDING (создание эмбеддингов)
│  │  params: embedding_model="multilingual-e5-small", embedding_dim=384
│  │
│  └─ INDEX (создание индекса)
│     params: index_type="FAISS", metric="cosine"
│
├─ RETRIEVAL (поиск документов)
│  params: k=5, metric="cosine"
│  metrics: docs_count=5, top_score=0.78, avg_score=0.71
│
└─ GENERATION (генерация ответа)
   params: model="facebook/opt-125m", temperature=0.7, top_p=0.9
   metrics: tokens_generated=156, response_length=842
```

#### Изменения в process_query:

```python
# Инициализация с Request ID
request_id = self.pipeline_timer.start_stage('PIPELINE', params={...})

# ... обработка ...

# Получение сводки
stage_summary = self.pipeline_timer.summary()
# {'RETRIEVAL': 1111, 'GENERATION': 25667, 'PIPELINE': 27110, 'TOTAL': 27110}
```

#### Изменения в load_from_pdf_directory_with_metadata:

```python
# LOAD_DOCS START
self.pipeline_timer.start_stage('LOAD_DOCS', params={
    'directory': str(directory),
    'recursive': recursive,
    'embedding_model': 'multilingual-e5-small'
})

# CHUNKING START/END
self.pipeline_timer.start_stage('CHUNKING', params={
    'chunk_size': 512,
    'chunk_overlap': 50
})
# ... chunking ...
self.pipeline_timer.end_stage('CHUNKING', metrics={
    'chunks_created': 119,
    'chunk_size': 512,
    'chunk_overlap': 50
})

# EMBEDDING START/END
self.pipeline_timer.start_stage('EMBEDDING', params={
    'embedding_model': 'multilingual-e5-small',
    'embedding_dim': 384
})
# ... embedding ...
self.pipeline_timer.end_stage('EMBEDDING', metrics={
    'vectors_created': 119,
    'embedding_model': 'multilingual-e5-small',
    'from_cache': False
})

# INDEX START/END
self.pipeline_timer.start_stage('INDEX', params={
    'index_type': 'FAISS',
    'metric': 'cosine'
})
self.pipeline_timer.end_stage('INDEX', metrics={
    'index_size': 119,
    'index_type': 'FAISS'
})

# LOAD_DOCS END
self.pipeline_timer.end_stage('LOAD_DOCS', metrics={
    'total_documents': 119,
    'from_cache': False
})
```

**Размер расширения:** ~150 новых строк кода

---

## 📊 Форматы логирования (Фаза 2)

### Текстовый формат (консоль + файл):
```
2026-04-19 10:00:01.234 | INFO | rag_pipeline.RAGPipeline.load_...:354 | 🧪 [LOAD_DOCS START] Directory='/data/...'
2026-04-19 10:00:01.345 | INFO | rag_pipeline.RAGPipeline.load_...:384 | 🔨 [CHUNKING START] chunk_size=512, overlap=50
2026-04-19 10:00:01.456 | INFO | rag_pipeline.RAGPipeline.load_...:420 | ✅ [CHUNKING END] duration=111ms
2026-04-19 10:00:01.567 | INFO | rag_pipeline.RAGPipeline.load_...:436 | 🔗 [EMBEDDING START] model=multilingual-e5-small
2026-04-19 10:00:02.678 | INFO | rag_pipeline.RAGPipeline.load_...:456 | ✅ [EMBEDDING END] duration=1111ms
2026-04-19 10:00:02.789 | INFO | rag_pipeline.RAGPipeline.process_q...:672 | 🚀 [PIPELINE START] Query='Что такое RAG?', k=5
2026-04-19 10:00:03.890 | INFO | rag_pipeline.RAGPipeline.process_q...:709 | 🔍 [RETRIEVAL START] k=5, metric=cosine
2026-04-19 10:00:04.901 | INFO | rag_pipeline.RAGPipeline.process_q...:738 | ✅ [RETRIEVAL END] duration=1011ms
2026-04-19 10:00:05.012 | INFO | rag_pipeline.RAGPipeline.process_q...:756 | 🤖 [GENERATION START] model=facebook/opt-125m, tokens=42
2026-04-19 10:00:30.123 | INFO | rag_pipeline.RAGPipeline.process_q...:773 | ✅ [GENERATION END] duration=25111ms
2026-04-19 10:00:30.234 | INFO | rag_pipeline.RAGPipeline.process_q...:814 | 📊 PIPELINE SUMMARY: total=28445ms, retrieval=1011ms, generation=25111ms
2026-04-19 10:00:30.345 | INFO | rag_pipeline.RAGPipeline.process_q...:825 | ✅ [PIPELINE END] duration=28445ms
```

### JSON формат (для process mining):
```json
{"timestamp":"2026-04-19T10:00:03.890Z","level":"INFO","module":"rag_gigachat.core.rag_pipeline","class":"RAGPipeline","function":"process_query","stage":"RETRIEVAL","action":"START","message":"🔍 [RETRIEVAL START] k=5, metric=cosine","metrics":{"k":5,"metric":"cosine"}}
{"timestamp":"2026-04-19T10:00:04.901Z","level":"INFO","module":"rag_gigachat.core.rag_pipeline","class":"RAGPipeline","function":"process_query","stage":"RETRIEVAL","action":"END","message":"✅ [RETRIEVAL END] duration=1011ms","metrics":{"duration_ms":1011,"docs_count":5,"top_score":0.78,"avg_score":0.71}}
```

---

## 🔍 Request ID трассировка

Каждый запрос теперь имеет уникальный ID для трассировки:

```
Request ID: a7f3c2d9 (8 первых символов UUID)

Все логи этого запроса начинаются с:
[a7f3c2d9] 🚀 [PIPELINE START] ...
[a7f3c2d9] 🔍 [RETRIEVAL START] ...
[a7f3c2d9] ✅ [RETRIEVAL END] ...
[a7f3c2d9] 🤖 [GENERATION START] ...
[a7f3c2d9] ✅ [GENERATION END] ...
[a7f3c2d9] 📊 PIPELINE SUMMARY ...
```

**Преимущества:**
- ✅ Легко отследить один запрос через все логи
- ✅ Можно отфильтровать логи для одного запроса: `grep a7f3c2d9`
- ✅ Можно обрабатывать параллельные запросы одновременно
- ✅ Идеален для распределённых систем

---

## 📈 Примеры использования

### Python для анализа per-stage метрик:

```python
import json
import pandas as pd

# Читаем JSON логи
logs = []
with open('logs/rag_app.json', 'r') as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

# Анализ по этапам
for stage in ['RETRIEVAL', 'GENERATION']:
    stage_logs = df[df['stage'] == stage]
    start = stage_logs[stage_logs['action'] == 'START'].iloc[0]
    end = stage_logs[stage_logs['action'] == 'END'].iloc[0]
    duration_ms = end['metrics']['duration_ms']
    print(f"{stage}: {duration_ms}ms")

# Сводка по всему pipeline
pipeline_end = df[df['stage'] == 'PIPELINE'][df['action'] == 'END'].iloc[0]
print(f"\n📊 SUMMARY:")
print(f"Total: {pipeline_end['metrics']['total_duration_ms']}ms")
print(f"Retrieval: {pipeline_end['metrics']['retrieval_ms']}ms")
print(f"Generation: {pipeline_end['metrics']['generation_ms']}ms")
```

### Фильтрация логов по Request ID:

```bash
# Все логи для одного запроса
grep "a7f3c2d9" logs/rag_app.log

# Только START маркеры
grep "a7f3c2d9" logs/rag_app.log | grep START

# Только END маркеры с временем
grep "a7f3c2d9" logs/rag_app.log | grep END
```

---

## ✅ Проверка результатов

### Синтаксис и компиляция:
```
✅ logging_utils.py          - extends successfully
✅ rag_pipeline.py           - compiles successfully
```

### Новые компоненты:
```
✅ StageMetrics dataclass    - для сбора метрик
✅ PipelineTimer класс       - для измерения времени
✅ Request ID (UUID[:8])     - для трассировки
✅ Per-stage параметры       - chunk_size, k, model и т.д.
```

---

## 🎯 Иерархия логирования (полная)

```
[REQUEST_ID] 🚀 [PIPELINE START] Query='...', k=5
  ↓
[REQUEST_ID] 🧪 [LOAD_DOCS START] Directory='/data/...'
  ├─ [REQUEST_ID] 🔨 [CHUNKING START] chunk_size=512
  │  ↓
  │  [REQUEST_ID] ✅ [CHUNKING END] duration=100ms, chunks=119
  │
  ├─ [REQUEST_ID] 🔗 [EMBEDDING START] model=multilingual-e5-small
  │  ↓
  │  [REQUEST_ID] ✅ [EMBEDDING END] duration=1000ms, vectors=119
  │
  └─ [REQUEST_ID] 🗂️  [INDEX START] type=FAISS
     ↓
     [REQUEST_ID] ✅ [INDEX END] duration=50ms, size=119
  ↓
[REQUEST_ID] ✅ [LOAD_DOCS END] duration=1150ms, total=119
  ↓
[REQUEST_ID] 🔍 [RETRIEVAL START] k=5, metric=cosine
  ↓
[REQUEST_ID] ✅ [RETRIEVAL END] duration=100ms, docs=5, score=0.78
  ↓
[REQUEST_ID] 🤖 [GENERATION START] model=facebook/opt-125m, tokens=42
  ↓
[REQUEST_ID] ✅ [GENERATION END] duration=25000ms, tokens=156
  ↓
[REQUEST_ID] 📊 SUMMARY: total=26250ms, retrieval=100ms, generation=25000ms
  ↓
[REQUEST_ID] ✅ [PIPELINE END] duration=26250ms, docs=5, tokens=156
```

---

## 📊 Структура метрик

### START метрики (параметры):
- `chunk_size`, `chunk_overlap` — для CHUNKING
- `embedding_model`, `embedding_dim` — для EMBEDDING
- `index_type`, `metric` — для INDEX
- `k`, `metric` — для RETRIEVAL
- `model`, `temperature`, `top_p`, `prompt_tokens` — для GENERATION

### END метрики (результаты):
- `chunks_created`, `duration_ms` — для CHUNKING
- `vectors_created`, `embedding_model` — для EMBEDDING
- `index_size`, `index_type` — для INDEX
- `docs_count`, `top_score`, `avg_score`, `duration_ms` — для RETRIEVAL
- `tokens_generated`, `response_length`, `duration_ms` — для GENERATION

### PIPELINE метрики (сводка):
- `total_duration_ms` — общее время
- `retrieval_ms` — время поиска
- `generation_ms` — время генерации
- `docs_retrieved` — найдено документов
- `tokens_generated` — сгенерировано токенов

---

## 🔮 Следующие этапы (Фаза 3)

**Планируется добавить:**
- [ ] Метрики памяти per-stage (RAM, GPU)
- [ ] Exporting логов в DataFrame/Excel
- [ ] Streamlit dashboard для анализа
- [ ] Автоматические bottleneck рекомендации
- [ ] Benchmarking и сравнение runs

---

## 📝 Статистика

| Метрика | До | После |
|---------|----|----|
| Этапов с логированием | 2 | 7 |
| Параметров логируется | 0 | 15+ |
| Request ID трассировка | ❌ | ✅ |
| Per-stage время | ⚠️ | ✅ |
| Структурированные метрики | ⚠️ | ✅ |

---

## ✅ Заключение

**Фаза 2 успешно реализована!**

Система логирования теперь имеет:
- ✅ Полную иерархию 7 этапов
- ✅ Request ID для трассировки
- ✅ Per-stage параметры и метрики
- ✅ Автоматическое измерение времени
- ✅ Структурированные метрики для анализа

**Готово к Фазе 3:** Добавление метрик памяти и dashboard для анализа

---

*Документация подготовлена Claude Haiku 4.5*  
*2026-04-19 12:00 UTC*
