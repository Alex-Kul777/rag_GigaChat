# 📊 Анализ качества логов для Process Mining

**Дата анализа:** 2026-04-19  
**Режим:** DEBUG (facebook/opt-125m)  
**Запрос:** "Что такое RAG?"  
**Общее время выполнения:** 26.82 сек

---

## 📈 Сводка результатов проверки

| Критерий | Статус | Комментарий |
|----------|--------|-----------|
| **Временные метки** | ❌ ОТСУТСТВУЮТ | Нет timestamp в формате ГГГГ-ММ-ДД ЧЧ:ММ:СС.ммс |
| **Уровни логирования** | ⚠️ ЧАСТИЧНО | Смешаны `logging.DEBUG`, print, и эмодзи-маркеры |
| **START/END маркеры** | ❌ ОТСУТСТВУЮТ | Нет явной границы начала/конца каждого этапа |
| **Идентификаторы процессов** | ⚠️ ЧАСТИЧНО | Есть [graph], [retrieval], но неполные |
| **Последовательность шагов** | ✅ ЛОГИЧНА | Этапы следуют в правильном порядке |
| **Правдоподобность времен** | ✅ OK | 26.82 сек - правдоподобно для загрузки + обработки |
| **Дублирование логов** | ❌ КРИТИЧНО | Ответ зациклен (18 повторений "Что такое RAG?") |
| **Единообразность формата** | ❌ НИЗКАЯ | Смешанные форматы: print(), logging, эмодзи, [DEBUG] |

---

## 🔍 Детальный анализ логов

### Этап 1: Инициализация модели
```
🐛 DEBUG MODE: facebook/opt-125m (fast, 125M params)  ← OK
📦 LLMManager: OFFLINE mode, model_type=local          ← OK
logging_config.log_level: DEBUG                        ← OK
```
**Проблемы:**
- ❌ Нет timestamp
- ❌ Нет [START] маркера
- ❌ Время загрузки не логировано (видно только в прогресс-баре)

**Длительность (расчётная):** ~2-3 сек (из других запусков в debug)

---

### Этап 2: Загрузка документов
```
[DEBUG] Найдено PDF файлов: 3
[DEBUG] PDF файл: Глубокое обучение_short.pdf
[DEBUG] PDF файл: Deep_Learning_by_Ian_Goodfellow_2016_short.pdf
[DEBUG] PDF файл: Глоссарий RAG.pdf

Обработка PDF файлов: 100%|██████████| 3/3 [00:07<00:00, 2.36s/it]

[DEBUG] Получено документов: 124
```
**Проблемы:**
- ❌ Нет timestamp для каждого лога
- ❌ Нет [LOAD_DOCS START/END] маркеров
- ❌ Прогресс-бар не в логах, а в stdout (неструктурировано)
- ⚠️ Информация о cache пропущена в выводе

**Длительность:** 7 сек (видно из прогресс-бара)  
**Документов загружено:** 124

---

### Этап 3: Обработка текстов / Chunking
```
Обработка текстов: 100%|██████████| 119/119 [00:00<00:00, 170348.87it/s]
```
**Проблемы:**
- ❌ Нет информации о параметрах: chunk_size, overlap
- ❌ Нет timestamp
- ❌ Нет [CHUNKING] маркера
- ❌ Дефакто 119 документов = 119 чанков? Нет информации

**Длительность:** <1 сек  
**Обработано чанков:** 119

---

### Этап 4: Инициализация индекса
```
🔍 [init] Начало обработки запроса        ← Неточный текст (не про индекс)
🔍 [init] Индекс инициализирован
```
**Проблемы:**
- ❌ Нет timestamp
- ❌ Нет информации о типе индекса (FAISS? какой тип метрики?)
- ❌ Нет информации о размерности embedding
- ❌ Нет информации о количестве векторов в индексе
- ⚠️ Маркер `[init]` недостаточно точен

**Отсутствующие данные:**
- Тип индекса: FAISS (но это не явно)
- Размерность: ? (768? 1024? не известно)
- Количество векторов: 119 (расчётное на основе чанков)
- Время построения индекса: не логировано

---

### Этап 5: Загрузка модели LLM
```
🔍 DEBUG: Загрузка модели facebook/opt-125m
🔍 DEBUG: dtype: torch.float32, device: cuda
🔍 DEBUG: Создаем pipeline...

Loading weights: 100%|██████████| 197/197 [00:01<00:00, 107.43it/s]

✅ DEBUG: Модель успешно загружена (text-generation)
🔍 GPU память используется: 0.48 GB
```
**Проблемы:**
- ❌ Нет timestamp
- ⚠️ Прогресс-бар в stdout, не в логах
- ❌ Нет [GENERATION START] маркера

**Положительные моменты:**
- ✅ Информация о dtype и device (cuda)
- ✅ GPU память логирована

**Время загрузки:** 1 сек  
**Память:** 0.48 GB

---

### Этап 6: Поиск релевантных документов (RETRIEVAL)
```
🔍 [retrieval] Поиск релевантных документов...
🔍 [retrieval] Найдено 5 документов
🔍 [retrieval] Топ: '33. Latency / Throughput...' (score=0.7800)
```
**Проблемы:**
- ❌ Нет timestamp
- ❌ Нет [RETRIEVAL START/END] маркеров
- ❌ Время выполнения retrieval не логировано
- ❌ Параметры поиска не логированы (k=5, метрика=cosine?, similarity_threshold=?)
- ⚠️ Выводится только топ документ, остальные 4 скрыты

**Положительные моменты:**
- ✅ Найдено k=5 документов
- ✅ Логируется score (0.7800)
- ✅ Полная информация в итоговой таблице

**Параметры:**
- k: 5 ✅
- Метрика сходства: unknown ❌
- Threshold: unknown ❌
- Время выполнения: unknown ❌

---

### Этап 7: Генерация ответа (GENERATION)
```
🔍 [generation] Генерация ответа с помощью LLM...
🔍 [generation] Ответ получен
```
**КРИТИЧЕСКАЯ ПРОБЛЕМА:** ❌ **ЗАЦИКЛИВАНИЕ МОДЕЛИ**
```
Answer: Что такое RAG? Что такое RAG? Что такое RAG? 
        Что такое RAG? Что такое RAG? Что такое RAG?
        [повторяется 18 раз]
```

**Проблемы:**
- ❌ Нет timestamp
- ❌ Нет [GENERATION START/END] маркеров
- ❌ Время генерации не логировано
- ❌ Параметры генерации не логированы (max_tokens, temperature, top_p и т.д.)
- ❌ Длина prompt не логирована
- ❌ Количество токенов логировано только итого (295), но не по этапам

**Критическая ошибка:**
- 🔴 Модель зациклилась вместо генерации нормального ответа
- Причина: возможно, неправильная конфигурация max_tokens или проблема с декодированием

**Параметры:**
- Модель: facebook/opt-125m ✅
- Max tokens: unknown ❌
- Temperature: unknown ❌
- Всего токенов в ответе: 295 (но бесполезные из-за зацикливания)

---

### Этап 8: Итоговые метрики
```
⏱️  Время обработки: 26.82 сек
🔢 Токенов в ответе: 295
```
**Проблемы:**
- ❌ Нет детализации по этапам
- ❌ Нет памяти (только GPU была логирована)
- ❌ Время латентности retrieval не раздельно

**Имеющаяся информация:**
- Общее время: 26.82 сек ✅
- Токены: 295 ✅

**Отсутствующая информация:**
- Время по этапам (Load=?, Chunking=?, Index=?, Retrieval=?, Generation=?)
- Пиковое использование памяти CPU и GPU
- Latency retrieval отдельно
- Latency generation отдельно

---

## 🎯 Таблица анализа для Process Mining

| Этап | Timestamp | Длительность (ms) | START | END | Документов | Статус | Примечание |
|------|-----------|------------------|-------|-----|------------|--------|-----------|
| **INIT** | ❌ | ~2000 | ❌ | ❌ | - | ✅ | Загрузка модели opt-125m |
| **LOAD_DOCS** | ❌ | 7000 | ❌ | ❌ | 3 PDF | ✅ | 124 документа загружено |
| **CHUNKING** | ❌ | <1000 | ❌ | ❌ | 119 чанков | ✅ | Параметры неизвестны |
| **EMBEDDING** | ❌ | ? | ❌ | ❌ | 119 vec | ⚠️ | Не логировано отдельно |
| **INDEX** | ❌ | ? | ❌ | ❌ | 119 | ⚠️ | Не логировано отдельно |
| **RETRIEVAL** | ❌ | ? | ❌ | ❌ | 5/119 | ✅ | k=5, score=0.78 |
| **GENERATION** | ❌ | ? | ❌ | ❌ | 1 | ❌ | **ЗАЦИКЛИВАНИЕ!** |
| **RESPONSE** | ❌ | 26820 | ❌ | ❌ | - | ⚠️ | Общее время |

---

## 📋 Проверочный лист: START/END маркеры

```
⏹️ LOAD_DOCS
   ❌ [LOAD_DOCS START] - ОТСУТСТВУЕТ
   [Логирование PDF файлов...]
   ❌ [LOAD_DOCS END] - ОТСУТСТВУЕТ

⏹️ CHUNKING
   ❌ [CHUNKING START] - ОТСУТСТВУЕТ
   [Обработка текстов...]
   ❌ [CHUNKING END] - ОТСУТСТВУЕТ

⏹️ EMBEDDING
   ❌ [EMBEDDING START] - ОТСУТСТВУЕТ
   [Неявно выполняется в retrieval setup]
   ❌ [EMBEDDING END] - ОТСУТСТВУЕТ

⏹️ INDEX
   ❌ [INDEX START] - ОТСУТСТВУЕТ
   [INIT Индекс инициализирован]
   ❌ [INDEX END] - ОТСУТСТВУЕТ

⏹️ RETRIEVAL
   ✅ [retrieval] Поиск релевантных документов...  ← Похоже на START
   [Поиск выполняется]
   ✅ [retrieval] Найдено 5 документов            ← Похоже на END

⏹️ GENERATION
   ✅ [generation] Генерация ответа с помощью LLM...  ← Похоже на START
   [Генерация выполняется]
   ✅ [generation] Ответ получен                      ← Похоже на END
```

---

## 🚨 Критические проблемы

### 1. ❌ ОТСУТСТВИЕ ВРЕМЕННЫХ МЕТОК (CRITICAL)
**Проблема:** Нет timestamp в любом логе  
**Последствия для PM:** Невозможно построить timeline, анализировать последовательность, определять bottlenecks  
**Решение:** Добавить `%(asctime)s` в формат логирования

### 2. 🔴 ЗАЦИКЛИВАНИЕ ГЕНЕРАЦИИ (CRITICAL)
**Проблема:** Модель повторяет вопрос 18 раз вместо ответа  
**Последствия:** Неправильный ответ, бесполезные токены  
**Решение:** Проверить `max_new_tokens`, `do_sample`, `top_p`, `temperature`

### 3. ❌ ОТСУТСТВИЕ START/END МАРКЕРОВ (HIGH)
**Проблема:** Невозможно определить точные границы этапов  
**Последствия для PM:** Неточная реконструкция процесса  
**Решение:** Добавить явные [STEP_NAME START] и [STEP_NAME END] маркеры

### 4. ❌ ОТСУТСТВИЕ СТРУКТУРИРОВАННОГО ФОРМАТА (HIGH)
**Проблема:** Логи в разных форматах (print, logging, прогресс-бары)  
**Последствия:** Сложно парсить и анализировать автоматически  
**Решение:** Использовать JSON логирование или единый структурированный формат

### 5. ❌ НЕ ЛОГИРУЕТСЯ ВРЕМЯ ОТДЕЛЬНЫХ ЭТАПОВ (HIGH)
**Проблема:** Только общее время, нет детализации  
**Последствия для PM:** Невозможно найти bottleneck  
**Решение:** Логировать `elapsed_time_ms` для каждого этапа

---

## 💡 Рекомендации по улучшению логирования

### 🎯 Краткосрочные улучшения (1-2 часа)

#### 1. Добавить временные метки
```python
# ДО:
logger.info("Загрузка документов начата")

# ПОСЛЕ:
import logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)
logger.info("📦 [LOAD_DOCS START] Загрузка PDF файлов из data/domain_2_Debug/books")
```

#### 2. Добавить START/END маркеры
```python
logger.info("🧪 [LOAD_DOCS START] Найдено PDF файлов: 3")
# ... обработка ...
logger.info("✅ [LOAD_DOCS END] Загружено документов: 124, время: 7.2s")
```

#### 3. Логировать ключевые параметры
```python
logger.debug(f"📋 [CHUNKING] chunk_size={chunk_size}, overlap={overlap}, total_chunks={len(chunks)}")
logger.debug(f"🔢 [EMBEDDING] model={embedding_model}, dim={embed_dim}, vectors={num_vectors}")
logger.debug(f"🔍 [RETRIEVAL] k={k}, metric=cosine, threshold={similarity_threshold}")
```

#### 4. Использовать единообразный формат
```
TIMESTAMP | LEVEL | STAGE | MESSAGE | METRICS
2026-04-19 10:00:01.234 | INFO | LOAD_DOCS START | Found 3 PDFs | file_count=3
2026-04-19 10:00:08.456 | INFO | LOAD_DOCS END | 124 documents | duration_ms=7222, doc_count=124
```

### 🏗️ Долгосрочные улучшения (4-8 часов)

#### 5. Структурированное логирование (JSON)
```python
import json
import logging
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "stage": getattr(record, "stage", "UNKNOWN"),
            "message": record.getMessage(),
            "metrics": getattr(record, "metrics", {}),
        }
        return json.dumps(log_obj, ensure_ascii=False)

logger.info("Start", extra={"stage": "LOAD_DOCS", "metrics": {"file_count": 3}})
# Выход: {"timestamp": "2026-04-19T10:00:01Z", "level": "INFO", "stage": "LOAD_DOCS", ...}
```

#### 6. Встроенные таймеры для каждого этапа
```python
import time

class PipelineTimer:
    def __init__(self):
        self.stages = {}
    
    def start(self, stage_name):
        self.stages[stage_name] = {"start": time.time()}
    
    def end(self, stage_name, metrics=None):
        if stage_name in self.stages:
            elapsed = time.time() - self.stages[stage_name]["start"]
            logger.info(
                f"✅ [{stage_name} END]",
                extra={
                    "stage": stage_name,
                    "metrics": {"duration_ms": int(elapsed * 1000), **(metrics or {})}
                }
            )

# Использование:
timer = PipelineTimer()
timer.start("LOAD_DOCS")
documents = load_documents()
timer.end("LOAD_DOCS", {"doc_count": len(documents)})
```

#### 7. Трассировка распределённого запроса (Request ID)
```python
import uuid
from contextvars import ContextVar

request_id = ContextVar('request_id', default=None)

def process_query(query: str):
    rid = str(uuid.uuid4())[:8]
    request_id.set(rid)
    
    logger.info(f"[{rid}] [PIPELINE START] Query: '{query}'")
    try:
        logger.info(f"[{rid}] [LOAD_DOCS START]")
        # ...
        logger.info(f"[{rid}] [RETRIEVAL] Found k=5")
        # ...
        logger.info(f"[{rid}] [PIPELINE END] Total time: 26.82s")
    except Exception as e:
        logger.error(f"[{rid}] [PIPELINE ERROR] {e}")
```

#### 8. Метрики per-stage в единой таблице
```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class StageMetrics:
    stage_name: str
    timestamp: str
    duration_ms: int
    status: str  # OK, ERROR, TIMEOUT
    input_size: int = 0
    output_size: int = 0
    memory_mb: float = 0.0
    error_msg: str = None

class PipelineMetricsCollector:
    def __init__(self):
        self.metrics: List[StageMetrics] = []
    
    def add_stage(self, metrics: StageMetrics):
        self.metrics.append(metrics)
        logger.info(
            f"📊 [{metrics.stage_name}] "
            f"time={metrics.duration_ms}ms, "
            f"in={metrics.input_size}, "
            f"out={metrics.output_size}, "
            f"mem={metrics.memory_mb:.1f}MB"
        )
    
    def to_dataframe(self):
        """Для анализа в pandas/Excel"""
        import pandas as pd
        return pd.DataFrame([
            {
                "Stage": m.stage_name,
                "Timestamp": m.timestamp,
                "Duration (ms)": m.duration_ms,
                "Status": m.status,
                "Input Size": m.input_size,
                "Output Size": m.output_size,
                "Memory (MB)": m.memory_mb,
            }
            for m in self.metrics
        ])
```

#### 9. Исправить зацикливание генерации
```python
# В llm_manager.py load_local_model():
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
    device_map="auto" if torch.cuda.is_available() else "cpu",
    max_new_tokens=512,           # ← ДОБАВИТЬ
    do_sample=True,                # ← ДОБАВИТЬ
    temperature=0.7,               # ← ДОБАВИТЬ
    top_p=0.9,                     # ← ДОБАВИТЬ
    repetition_penalty=1.2,        # ← ДОБАВИТЬ ДЛЯ ПРЕДОТВРАЩЕНИЯ ЗАЦИКЛИВАНИЯ
    no_repeat_ngram_size=3,        # ← ДОБАВИТЬ
)
```

---

## 📊 Итоговая таблица рекомендаций

| # | Рекомендация | Приоритет | Время | Файл | Импакт |
|---|---|---|---|---|---|
| 1 | Добавить timestamp в логи | 🔴 CRITICAL | 0.5ч | logging_config.py | ВЫСОКИЙ |
| 2 | Добавить [START]/[END] маркеры | 🔴 CRITICAL | 1ч | rag_pipeline.py | ВЫСОКИЙ |
| 3 | Логировать параметры этапов | 🟡 HIGH | 1.5ч | rag_pipeline.py | СРЕДНИЙ |
| 4 | Логировать время per-stage | 🟡 HIGH | 2ч | rag_pipeline.py | СРЕДНИЙ |
| 5 | Исправить зацикливание генерации | 🔴 CRITICAL | 0.5ч | llm_manager.py | ВЫСОКИЙ |
| 6 | Структурировать логи (JSON) | 🟢 MEDIUM | 4ч | logging_config.py | СРЕДНИЙ |
| 7 | Добавить Request ID трассировку | 🟢 MEDIUM | 2ч | rag_pipeline.py | СРЕДНИЙ |
| 8 | Метрики в DataFrame | 🟢 MEDIUM | 2ч | rag_pipeline.py | СРЕДНИЙ |

---

## 🎯 План действий для реализации

### **Фаза 1: Критические исправления (2-3 часа)**
1. ✅ Добавить `%(asctime)s` в формат логирования
2. ✅ Добавить [START]/[END] маркеры в 6 основных этапов
3. ✅ Исправить зацикливание генерации в llm_manager.py
4. ✅ Добавить логирование параметров (chunk_size, k, model и т.д.)

### **Фаза 2: Метрики и структурирование (4-6 часов)**
5. ✅ Добавить PipelineTimer класс для измерения каждого этапа
6. ✅ Логировать время per-stage в единообразном формате
7. ✅ Добавить Request ID для трассировки
8. ✅ Добавить StageMetrics dataclass

### **Фаза 3: Продвинутые возможности (6-8 часов)**
9. ✅ Реализовать JSONFormatter для структурированных логов
10. ✅ Добавить метрики памяти per-stage
11. ✅ Реализовать экспорт в DataFrame/Excel для анализа
12. ✅ Добавить dashboarding логов в Streamlit

---

## 📋 Следующие шаги

1. **Немедленно:** Исправить зацикливание генерации (max_new_tokens, repetition_penalty)
2. **Сегодня:** Добавить timestamp и START/END маркеры
3. **На неделю:** Добавить метрики per-stage
4. **На месяц:** Полностью переписать логирование на структурированный формат

---

**Готово к реализации? Хотите начать с Фазы 1?**
