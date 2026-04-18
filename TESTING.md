# 🧪 Тестирование RAG Pipeline

## 🚀 Выбор профиля моделей

RAG система поддерживает несколько предустановленных профилей моделей для разных сценариев:

| Профиль | LLM | Размер | Скорость | Для чего |
|---------|-----|--------|----------|----------|
| **production** (default) | Qwen2.5-0.5B | ~1.1 GB | ⚡⚡⚡ | Production, балансный выбор |
| **quality** | TinyLlama-1.1B | ~2.3 GB | ⚡⚡ | Лучшее качество ответов |
| **llama** | Llama-3.2-1B | ~2.1 GB | ⚡⚡ | Новая Meta, оптимизирована для мобильных |
| **testing** | GPT2 | ~370 MB | ⚡⚡⚡⚡ | Быстрые тесты на ноутбуке |
| **ci** | DistilGPT2 | ~350 MB | ⚡⚡⚡⚡⚡ | CI/CD пайплайны (минимальный) |

### Запуск с выбранным профилем

```bash
# Production (default, текущая конфигурация)
python app.py --mode query --query "Что такое RAG?" --documents data/domain_2_Debug/books

# Качество (лучшие ответы, медленнее)
export RAG_MODEL_PROFILE=quality
python app.py --mode query --query "Что такое RAG?" --documents data/domain_2_Debug/books

# Быстрое тестирование (на слабом ПК)
export RAG_MODEL_PROFILE=testing
python app.py --mode query --query "Что такое RAG?" --documents data/domain_2_Debug/books

# CI/CD (суперминимальный)
export RAG_MODEL_PROFILE=ci
pytest tests/integration/test_rag_pipeline_query.py -v

# Llama (новая модель Meta)
export RAG_MODEL_PROFILE=llama
python app.py --mode ui
```

### Просмотр доступных профилей

```python
from src.rag_gigachat.config import print_model_profiles
print_model_profiles()
```

---

## Обзор

Проект содержит два типа тестов для проверки RAG pipeline:

### 1️⃣ **pytest - Интеграционный тест RAGPipeline** 
Файл: `tests/integration/test_rag_pipeline_query.py`

Проверяет:
- ✅ Инициализацию RAGPipeline
- ✅ Загрузку PDF документов
- ✅ Обработку запросов
- ✅ Поиск документов (retrieval)
- ✅ Качество ответов
- ✅ Метрики генерации (время, токены)
- ✅ Параметризованные запросы

**Запуск:**
```bash
# Все тесты
pytest tests/integration/test_rag_pipeline_query.py -v

# Один тест
pytest tests/integration/test_rag_pipeline_query.py::TestRAGPipelineQuery::test_pipeline_initialization -v

# С логированием
pytest tests/integration/test_rag_pipeline_query.py -v -s

# С покрытием (может быть медленнее)
pytest tests/integration/test_rag_pipeline_query.py -v --cov=rag_gigachat
```

**Время выполнения:** ~40-60 сек на первый запуск (загрузка моделей), ~30-40 сек на последующие

**Отметка:** Тест помечен как интеграционный, может быть пропущен с флагом `-m "not integration"`

---

### 2️⃣ **pytest - CLI тесты app.py**
Файл: `tests/integration/test_app_cli.py`

Проверяет:
- ✅ Запуск app.py в режиме query
- ✅ Парсинг аргументов командной строки
- ✅ Корректность вывода
- ✅ Сохранение результата в JSON
- ✅ Обработку ошибок
- ✅ Параметры K для поиска
- ✅ Производительность

**Запуск:**
```bash
# Все CLI тесты
pytest tests/integration/test_app_cli.py -v

# Один тест
pytest tests/integration/test_app_cli.py::TestAppCLI::test_app_query_mode_basic -v

# Все интеграционные тесты
pytest tests/integration/ -v
```

**Время выполнения:** ~180 сек на тест (загрузка моделей + генерация ответа)

---

### 3️⃣ **Ручное тестирование - app.py CLI**

**Базовый запрос:**
```bash
python app.py --mode query \
  --query "Что такое RAG?" \
  --documents data/domain_2_Debug/books \
  --k 3
```

**Сохранение результата в JSON:**
```bash
python app.py --mode query \
  --query "Что такое RAG?" \
  --documents data/domain_2_Debug/books \
  --k 5 \
  --output result.json
```

**С другим методом поиска:**
```bash
python app.py --mode query \
  --query "Как работает RAG?" \
  --documents data/domain_2_Debug/books \
  --retrieval_type dense \
  --k 3
```

**Помощь:**
```bash
python app.py --help
```

---

## 📊 Примеры запросов для тестирования

Данные находятся в: `data/domain_2_Debug/books/Глоссарий RAG.pdf`

### Рекомендуемые запросы:
1. **"Что такое RAG?"** - базовый вопрос о RAG
2. **"Как RAG работает?"** - процесс работы
3. **"Какие компоненты у RAG?"** - архитектура
4. **"RAG и нейросети"** - связь с ML
5. **"Преимущества RAG"** - бенефиты системы

---

## 🐛 Отладка

### Включить DEBUG логирование:
Отредактируйте `src/rag_gigachat/config.py`:
```python
logging_config = LoggingConfig(
    log_level="DEBUG",  # вместо "INFO"
    ...
)
```

### Очистить кэш PDF:
```bash
rm -rf data/cache/
rm -rf data/vectorstore/
```

### Явная перезагрузка документов:
В коде используйте `force_reload=True`:
```python
pipeline.load_from_pdf_directory_with_metadata(
    directory, 
    force_reload=True
)
```

---

## 📈 Метрики производительности

| Операция | Время | Примечание |
|----------|-------|-----------|
| Загрузка embedding модели | ~5 сек | Первый запуск |
| Загрузка LLM модели (Qwen) | ~10 сек | Первый запуск |
| Загрузка PDF документов | ~2 сек | 1 PDF файл (Глоссарий) |
| Поиск документов (retrieval) | ~0.5 сек | 3-5 документов |
| Генерация ответа | ~20-60 сек | Зависит от модели/CUDA |
| **Итого** | **~40-80 сек** | На один запрос |

---

## 🎯 Критерии успешного теста

✅ **Инициализация pipeline** - модели загружены, no errors  
✅ **Загрузка документов** - vector_store_initialized=True  
✅ **Поиск работает** - найдено ≥ 1 документа  
✅ **Ответ адекватен** - содержит релевантные ключевые слова  
✅ **Время разумно** - < 120 сек на один запрос  
✅ **Нет ошибок** - exit code = 0  

---

## 📝 Структура тестов

```
tests/
├── integration/
│   ├── __init__.py
│   ├── test_rag_pipeline_query.py    # pytest: RAGPipeline API
│   └── test_app_cli.py               # pytest: app.py CLI
├── conftest.py
└── ...
```

---

## 🚀 Быстрый старт тестирования

```bash
# 1. Проверить что все работает (быстро)
pytest tests/integration/test_rag_pipeline_query.py::TestRAGPipelineQuery::test_pipeline_initialization -v

# 2. Запустить app.py вручную
python app.py --mode query --query "Что такое RAG?" --documents data/domain_2_Debug/books

# 3. Запустить все тесты (долго)
pytest tests/integration/ -v
```

---

## ⚙️ Специальные флаги

### pytest флаги:
- `-v` - verbose output
- `-s` - show print statements
- `--tb=short` - краткий traceback
- `-x` - остановиться на первой ошибке
- `-k "keyword"` - запустить тесты по ключевому слову

### app.py флаги:
- `--mode` - режим работы (query/ui)
- `--query` - текст запроса
- `--documents` - папка с PDF
- `--k` - количество документов
- `--retrieval_type` - тип поиска (dense/sparse/hybrid)
- `--output` - путь для сохранения JSON

---

**Последнее обновление:** 2026-04-18  
**Статус:** ✅ Работает полностью
