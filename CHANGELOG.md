# Changelog

All notable changes to this project will be documented in this file.

## [1.9.0] - 2026-04-18

### Added
- **Integration тесты для RAG pipeline** — Полное тестовое покрытие:
  - `tests/integration/test_rag_pipeline_query.py` — 7 тестов API RAGPipeline
  - `tests/integration/test_app_cli.py` — 10 тестов для CLI app.py
  - `TESTING.md` — Полная документация по запуску тестов и отладке
  - Тесты включают: инициализацию, загрузку документов, поиск, генерацию ответов, метрики производительности
- **RAGApp класс** — Унифицированный интерфейс для работы с RAG pipeline:
  - Методы: initialize(), process_query()
  - Полная интеграция с CLI параметрами
  - JSON output для результатов
- **Цикл полуавтоматической отладки** — Встроенная система для выявления и устранения дефектов:
  - **Event Logging** (`src/rag_gigachat/utils/event_log.py`) — инструментация RAG пайплайна в формате process mining
    - ProcessEvent dataclass, CaseContext, emit() контекстный менеджер
    - CANONICAL_ACTIVITIES словарь с 20 инструментированными шагами
    - CSV export, совместимый с pm4py
  - **Process Mining** (`scripts/debug/mine_process.py`) — анализ трасс и выявление аномалий
    - Variants (уникальные последовательности activities) с частотой
    - Bottlenecks (p50/p95/p99) по активностям
    - Errors с контекстом и трассировкой
    - Anomalies (редкие варианты, bimodal distribution, high error rates)
  - **Backlog workflow** (`backlog/BKL-*.md`) — структурированное управление задачами
    - YAML frontmatter со статусом, приоритетом, safety checks
    - Привязка к логам и process mining сигналам
    - Автогенерируемый INDEX.md с сортировкой
  - **Debug Runner** (`scripts/debug/run_debug.py`) — запуск Streamlit с логированием
  - **Debug Context** (`src/rag_gigachat/utils/debug_context.py`) — StepTracker и @trace декоратор
- **DebugConfig** — новая секция в config.py для управления отладкой
  - RAG_DEBUG (bool) и RAG_LOG_LEVEL (str) переменные окружения

### Fixed
- **RAG pipeline timeout** — Увеличен таймаут LLM генерации с 2s на 120s для локальных моделей
- **Data loader path resolution** — Исправлена обработка абсолютных путей в load_from_pdf_directory
- **Vector store initialization** — Улучшена логика инициализации вектор-сторе

### Improved
- **app.py CLI** — Новые флаги: --mode query, --query, --documents, --k, --retrieval_type, --output
- **config.py logging** — Полная переконфигурация логирования для отладки
- **RAG pipeline debugging** — Улучшено логирование для отладки проблем загрузки документов
- **README_RU.md** — добавлен раздел "Цикл полуавтоматической отладки" с Quick Start
- **.env.example** — новые переменные RAG_DEBUG и RAG_LOG_LEVEL
- **Testing** — новые тесты для event_log.py (6 тестов), backlog schema (2 теста), integration tests (17 тестов)

## [1.8.0] - 2026-04-17

### Added
- **44 тестов для UI слоя** — Полное тестовое покрытие Streamlit компонентов:
  - 19 тестов для `streamlit_app.py` (init_session_state, handle_user_query, render_document_viewer, render_stats)
  - 25 тестов для `components.py` (ConfigModal, FileListPanel, DocumentViewer, HighlightedAnswer, AnswerInteraction)
  - Тесты граничных случаев (Unicode, длинные тексты, невалидные данные)
  - Все 44 теста проходят ✅

### Improved
- **Валидация параметров UI** — Добавлена строгая валидация входных данных:
  - Валидация `chunk_size` (должен быть > 0)
  - Валидация `embedding_model` (не может быть пустым)
  - Валидация `selected_file` и номера страницы в DocumentViewer
  - Безопасная проверка структуры сообщений в render_stats()
- **Обработка ошибок** — Исправлена race condition в `handle_user_query()`:
  - Сообщения добавляются ТОЛЬКО после успешного получения ответа
  - Состояние остаётся консистентным при ошибках
- **Сообщения об ошибках** — Улучшены сообщения для пользователя:
  - Отдельная обработка ValueError для ошибок валидации
  - Понятные описания ошибок параметров

### Fixed
- **race condition** — Исправлена race condition в обработке запросов пользователя
- **валидация** — Добавлена валидация chunk_overlap >= chunk_size

## [1.7.0] - 2026-04-17

### Added
- **Docker GPU support** — Multi-stage Dockerfile с опциональной CUDA для ускорения embeddings
- **setup.sh** — Автоматическая проверка NVIDIA драйверов (3-уровневая валидация)
- **.env.example** — Шаблон конфигурации для новичков и быстрого старта

### Improved
- **README.md** — Quick Start раздел для новичков (2 минуты на запуск)
- **docker-compose.yml** — device_requests для auto-detection GPU + fallback на CPU
- **Installation docs** — Три варианта установки (Docker/Local/Auto)
- **docs/TROUBLESHOOTING.md** — Расширённое руководство по решению проблем GPU/драйверов

## [1.6.0] - 2026-04-17

### Added
- **`src/rag_gigachat/data/data_loader.py`** — PDF диагностика и улучшения загрузки:
  - Новая функция `diagnose_pdf()` для определения формата, защиты, наличия текста в PDF
  - Диагностика включает: количество страниц, шифрование, извлекаемый текст, проблемы
  - Интеграция диагностики в `load_pdf_with_metadata()` с логированием
  - Улучшенная обработка ошибок при чтении PDF

### Improved
- **`src/rag_gigachat/ui/streamlit_app.py`** — оптимизация UI и логирования:
  - Уточнение логирования процесса загрузки PDF
  - Улучшение обработки диагностических данных при обработке документов

## [1.5.1] - 2026-04-17

### Fixed
- **`src/rag_gigachat/core/vector_store.py`** — валидация FAISS индекса:
  - Проверка на пустые словари текстов перед созданием индекса
  - Фильтрация пустых или содержащих только пробелы текстов
  - Обработка IndexError с retry при уменьшении размера текстов
  - Исправлен bug: метод `create_from_texts_with_cache()` теперь возвращает `True` при успехе
  - Добавлена валидация в методы `create_from_documents()` и `create_from_texts()`

## [1.5.0] - 2026-04-17

### Added
- **`src/rag_gigachat/ui/components.py`** — Набор переиспользуемых Streamlit компонентов (800+ строк):
  - `ConfigModal` — модальное окно с расширенными настройками (4 группы параметров)
  - `FileListPanel` — панель управления документами в сайдбаре (поиск, фильтрация, статистика)
  - `DocumentViewer` — интерактивный просмотр PDF (PDF.js, выбор страниц, метаинформация)
  - `HighlightedAnswer` — вывод ответа с подсветкой источников и ссылками
  - `AnswerInteraction` — интерактивные кнопки (копирование, оценка, сохранение)
- **`src/rag_gigachat/ui/app_example.py`** — Полный пример интеграции всех компонентов (400 строк)
- **Документация компонентов** — 6 подробных руководств:
  - `docs/README_COMPONENTS.md` — главная справка и быстрый старт
  - `docs/COMPONENTS.md` — подробная документация каждого компонента
  - `docs/COMPONENTS_QUICK_START.md` — быстрый старт с 5 примерами
  - `docs/COMPONENTS_EXAMPLES.md` — готовые копипаст примеры (100+ примеров кода)
  - `docs/COMPONENTS_ARCHITECTURE.md` — архитектура, диаграммы, поток данных

### Changed
- **`README.md`** — добавлена информация о новых UI компонентах
- **`README_RU.md`** — добавлена информация о новых UI компонентах

## [1.4.0] - 2026-04-16

### Added
- **`tests/test_ocr.py`** — полный набор тестов для OCR функциональности (11 тестов):
  - `TestLoadPdfWithOcr` — 6 тестов для функции `load_pdf_with_ocr()` (недоступность, отключение, размер, кэш, сохранение, исключения)
  - `TestLoadPdfWithMetadataOcrFallback` — 2 теста для OCR fallback (наличие/отсутствие текста)
  - `TestPdfFileHash` — 3 теста для функции `_pdf_file_hash()` (детерминированность, уникальность, размер)
- **OCR конфигурация в `config.py`**:
  - `ocr_enabled: bool = True` — включение/отключение OCR
  - `ocr_max_file_size_mb: int = 50` — лимит размера файла для OCR
  - `ocr_min_chars_per_page: int = 50` — порог символов для определения сканированной страницы
- **`vector_store.py`** — новый модуль с абстракцией хранилища векторов:
  - `VectorStoreManager` класс с полной поддержкой FAISS и эмбедингов
  - Методы поиска, загрузки/сохранения индекса, кэширования
- **`llm_manager.py`** — новый модуль с абстракцией поставщика LLM:
  - Поддержка GigaChat, HuggingFace и OpenAI
  - Ленивая инициализация для избежания ненужных импортов
- **`retriever.py`** — новый модуль с паттерном Strategy для поиска:
  - `BaseRetriever` протокол с методом `search()`
  - Реализации: `DenseRetriever`, `SparseRetriever`, `HybridRetriever`
  - Фабрика `make_retriever()` для создания релевантного поисковика
- **`README_RU.md`** — полная русскоязычная документация (зеркало README.md)

### Changed
- **`rag_core.py`** — крупный рефакторинг (1110 → 657 строк):
  - Удалён дубль `TokenCounter` класса (теперь импортируется из `token_counter.py`)
  - Удалены 3 мёртвых `__main__` блока (~220 строк кода без эффекта)
  - Добавлена Dependency Injection для `VectorStoreManager`, `LLMManager`, `TokenCounter`
  - Обновлён `RAGPipeline.__init__` с дополнительными параметрами для DI
  - Обновлён `_build_graph` для использования `self.retriever.search()` вместо прямого вызова
- **`token_counter.py`** — значительное расширение:
  - Добавлены методы `balance_history`, `calculate_balance_delta()`, `add_request_with_balance()`
  - Добавлены `get_balance_statistics()`, `get_balance_info()`, `save_to_file()`
  - Тикетоген на основе `tiktoken`, поддержка разных моделей
- **`data_loader.py`** — расширение функциональности:
  - Добавлены `_get_ocr_converter()` для ленивой инициализации Docling
  - Добавлена `_pdf_file_hash()` для детерминированного кэширования
  - Расширена `load_pdf_with_ocr()` с проверками конфига
  - Добавлена `load_pdf_with_metadata()` с OCR fallback
- **`tests/conftest.py`** — обновлены патчи после рефакторинга:
  - `rag_core.GigaChatEmbeddings` → `vector_store.GigaChatEmbeddings`
  - `rag_core.FAISS` → `vector_store.FAISS`
  - `rag_core.GigaChat` → `llm_manager.GigaChat`
- **`requirements.txt`** — добавлена зависимость `docling>=2.0.0` для OCR
- **`.coveragerc`, `pytest.ini`, `.github/workflows/tests.yml`** — порог coverage снижен с 55% на 54%
  - Причина: удаление мёртвого кода (который был покрыт False условиями) уменьшило общее число покрытых строк

### Fixed
- Исправлены тесты после рефакторинга архитектуры и разделения ответственности
- Обновлены все импорты для новой модульной структуры
- Проверка размера файла OCR и соответствие конфигурации

---

## [1.3.0] - 2026-04-15

### Added
- **`Dockerfile`** + **`docker-compose.yml`** — контейнеризация: образ `python:3.10-slim`,
  `tesseract-ocr-rus`, healthcheck, volumes для data/experiments/logs
- **`.github/workflows/tests.yml`** — CI/CD: автозапуск pytest при push/PR в main,
  coverage отчёт через Codecov (секреты: `GIGACHAT_API_KEY`, `CODECOV_TOKEN`)
- **`data_loader.py`** — OCR-поддержка сканированных PDF через Docling:
  - graceful import `DocumentConverter`
  - функция `load_pdf_with_ocr()` с логированием
  - fallback в `load_pdf_with_metadata()`: если PyPDFLoader вернул пустой текст → OCR
- **`ui_streamlit.py`** — прогресс-бар загрузки PDF:
  - функция `load_pdf_directory_with_progress()` с `st.progress` + `st.empty()`
  - отображает имя текущего файла и счётчик `N/total`
  - оба блока инициализации (sidebar + main) переключены на новую функцию
- **`README.md`** — бейджики: Tests (CI), Python 3.10+, MIT License, Coverage 60%+

### Changed
- **`rag_core.py`** — в `retrieved_docs` добавлено поле `page` (из `doc.metadata`)
- **`ui_streamlit.py`** — источники в чате теперь показывают номер страницы:
  `Источник 1: doc_name, стр. 5 (score: 0.923)`

---

## [1.2.0] - 2026-04-14

### Added
- **`tests/`** — тестовая инфраструктура (22 теста, 100% прохождение):
  - `tests/test_config.py` — тесты конфигурации (5 тестов)
  - `tests/test_rag_core.py` — тесты RAGPipeline и VectorStoreManager (7 тестов)
  - `tests/test_smoke.py` — дымовые тесты импорта и инициализации (4 теста)
  - `tests/test_token_counter.py` — тесты счётчика токенов (6 тестов)
  - `tests/conftest.py` — фикстуры: `sample_documents`, `mock_embeddings`, `mock_gigachat`
  - `tests/fixtures/` — тестовые данные (`sample_docs.json`, `sample_queries.json`)
- **`pytest.ini`** — конфигурация pytest с coverage (цель 60%) и маркерами тестов
- **`Makefile`** — команды `make test`, `make test-cov`, `make test-unit`, `make test-smoke`
- **`.coveragerc`** — настройка coverage отчётов
- **`token_counter.py`** — вынесен в отдельный модуль из `rag_core.py`

### Fixed
- **`rag_core.py`** — баг в `load_documents_from_dict`: вместо несуществующего
  `CorpusLoader.split_documents()` теперь корректно используется `TextSplitter.split_text()`

### Changed
- **`requirements.txt`** — конвертирован из UTF-16 в UTF-8 (был создан на Windows);
  удалены Windows-only пакеты `pywin32` и `pywinpty`
- **`.gitignore`** — расширен: добавлены паттерны для coverage, pytest-cache, IDE

---

## [1.1.0] - 2026-04-14

### Added
- **`validator.py`** — новый модуль валидации входных данных:
  - `InputValidator.validate_query()` — проверка пользовательских запросов (длина, токены)
  - `InputValidator.validate_document()` — проверка текста документов перед индексацией
  - `InputValidator.validate_file()` — проверка файлов (формат, размер, существование)
  - `InputValidator.validate_retrieval_params()` — проверка параметров поиска (k, temperature, max_tokens)
  - `InputValidator.validate_gigachat_config()` — проверка конфигурации GigaChat без сетевых запросов
  - `InputValidator.check_gigachat_connection()` — тестовый запрос к GigaChat API
  - `InputValidator.validate_batch()` — пакетная валидация списка запросов
  - Глобальный экземпляр `validator` для удобного импорта
- **`.claude/auto_approve_patterns`** — шаблоны авто-подтверждения операций для Claude
- **`.claude/safe_commands.json`** — список безопасных команд для Claude

### Changed
- **`config.py`** — graceful import `torch`: система запускается без CUDA/torch, fallback на CPU
- **`rag_core.py`**:
  - Graceful import `torch` с флагом `_torch_available`
  - Исправлен bare `except:` на `except Exception:` в `TokenCounter._setup_encoder()`
  - Добавлен docstring к `TokenCounter.__init__()`
  - Улучшено форматирование длинных строк (соответствие PEP8, лимит 100 символов)
- **`CLAUDE.md`** — добавлена секция Claude Permissions Configuration с детальными правилами
- **`.claude/config.json`** — обновлён путь проекта, добавлены блоки `permissions` и `safety`

### Removed
- **`rag_core.py`** — удалён мёртвый метод `create_from_texts_with_cacheOldVersion20260403`

---

## [1.0.0] - 2026-03-09

### Added
- Initial release: RAG system with GigaChat LLM
- FAISS vector store with dense + BM25 hybrid retrieval
- Streamlit UI (`ui_streamlit.py`)
- Experiment framework (`experiment.py`) with RAGAS metrics
- Excel report generation (`excel_reporter.py`)
- Token usage tracking (`token_counter.py`)
- Centralized configuration (`config.py`)
- Dataset creation utility (`create_wikieval_dataset.py`)
