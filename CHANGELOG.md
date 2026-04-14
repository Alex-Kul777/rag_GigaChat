# Changelog

All notable changes to this project will be documented in this file.

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
