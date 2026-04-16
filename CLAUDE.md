# RAG GigaChat Project Rules

## 🎯 Project Overview
RAG system with GigaChat LLM, FAISS vector search, and Streamlit UI for Russian document processing.

## 🏗️ Architecture (src/ package structure)
- **Core**: `src/rag_gigachat/core/` - RAG pipeline, vector store, LLM manager
- **Data**: `src/rag_gigachat/data/` - Document loading with OCR support
- **UI**: `src/rag_gigachat/ui/` - Streamlit components
- **Reporting**: `src/rag_gigachat/reporting/` - Evaluation and Excel reports
- **Config**: `src/rag_gigachat/config.py` - Centralized configuration
- **Models**: `src/rag_gigachat/models.py` - Data models

## 📁 Key Files
- `app.py` - Unified entry point (root)
- `src/rag_gigachat/core/rag_pipeline.py` - Main RAG pipeline
- `src/rag_gigachat/ui/streamlit_app.py` - Streamlit UI
- `src/rag_gigachat/config.py` - Centralized configuration
- `src/rag_gigachat/experiment.py` - Experiment runner
- `src/rag_gigachat/reporting/evaluator.py` - Quality metrics
- `src/rag_gigachat/token_counter.py` - Token usage tracking
- `src/rag_gigachat/reporting/excel_reporter.py` - Excel report generation

## 🔧 Development Rules

### Code Style
- Use type hints for all functions
- Add docstrings in Russian/English
- Maximum line length: 100 chars
- Use 4 spaces for indentation

### Error Handling
- Use try/except with specific exceptions
- Log errors with `logger.error()`
- Add debug prints with `print(f"🔍 DEBUG: ...")`

### Logging
- Use module-level logger: `logger = logging.getLogger(__name__)`
- Log levels: DEBUG for details, INFO for progress, ERROR for failures
- Add emojis for better readability in console

### Git Commits
- Format: `type: description`
- Types: feat, fix, docs, refactor, test, chore
- Example: `feat: add BM25 retriever support`

## 🚫 What NOT to Do
- Don't commit `.env` files with API keys
- Don't commit `data/cache/` or `data/vectorstore/`
- Don't push large files (>10MB)
- Don't hardcode API keys in source code

## 🔄 Development Workflow
1. Pull latest changes: `git pull origin main`
2. Create feature branch: `git checkout -b feature/name`
3. Make changes with proper logging
4. Test locally:
   - UI: `python app.py --mode ui` или `streamlit run src/rag_gigachat/ui/streamlit_app.py`
   - CLI: `python app.py --mode query`
   - Tests: `pytest tests/ -v`
5. Commit with clear message
6. Push and create PR

## 📊 Testing Requirements
- Test PDF loading with small files first
- Verify embeddings with GigaChat API
- Check token limits (max 500 per request)
- Validate metrics calculation

## 🐛 Common Issues & Solutions
| Issue | Solution |
|-------|----------|
| Tokens limit exceeded | Reduce `chunk_size` in config.py |
| GigaChat connection | Check API key in .env |
| Memory error | Decrease `batch_size` in rag_core.py |
| Cache issues | Clear `data/cache/` directory |

## 📝 Documentation
- Keep README.md updated
- Keep README_RU.md updated (README.md in Russian)
- Add docstrings for new functions
- Update CHANGELOG.md for changes
- Document new configuration parameters

## 🤖 Claude-Specific Instructions
- When suggesting code, include error handling
- Add logging statements for debugging
- Use existing patterns from the codebase
- Consider token limits for GigaChat API
- Prioritize backward compatibility

# Claude Permissions Configuration

> Техническое применение правил — файл `.claude/settings.local.json` (в ~/.claude/).
> Этот раздел описывает политику; менять оба файла синхронно.

## Project Boundary
- **Root**: `/home/kap/projects/rag_GigaChat/`
- Всё внутри root — авто-одобрение для чтения и правки не-критичных файлов
- За пределами root — всегда спрашивать

## Critical Files — Always Ask Before Editing
- `CLAUDE.md` — правила проекта
- `src/rag_gigachat/config.py`, `config.py` — конфигурация и пути
- `.env` — секреты и API ключи
- `requirements.txt` — зависимости проекта
- `.git/config`, `.gitignore` — git конфигурация

## Command Policy

| Категория | Команды | Действие |
|-----------|---------|----------|
| 🟢 Авто | `ls`, `cat`, `grep`, `awk`, `wc`, `find`, `head`, `tail` | Выполнять |
| 🟢 Авто | `python3`, `.venv/bin/python`, `.venv/bin/pip`, `.venv/bin/pytest` | Выполнять |
| 🟢 Авто | `git status/log/diff/show/branch/add/commit` | Выполнять |
| 🟢 Авто | `pytest tests/`, `PYTHONPATH=src python -m pytest` | Выполнять |
| 🟡 Спросить | `git push`, `rm`, `mv`, `cp`, `chmod`, `sed -i` | Подтвердить |
| 🔴 Запрещено | `rm -rf /`, `sudo`, `curl \| bash`, `eval $(curl...)` | Не выполнять |

## Settings Configuration
Разрешения сохранены в `~/.claude/settings.local.json`:
- **auto_approve**: read, bash_commands (safe patterns)
- **ask_first**: git push, rm/mv/cp, critical files
- **forbidden**: dangerous system commands
- **venv**: auto-use `.venv/` для Python/pip
- **testing**: pytest configuration и coverage threshold

## File Deletions
Удаление любых файлов — **всегда спрашивать**, даже если явно попросили.

## Virtual Environment (venv)
- **Path**: `.venv/` в корне проекта
- **Auto-use**: Все `python`, `pip`, `pytest` команды автоматически используют venv
- **No activation needed**: CLI автоматически использует `.venv/bin/python` и `.venv/bin/pip`

### Auto-Approve Commands in venv:
```bash
.venv/bin/python -m pytest tests/           # Запуск тестов
.venv/bin/python app.py --mode ui          # Запуск приложения
.venv/bin/pip install <package>            # Установка зависимостей
PYTHONPATH=src .venv/bin/python -m pytest  # Тесты с путями
```

## Testing Configuration
- **Framework**: pytest (конфиг в `pytest.ini`)
- **Test paths**: `tests/` с подпапками `unit/` и `integration/`
- **Coverage threshold**: 54% (обязательно)
- **Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- **Run tests**: `pytest tests/ -v --cov=rag_gigachat`

### Auto-Approve Test Commands:
```bash
pytest tests/ -v                            # Все тесты verbose
pytest tests/unit/ -v                       # Только unit тесты
.venv/bin/pytest tests/ --cov=rag_gigachat # С покрытием
PYTHONPATH=src .venv/bin/pytest tests/     # С явным путём
```