# RAG GigaChat Project Rules

## 🎯 Project Overview
RAG system with GigaChat LLM, FAISS vector search, and Streamlit UI for Russian document processing.

## 🏗️ Architecture
- **UI Layer**: Streamlit (`ui_streamlit.py`)
- **RAG Core**: LangChain + LangGraph (`rag_core.py`)
- **Embeddings**: GigaChat embeddings
- **Vector Store**: FAISS
- **Search**: Dense + BM25 hybrid
- **Evaluation**: RAGAS metrics + custom metrics

## 📁 Key Files
- `app.py` - Unified entry point
- `rag_core.py` - Main RAG pipeline
- `ui_streamlit.py` - Streamlit UI components
- `config.py` - Centralized configuration
- `experiment.py` - Experiment runner
- `evaluator.py` - Quality metrics
- `token_counter.py` - Token usage tracking
- `excel_reporter.py` - Excel report generation

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
4. Test locally: `streamlit run ui_streamlit.py`
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

> Техническое применение правил — файл `.claude/settings.local.json`.
> Этот раздел описывает политику; менять оба файла синхронно.

## Project Boundary
- **Root**: `/home/kap/projects/rag_GigaChat/`
- Всё внутри root — авто-одобрение для чтения и правки не-критичных файлов
- За пределами root — всегда спрашивать

## Critical Files — Always Ask
- `config.py`, `.env` — конфигурация и секреты
- `CLAUDE.md` — правила
- `requirements.txt` — зависимости
- `.git/config`, `.gitignore`

## Command Policy

| Категория | Команды | Действие |
|-----------|---------|----------|
| 🟢 Авто | `ls`, `cat`, `grep`, `awk`, `wc`, `find`, `head`, `tail` | Выполнять |
| 🟢 Авто | `python3`, `.venv/bin/python`, `.venv/bin/pip`, `.venv/bin/pytest` | Выполнять |
| 🟢 Авто | `git status/log/diff/show/branch/add/commit` | Выполнять |
| 🟢 Авто | `pip install` (в venv), `gh run/pr/issue` | Выполнять |
| 🟡 Спросить | `git push`, `rm`, `mv`, `cp`, `chmod`, `sed -i` | Подтвердить |
| 🔴 Запрещено | `rm -rf /`, `sudo`, `curl \| bash`, `eval $(curl...)` | Не выполнять |

## File Deletions
Удаление любых файлов — **всегда спрашивать**, даже если явно попросили.