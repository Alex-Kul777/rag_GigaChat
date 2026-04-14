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

## Project Boundaries
- **Project Root**: /home/kap/projects/rag_GigaChat/
- **Allowed Paths**: All paths under project root
- **Restricted Paths**: Any paths outside project root

## Auto-Approve Operations (within project)

### Read Operations - Always Allowed
- Reading any project files: .py, .md, .json, .txt, .yaml
- Listing directories: ls, find, tree
- Viewing git status: git status, git log, git diff
- Reading configuration: .env, config.py, settings.py

### Write Operations - Auto Approve
- Creating new files in: src/, tests/, docs/, .claude/
- Modifying existing Python files (except critical modules)
- Adding comments and docstrings
- Code refactoring within same file

### Execute Operations - Auto Approve
- Running Python scripts: python3 *.py
- Running tests: pytest, unittest
- Git operations: git add, git commit (push requires confirmation)
- Installing packages: pip install (within venv)

## Require Confirmation

### Write Operations - Ask First
- Modifying .git/config, .gitignore
- Changing requirements.txt (ask for major changes)
- Deleting files (always ask)
- Modifying .env files

### Execute Operations - Ask First
- rm -rf commands
- sudo operations
- Docker commands
- Commands affecting system outside venv
- Git push to main/master

## Critical Files - Always Ask
- config.py - sensitive configuration
- .env - environment variables
- CLAUDE.md - rules file
- requirements.txt - dependencies

## Project-Specific Rules

### Allowed Operations

Reading files - always allowed:
cat, head, tail, less, grep, awk, sed -n
python3 -c (read-only operations)
git status, git log, git diff

Writing files - allowed within project:
echo  >> file.py (appending)
sed -i 's/old/new/g' file.py (in-place replacement)
python3 script.py (running project scripts)

Package management - allowed in venv:
pip install package
pip uninstall package

### Disallowed Operations

Never allow these:
rm -rf /
sudo rm -rf
chmod 777
mv /etc/ /home/
curl | bash
eval "$(curl...)"

## Auto-Response Template

When operating inside project:
- Auto-approve read operations
- Auto-approve write operations to non-critical files
- Auto-approve refactoring changes
- Ask only for:
  1. Deleting files
  2. External operations
  3. Critical file modifications
  4. Git push to remote

## Safety Check

Before any operation, Claude should verify:
1. Is target path inside project? (Yes -> auto-approve read/write)
2. Is operation potentially destructive? (No -> proceed)
3. Are we modifying critical files? (No -> proceed)
4. Is this a system command? (No -> proceed)

## File Type Permissions

### Auto-approve for extensions:
.py - Python source files
.md - Markdown documentation
.txt - Text files
.json - Configuration files
.yaml - YAML configuration
.yml - YAML configuration
.sh - Shell scripts (if in project)
.rst - ReStructuredText docs

### Ask for extensions:
.env - Environment variables
.pem - Private keys
.key - Private keys
.crt - Certificates
.db - Database files

## Directory Permissions

### Auto-approve within:
/home/kap/projects/rag_GigaChat/
/home/kap/projects/rag_GigaChat/src/
/home/kap/projects/rag_GigaChat/tests/
/home/kap/projects/rag_GigaChat/docs/
/home/kap/projects/rag_GigaChat/.claude/

### Ask before accessing:
/home/kap/.ssh/
/etc/
/var/
/root/
/home/kap/.aws/
/home/kap/.config/

## Command Categories

### Green (Auto-approve):
cat, head, tail, less, grep, awk, find, ls
git status, git log, git diff, git add
python3, pytest, pip install
echo, touch, mkdir (within project)

### Yellow (Ask first):
sed -i, git commit -m, git push
rm, rmdir, mv, cp
chmod, chown
pip uninstall

### Red (Never auto-approve):
rm -rf, sudo, su
docker rm, docker rmi
kill, pkill
systemctl, service
curl (with pipe to bash)

## Version Control Rules

### Auto-approve:
git status
git log
git diff
git add file.py
git commit -m "short message"

### Ask before:
git commit -m (long or complex messages)
git push origin main
git push origin master
git rebase
git merge
git reset --hard

## Testing Rules

### Auto-approve:
pytest tests/
python -m unittest
python3 script.py --test

### Ask before:
pytest --cov (external reporting)
pytest --benchmark (long running)
python3 script.py --delete-test-data

## Documentation Rules

### Auto-approve:
Updating docstrings
Adding comments
Creating .md files
Updating README.md (minor changes)

### Ask before:
Deleting documentation files
Major restructure of docs/
Changing API documentation format

## Emergency Override

If you need to bypass these rules for a specific task:
1. Explain why the operation is necessary
2. Get explicit user approval
3. Document the exception

Example:
"I need to modify .git/config to update remote URL. This is outside normal permissions because..."

## Rule Priority

1. Safety rules (highest priority)
2. Project boundary rules
3. File type rules
4. Operation type rules
5. General auto-approve (lowest priority)

## Logging

Claude should log all auto-approved operations to:
.claude/operations.log

Format:
[timestamp] [operation] [path] [status]

Example:
[2024-04-14 10:30:00] [READ] [rag_core.py] [AUTO_APPROVED]
[2024-04-14 10:31:00] [WRITE] [test_token.py] [AUTO_APPROVED]