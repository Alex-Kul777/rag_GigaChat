# 🚀 Процесс релиза RAG GigaChat

Ты - менеджер релизов. Выполни все шаги автоматически, следуя семантическому версионированию и best practices.

> **🔄 Обозначение фонового выполнения**: `🔄` перед шагом означает — выполни эту операцию в фоновом режиме (используй `run_in_background: true` в Bash tool). Это позволяет параллельно выполнять несколько операций, не ждая результата каждого шага.

---

## 📦 В очереди для следующего релиза

### ✅ Выполнено (коммиты fcac4a4 и 8fc44fb):
- ✨ **Multi-stage Dockerfile с GPU поддержкой** (ARG CUDA_VERSION)
- ✨ **docker-compose.yml с device_requests** (GPU auto-detection)
- ✨ **setup.sh с проверкой NVIDIA драйверов** (3-уровневая валидация)
- ✨ **README.md Quick Start** (2 минуты для новичков)
- ✨ **.env.example шаблон** (для конфигурации)
- ✨ **docs/TROUBLESHOOTING.md** (расширённое руководство)

### 🎯 Категория: MINOR (новые функции + улучшения)
Версия: **v1.7.0** (было v1.6.0)

### 📝 Для CHANGELOG.md добавить:
```markdown
### Added
- **Docker GPU support** — Multi-stage Dockerfile с опциональной CUDA
- **setup.sh** — Автоматическая проверка NVIDIA драйверов
- **.env.example** — Шаблон конфигурации для новичков

### Improved
- **README.md** — Quick Start раздел (2 минуты)
- **docker-compose.yml** — device_requests + runtime fallback
- **Installation docs** — Три варианта установки (Docker/Local/Auto)
- **Troubleshooting** — Новый docs/TROUBLESHOOTING.md с решениями
```

---

## 📋 Правила

- **Версионирование**: Auto-detect на основе типов коммитов (breaking → major, feat → minor, fix → patch)
- **Валидация**: Проверь clean working tree, пройди тесты, обнови CHANGELOG/README перед коммитом
- **Git теги**: Каждый релиз должен иметь аннотированный tag вида `v1.2.3`
- **Сообщения**: Используй формат `release: v{VERSION} - {DESCRIPTION}` с датой
- **Откат**: Если шаг не пройден, остановись и сообщи об ошибке (не продолжай)
- **Фоновое выполнение**: команды с `🔄` — используй `run_in_background: true` в Bash tool

## 🔄 Процесс

### Фаза 1️⃣: Предварительные проверки

```bash
🔄 # 1.1 Проверка статуса репозитория
git status
# ОШИБКА если: untracked важные файлы, unstaged изменения
# WARN если: файлы в .gitignore но не должны быть там

🔄 # 1.2 Анализ коммитов для версионирования
git log --oneline -20
# Проанализируй: есть ли "breaking:"? → major bump
#               есть ли "feat:"?      → minor bump
#               иначе                 → patch bump

🔄 # 1.3 Запуск тестов (обязательно)
pytest tests/ -v --tb=short
# ОШИБКА если: тесты не прошли

🔄 # 1.4 Проверка кода (опционально)
black --check src/
# WARN если: style issues
```

### Фаза 2️⃣: Определение версии

```bash
🔄 # Прочитай CHANGELOG.md и найди текущую версию
# CURRENT_VERSION = первая строка "## [X.Y.Z]"

# Определи НОВУЮ версию на основе коммитов:
# IF "breaking:" найден     → MAJOR bump (1.5.1 → 2.0.0)
# ELSE IF "feat:" найден    → MINOR bump (1.5.1 → 1.6.0)
# ELSE IF "fix:" найден     → PATCH bump (1.5.1 → 1.5.2)
# ELSE                       → БЕЗ РЕЛИЗА (пропусти)

# Выведи результат одной строкой:
# "🔢 Версия: 1.5.1 → 1.6.0 (MINOR - есть feature commits)"
```

### Фаза 3️⃣: Обновление документации

```bash
# 3.1 Обнови CHANGELOG.md в начало файла
# 🔄 Добавь новый раздел:
## [NEW_VERSION] - YYYY-MM-DD

### Added
- **файлы** — описание новых функций

### Improved
- **файлы** — описание улучшений

### Fixed
- **файлы** — описание багфиксов
#
# Выведи: "📋 CHANGELOG.md обновлён"

# 3.2 Обнови README.md
# 🔄 После секции "Overview" добавь раздел:
## ✨ What's New in vX.Y.Z
- Feature 1 description
- Feature 2 description
- ...
#
# Выведи: "📖 README.md обновлён"

# 3.3 Обнови README_RU.md
# 🔄 Синхронизируй с README.md (добавь русский перевод нового раздела)
# Выведи: "📖 README_RU.md синхронизирован"
```

### Фаза 4️⃣: Коммит и теги

```bash
🔄 # 4.1 Стадируй только файлы документации
git add CHANGELOG.md README.md README_RU.md
# НИКОГДА: .egg-info/, __pycache__/, .pyc

🔄 # 4.2 Создай коммит с версией
git commit -m "release: v{NEW_VERSION} - {BRIEF_DESCRIPTION} [{DATE}]"
# Пример: "release: v1.6.0 - PDF диагностика и улучшения [2026-04-17]"
# Выведи: "✅ Коммит: {COMMIT_HASH}"

🔄 # 4.3 Создай аннотированный tag
git tag -a v{NEW_VERSION} -m "Release v{NEW_VERSION} - {DESCRIPTION}"
# Выведи: "🏷️ Tag: v{NEW_VERSION}"
```

### Фаза 5️⃣: Публикация

```bash
🔄 # 5.1 Запуши коммит и теги
git push origin main
git push origin --tags
# Выведи: "🚀 Опубликовано в GitHub"

🔄 # 5.2 Проверь CI/CD
# Выведи: "✅ Релиз завершён" или "⚠️ Проверь GitHub Actions"
```

## ✅ Финальный чек-лист

```
✅ Рабочее дерево clean
✅ Все тесты прошли
✅ CHANGELOG обновлён
✅ README.md обновлён
✅ README_RU.md синхронизирован
✅ Коммит создан с версией
✅ Git tag создан
✅ Запушено в origin/main и origin/vX.Y.Z
```

## 🔧 Откат при ошибке

Если какой-то шаг не прошёл:
```bash
# Откати последний коммит (если ещё не запушен)
git reset --soft HEAD~1

# Удали локальный tag (если создан)
git tag -d v{NEW_VERSION}

# Восстанови CHANGELOG, README
git checkout HEAD -- CHANGELOG.md README.md README_RU.md
```

## 📊 Примеры версионирования

| Коммиты | Текущая | Новая | Тип |
|---------|---------|-------|-----|
| `breaking: ...` | 1.5.0 | 2.0.0 | MAJOR |
| `feat: PDF diagnostics` | 1.5.0 | 1.6.0 | MINOR |
| `fix: FAISS validation` | 1.5.0 | 1.5.1 | PATCH |
| Только `docs:`, `chore:` | 1.5.0 | 1.5.0 | NO RELEASE |

## 🎯 Использование в Claude Code

**Когда вызывать релиз:**
```
user: Выполни релиз по инструкции из .claude/release-prompt-auto.md
claude: Читает этот файл, выполняет все фазы в правильном порядке, использует run_in_background для 🔄 операций
```

**Что происходит:**
1. Claude выполняет фазу 1-5 последовательно
2. Команды с 🔄 запускаются в фоне (не ждём результата сразу)
3. После каждой фазы — краткий отчёт (1 строка)
4. При ошибке — остановка и предложение отката

## 🎯 Команды для быстрого запуска

```bash
🔄 # Полный релиз в один раз
python app.py --version  # провери что версия синхронизирована

# Или вручную по фазам
🔄 git log --oneline -10  # 1. Проверь коммиты
🔄 pytest tests/ -v        # 2. Запусти тесты
# ... редактируй CHANGELOG и README вручную
🔄 git commit ...          # 3. Коммит
🔄 git tag -a ...         # 4. Tag
🔄 git push ...           # 5. Push
```

## 💡 Tips

- Каждый раз перед релизом проверь что все тесты зелёные: `pytest tests/ -v`
- Версия должна быть в CHANGELOG.md, README.md и обновляться при каждом релизе
- Git теги (v1.2.3) служат точками восстановления в истории проекта
- Если релиз вышел неудачно, используй откатные команды из раздела "Откат при ошибке"
