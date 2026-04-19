# 📚 RAG GigaChat Documentation

Структурированная документация по разработке, компонентам и архитектуре RAG системы.

## 📖 Разделы

### [development/](development/)
Руководства по разработке, тестированию и отладке:
- [hybrid-mode-guide.md](development/HYBRID_MODE_GUIDE.md) — Гибридный режим поиска (dense + sparse)
- [gigachat-local-mode.md](development/GIGACHAT_LOCAL_MODE.md) — Использование локального GigaChat
- [quick-reference.md](development/QUICK_REFERENCE.md) — Шпаргалка по коммандам и API
- [testing.md](development/TESTING.md) — Тестирование (unit, integration, smoke)
- [test-question-guide.md](development/TEST_QUESTION_GUIDE.md) — Примеры тестовых вопросов
- [examples-debug-mode.md](development/examples_debug_mode.md) — Примеры использования debug mode
- [model-comparison.md](development/model-comparison.md) — Сравнение моделей GigaChat
- [logging-guide.md](development/LOGGING_GUIDE.md) — Полный гайд логирования и процесс-майнинга
- [troubleshooting.md](development/TROUBLESHOOTING.md) — Часто задаваемые вопросы и решения

### [components/](components/)
Документация по Streamlit компонентам:
- [overview.md](components/overview.md) — Обзор 5 основных компонентов
- [quick-start.md](components/quick-start.md) — Быстрый старт (30 сек)
- [examples.md](components/examples.md) — Копипаст примеры
- [architecture.md](components/architecture.md) — Архитектура и диаграммы

### [archive/](archive/)
Архивированные сессионные отчёты и исторические документы (для справки):
- SESSION_*.md, PHASE*.md, IMPLEMENTATION_*.md, DEBUG_*.md и др.

---

## 🚀 Быстрый навигатор

| Задача | Документ |
|--------|----------|
| Начать разработку | [development/quick-reference.md](development/QUICK_REFERENCE.md) |
| Добавить компонент | [components/overview.md](components/overview.md) |
| Написать тесты | [development/testing.md](development/TESTING.md) |
| Отладить проблему | [development/troubleshooting.md](development/TROUBLESHOOTING.md) |
| Использовать логирование | [development/logging-guide.md](development/LOGGING_GUIDE.md) |
| Сравнить модели | [development/model-comparison.md](development/model-comparison.md) |

---

## 📂 Полная структура

```
docs/
├── README.md                    # этот файл
├── development/                 # разработка и отладка
│   ├── hybrid-mode-guide.md
│   ├── gigachat-local-mode.md
│   ├── quick-reference.md
│   ├── testing.md
│   ├── test-question-guide.md
│   ├── examples-debug-mode.md
│   ├── model-comparison.md
│   ├── logging-guide.md
│   └── troubleshooting.md
├── components/                  # Streamlit компоненты
│   ├── overview.md
│   ├── quick-start.md
│   ├── examples.md
│   └── architecture.md
└── archive/                     # исторические документы
    ├── SESSION_SUMMARY_*.md
    ├── PHASE*.md
    ├── IMPLEMENTATION_*.md
    ├── DEBUG_*.md
    └── ...
```

---

## 🔗 Важные файлы вне docs/

- [CLAUDE.md](../CLAUDE.md) — Правила проекта для Claude (обязателен в корне)
- [README.md](../README.md) — Основная документация проекта
- [CHANGELOG.md](../CHANGELOG.md) — История изменений по версиям
- [backlog/](../backlog/) — Отслеживание багов и фич (BKL-*)

---

## 📝 Как использовать документацию

1. **Новичок?** Начните с [development/quick-reference.md](development/QUICK_REFERENCE.md)
2. **Развёртываю UI?** Смотрите [components/overview.md](components/overview.md)
3. **Пишу тесты?** Используйте [development/testing.md](development/TESTING.md)
4. **Нужна помощь?** Проверьте [development/troubleshooting.md](development/TROUBLESHOOTING.md)

---

Последнее обновление: 2026-04-19
