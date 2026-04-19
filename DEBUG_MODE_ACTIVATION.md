# 🐛 Debug-режим: Как правильно включить?

**Статус:** 🔴 **Частая ошибка** - пользователи забывают установить `RAG_DEBUG_MODE`

---

## ❌ **ЧТО НЕ РАБОТАЕТ**

### Попытка 1: Просто запустить приложение
```bash
python app.py --mode ui

# ❌ Результат: Используется Qwen (500M параметров) - МЕДЛЕННО!
# Логи показывают:
# [INFO] PRODUCTION-MODE: Используется Qwen/Qwen2.5-0.5B-Instruct (500M параметров)
```

### Попытка 2: Установить флаг в коде
```python
# streamlit_app.py
debug_config.debug_mode = True  # Не работает!

python app.py --mode ui
# ❌ Не поможет - уже инициализирован до вашего кода
```

### Попытка 3: Запустить с флагом
```bash
python app.py --mode ui --debug
# ❌ Нет такого флага! Приложение не знает о нём
```

---

## ✅ **ЧТО РАБОТАЕТ**

### Способ 1: Установить переменную окружения (РЕКОМЕНДУЕТСЯ)
```bash
# Установить переменную
export RAG_DEBUG_MODE=true

# Запустить приложение
python app.py --mode ui

# ✅ Результат: Используется facebook/opt-125m (125M параметров) - БЫСТРО!
# Логи показывают:
# [INFO] DEBUG-MODE ENABLED: Using fast model facebook/opt-125m (125M) instead of Qwen (500M)
# [INFO] ⏱️ Expected: Load ~2-3sec, Generate ~1-2sec, Memory ~400MB
```

### Способ 2: Установить в одной строке
```bash
RAG_DEBUG_MODE=true python app.py --mode ui

# ✅ Работает идентично способу 1
```

### Способ 3: Добавить в .env файл (ДЛЯ ПОСТОЯННЫХ ПРОЕКТОВ)
```bash
# .env
RAG_DEBUG_MODE=true

# Убедитесь, что .env загружается:
python app.py --mode ui

# ✅ При каждом запуске будет активен debug-режим
```

### Способ 4: Установить глобально (НЕ РЕКОМЕНДУЕТСЯ)
```bash
# В ~/.bashrc или ~/.zshrc добавить:
export RAG_DEBUG_MODE=true

# После этого ALL скрипты будут использовать debug-режим
# (может повлиять на другие проекты!)
```

---

## 📊 **Сравнение: Что загружается**

### ❌ БЕЗ debug-режима (по умолчанию)
```
export RAG_DEBUG_MODE=false  # или не установлена
python app.py --mode ui

📦 PRODUCTION-MODE
├─ Модель: Qwen/Qwen2.5-0.5B-Instruct
├─ Параметры: 500 млн (500M)
├─ Размер: ~1 GB
├─ Загрузка: ~15 сек
├─ Генерация: ~3 сек
├─ Память: ~1.1 GB
└─ Качество: ⭐⭐⭐⭐⭐ (отлично)
```

### ✅ С debug-режимом
```
export RAG_DEBUG_MODE=true
python app.py --mode ui

🐛 DEBUG-MODE ENABLED
├─ Модель: facebook/opt-125m
├─ Параметры: 125 млн (125M)
├─ Размер: ~250 MB
├─ Загрузка: ~2-3 сек (7x быстрее!)
├─ Генерация: ~1-2 сек (3x быстрее!)
├─ Память: ~400 MB (2.75x экономнее!)
└─ Качество: ⭐⭐⭐ (достаточно для отладки)
```

---

## 🔧 **Диагностика: Какой режим используется?**

### Способ 1: Посмотреть логи при старте
```bash
export RAG_DEBUG_MODE=true
python app.py --mode ui 2>&1 | grep -i "mode\|debug"

# Ищите эти сообщения:
# ✅ DEBUG-MODE ENABLED: Using fast model facebook/opt-125m
# ❌ PRODUCTION-MODE: Using Qwen/Qwen2.5-0.5B-Instruct
```

### Способ 2: Проверить переменную окружения
```bash
echo "RAG_DEBUG_MODE=$RAG_DEBUG_MODE"

# Результаты:
# RAG_DEBUG_MODE=true   ✅ (debug-режим включен)
# RAG_DEBUG_MODE=false  ❌ (debug-режим отключен)
# RAG_DEBUG_MODE=       ❌ (переменная не установлена)
```

### Способ 3: Посмотреть в UI приложения
При запуске Streamlit приложение покажет:
```
✅ DEBUG-режим активен: Используется facebook/opt-125m (125M параметров)
```
или
```
📦 PRODUCTION-MODE: Используется Qwen (500M параметров, высокое качество)
```

---

## 📋 **Чек-лист: Включить debug-режим**

### ✅ Шаг 1: Установить переменную окружения
```bash
export RAG_DEBUG_MODE=true
```

### ✅ Шаг 2: Проверить установку
```bash
echo "RAG_DEBUG_MODE=$RAG_DEBUG_MODE"
# Должна быть: RAG_DEBUG_MODE=true
```

### ✅ Шаг 3: Запустить приложение
```bash
python app.py --mode ui
```

### ✅ Шаг 4: Проверить логи
```
Ищите в логах:
✅ DEBUG-MODE ENABLED: Using fast model facebook/opt-125m
⏱️  Expected: Load ~2-3sec, Generate ~1-2sec, Memory ~400MB
```

### ✅ Шаг 5: Заметить разницу в скорости!
- Загрузка: ~2-3 сек вместо 15 сек ⚡
- Генерация: ~1-2 сек вместо 3 сек ⚡
- Память: ~400 MB вместо 1.1 GB 💾

---

## 🚨 **Частые ошибки**

### Ошибка 1: "Я установил RAG_DEBUG_MODE, но ничего не изменилось"

**Проблема:** Переменная установлена для одной сессии shell

**Решение:**
```bash
# ❌ Неправильно - установил в одном терминале
terminal1: export RAG_DEBUG_MODE=true

# ❌ Потом запустил в ДРУГОМ терминале - переменная не видна
terminal2: python app.py

# ✅ Правильно - установить И запустить в ОДНОМ терминале
terminal1: export RAG_DEBUG_MODE=true && python app.py
```

### Ошибка 2: "Забываю каждый раз писать export"

**Решение:** Добавить в .bashrc или .zshrc
```bash
# В конце ~/.bashrc добавить:
export RAG_DEBUG_MODE=true

# Сохранить и перезагрузить:
source ~/.bashrc
```

### Ошибка 3: "Хочу отключить debug, но не могу"

**Решение:**
```bash
# Отключить на эту сессию
unset RAG_DEBUG_MODE
python app.py  # Используется production модель

# Или явно установить false
export RAG_DEBUG_MODE=false
python app.py  # Также использует production
```

---

## 🎯 **Быстрые команды**

### Включить debug-режим и запустить
```bash
RAG_DEBUG_MODE=true python app.py --mode ui
```

### Просмотреть логи debug-режима
```bash
RAG_DEBUG_MODE=true python app.py --mode ui 2>&1 | grep -E "DEBUG|MODE|model"
```

### Запустить один раз с debug, остальное в production
```bash
# Session 1: Debug
RAG_DEBUG_MODE=true python app.py --mode ui

# Session 2: Production (в новом терминале)
python app.py --mode ui  # Отдельный процесс, не видит RAG_DEBUG_MODE
```

---

## 📝 **Примеры использования**

### Пример 1: Быстрое прототипирование
```bash
# Нужна максимальная скорость для быстрой итерации
RAG_DEBUG_MODE=true python app.py --mode ui

# Загрузка: 3 сек
# Ответ: 1-2 сек
# Цикл: 4-5 сек 🚀
```

### Пример 2: Финальное тестирование
```bash
# Нужно проверить качество ответов
python app.py --mode ui  # Без debug-режима

# Используется полная Qwen модель
# Качество: максимальное ⭐⭐⭐⭐⭐
```

### Пример 3: Работа на слабой машине
```bash
# Компьютер с ограниченными ресурсами
RAG_DEBUG_MODE=true python app.py --mode ui

# Память: 400 MB (vs 1.1 GB)
# GPU: не требуется (CPU-safe)
# Скорость: приемлемая для отладки
```

---

## 💡 **Помните**

```
🚀 Быстро?   → RAG_DEBUG_MODE=true (debug-режим)
⭐ Качество? → python app.py (production-режим)
📊 Баланс?   → Выбирайте в зависимости от задачи
```

**Основное правило:**
- **Во время разработки** → debug-режим (быстро итерировать)
- **При финальном тестировании** → production-режим (проверить качество)
- **На production** → production-режим (лучшие результаты)
