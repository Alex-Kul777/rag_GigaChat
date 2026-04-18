# 🚀 Гибридный режим — Быстрый справочник

## По умолчанию работает автоматически

✅ **LLMManager** — скачивает модели автоматически
✅ **VectorStoreManager** — скачивает эмбеддинги автоматически
✅ **Оффлайн по умолчанию** — `HF_HUB_OFFLINE=1`

```python
# Просто используйте — всё скачается автоматически
from rag_gigachat.core.llm_manager import LLMManager
llm = LLMManager(model_name="gpt2").get_llm()
```

## API для явного использования

### Функции в `core/model_downloader.py`:

| Функция | Описание | Возвращает |
|---------|---------|-----------|
| `is_model_cached(name)` | Есть ли модель в кэше? | `bool` |
| `is_offline_mode_enabled()` | Оффлайн-режим включен? | `bool` |
| `check_and_download_model(name)` | Проверить/скачать модель | `bool` |
| `set_offline_mode(bool)` | Включить/отключить оффлайн | `Tuple[was_set, old_value]` |
| `get_hf_cache_dir()` | Директория кэша HF | `Path` |

### Примеры:

```python
from rag_gigachat.core.model_downloader import *

# Проверить кэш
if is_model_cached("gpt2"):
    print("В кэше!")

# Убедиться в доступности
if check_and_download_model("sentence-transformers/all-MiniLM-L6-v2"):
    print("Готово!")

# Проверить режим
if is_offline_mode_enabled():
    print("Оффлайн")
```

## Переменные окружения

```bash
# .env файл (опционально)
HF_HUB_OFFLINE=1              # По умолчанию включен
TRANSFORMERS_OFFLINE=1        # Рекомендуется включить
HF_HOME=/path/to/cache        # Пользовательская директория кэша
```

## Workflow

```
Первый запуск (требует интернет)
├─ python app.py
├─ Модели скачиваются в ~/.cache/huggingface/
├─ App переходит в оффлайн-режим
└─ Готово!

Последующие запуски (работают оффлайн)
├─ python app.py
├─ Модели загружаются из кэша
└─ Интернет не требуется!
```

## Обработка ошибок

```
❌ Ошибка: "Не удалось загрузить модель XXX"

Решение:
1. Проверьте интернет-соединение
2. Очистите кэш: rm -rf ~/.cache/huggingface/
3. Запустите снова

Или скачайте вручную:
HF_HUB_OFFLINE=0 python -c \
  "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
```

## Размеры моделей

| Модель | Размер | Описание |
|--------|--------|---------|
| `gpt2` | 350 MB | LLM (production профиль) |
| `distilgpt2` | 300 MB | Легкий LLM (CI профиль) |
| `sentence-transformers/all-MiniLM-L6-v2` | 90 MB | Эмбеддинги |
| **Итого** | **~450 MB** | На первый запуск |

## Тестирование

```bash
# Запустить примеры
python example_hybrid_mode.py

# Проверить синтаксис
python -m py_compile src/rag_gigachat/core/model_downloader.py
```

## Файлы реализации

```
src/rag_gigachat/core/
├─ model_downloader.py        ✨ Основной модуль
├─ llm_manager.py             ✅ Интегрирован
└─ vector_store.py            ✅ Интегрирован

Документация:
├─ HYBRID_MODE_GUIDE.md       📖 Полное руководство
├─ IMPLEMENTATION_SUMMARY.md  📋 Технические детали
└─ QUICK_REFERENCE.md         ⚡ Этот файл
```

## TL;DR

✅ **Просто работает** — ничего не нужно менять  
✅ **Автоматический** — скачивает при необходимости  
✅ **Безопасный** — восстанавливает состояние при ошибках  
✅ **Оффлайн** — работает без интернета после первого запуска  

---

Полная документация: `HYBRID_MODE_GUIDE.md`  
Примеры: `example_hybrid_mode.py`  
Техдокументация: `IMPLEMENTATION_SUMMARY.md`
