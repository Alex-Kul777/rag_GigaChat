# Debug-режим: примеры использования

## 🚀 Быстрый старт

### Включение debug-режима (ускорение в 7 раз на загрузку)

```bash
# Способ 1: Переменная окружения (самый быстрый)
export RAG_DEBUG_MODE=true
python app.py --mode ui
```

**Результат:**
- Загрузка: ~2 сек вместо ~15 сек (7x ускорение! ⚡)
- Генерация: ~1 сек вместо ~3 сек (3x ускорение! ⚡)
- Память: ~400 MB вместо ~1.1 GB (2.75x экономия! 💾)

---

## 💻 Примеры использования

### Пример 1: Быстрая отладка UI

```bash
export RAG_DEBUG_MODE=true
python app.py --mode ui
# Браузер откроется почти мгновенно!
```

### Пример 2: CLI запрос с отладкой

```bash
RAG_DEBUG_MODE=true python app.py --mode query --query "Что такое машинное обучение?"
```

Вывод:
```
🐛 DEBUG MODE: Using fast model facebook/opt-125m
🔍 DEBUG: Загрузка модели facebook/opt-125m
✅ DEBUG: Модель успешно загружена (text-generation)

⏱️ Ответ получен за 1.2 сек
📝 Ответ: Machine learning is a subset of artificial intelligence...
```

### Пример 3: Бенчмарк скорости отладки

```python
#!/usr/bin/env python3
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_gigachat.config import debug_config
from rag_gigachat.core.llm_manager import LLMManager

# Включить debug-режим
debug_config.debug_mode = True

print("🐛 DEBUG MODE BENCHMARK\n")

# Загрузка модели
start = time.time()
llm_manager = LLMManager(model_type="local")
llm = llm_manager.get_llm()
load_time = time.time() - start

# Генерация ответа
prompt = "What is AI in one sentence?"
start = time.time()
response = llm.invoke(prompt)
gen_time = time.time() - start

print(f"\n✨ Результаты:")
print(f"  Загрузка: {load_time:.2f}s ⚡")
print(f"  Генерация: {gen_time:.2f}s ⚡")
print(f"  Ответ: {str(response)[:100]}...")
```

### Пример 4: Переключение между режимами

```python
#!/usr/bin/env python3
from rag_gigachat.config import debug_config, model_config
from rag_gigachat.core.llm_manager import LLMManager

# 🐛 DEBUG: Быстрая модель (125M параметров)
debug_config.debug_mode = True
llm_debug = LLMManager(model_type="local").get_llm()
print(f"DEBUG модель: {model_config.llm_model_name}")
response_debug = llm_debug.invoke("Who are you?")
print(f"Ответ: {response_debug[:50]}...\n")

# 🚀 PRODUCTION: Полномощная модель (500M параметров)
debug_config.debug_mode = False
import importlib
import rag_gigachat.core.llm_manager
importlib.reload(rag_gigachat.core.llm_manager)

llm_prod = LLMManager(model_type="local").get_llm()
print(f"PRODUCTION модель: {model_config.llm_model_name}")
response_prod = llm_prod.invoke("Who are you?")
print(f"Ответ: {response_prod[:50]}...")
```

### Пример 5: Разработка с горячей перезагрузкой

```python
#!/usr/bin/env python3
# Отладочный скрипт для итеративной разработки
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_gigachat.config import debug_config
from rag_gigachat.core.rag_pipeline import RAGPipeline

# Включить debug для быстрого цикла
debug_config.debug_mode = True

# Инициализировать pipeline (быстро!)
pipeline = RAGPipeline()

# Итеративная разработка
test_queries = [
    "Какой язык программирования выбрать?",
    "Как оптимизировать производительность?",
    "Что такое машинное обучение?",
]

for query in test_queries:
    print(f"\n📝 Вопрос: {query}")
    response = pipeline.query(query)
    print(f"✅ Ответ: {response[:100]}...")
    # Просто изменяй код и перезапускай - очень быстро!
```

### Пример 6: Тестирование с разными профилями

```python
#!/usr/bin/env python3
import os
from rag_gigachat.config import ModelConfig

# Список всех доступных профилей
print("📦 Доступные профили:\n")
print(ModelConfig.list_profiles())

# Переключение между профилями
profiles = ["production", "quality", "testing", "ci"]

for profile in profiles:
    os.environ["RAG_MODEL_PROFILE"] = profile
    config = ModelConfig()
    print(f"\n{profile.upper()}: {config.llm_model_name}")
```

---

## 🔧 Конфигурация

### Текущие значения debug-режима

```python
# В src/rag_gigachat/config.py
debug_config.debug_mode = True  # Включить
debug_config.debug_model_name = "facebook/opt-125m"  # 125M параметров
```

### Характеристики debug-модели

- **Название:** facebook/opt-125m
- **Параметры:** 125 млн (vs 500M в production)
- **Размер на диске:** ~250 MB
- **Использование памяти:** ~400 MB
- **Время загрузки:** ~1.5 сек
- **Время генерации на CPU:** ~1 сек на 20 токенов
- **Поддержка языков:** Английский + отчасти другие

### Изменение debug-модели

Если нужна другая модель:

```python
# Для еще большей скорости (82 MB)
debug_config.debug_model_name = "distilgpt2"

# Для лучшего качества (680 MB)
debug_config.debug_model_name = "facebook/opt-350m"

# Для русского языка (только embeddings, не генерация)
debug_config.debug_model_name = "DeepPavlov/rubert-base-cased"
```

---

## 🎯 Когда использовать debug-режим

### ✅ ИСПОЛЬЗУЙТЕ для:

```bash
# Отладки pipeline
RAG_DEBUG_MODE=true python -m pytest tests/ -v

# Быстрого прототипирования
RAG_DEBUG_MODE=true python app.py --mode ui

# Локального тестирования на слабой машине
RAG_DEBUG_MODE=true python experiments/run_experiment.py

# Цикла разработки: код → тест → итерация
RAG_DEBUG_MODE=true python app.py --mode query --query "тест"
```

### ❌ НЕ ИСПОЛЬЗУЙТЕ для:

```bash
# Production развертывания (качество хуже)
python app.py --mode ui  # Без RAG_DEBUG_MODE

# Окончательной оценки качества RAG
RAG_MODEL_PROFILE=quality python app.py  # Используйте better model

# Демонстрации пользователю (качество слабое)
python app.py --mode ui  # Используйте полную модель
```

---

## 🐛 Отладка проблем

### Проблема: "Модель не загружается"

```bash
# Решение: Очистить кэш моделей
rm -rf ~/.cache/huggingface/hub/

# Повторить загрузку
RAG_DEBUG_MODE=true python app.py --mode query --query "test"
```

### Проблема: "Медленнее чем ожидается"

```bash
# Проверить, что debug-режим включен
export RAG_DEBUG_MODE=true
.venv/bin/python -c "from rag_gigachat.config import debug_config; print(f'Debug: {debug_config.debug_mode}')"

# Проверить, какая модель загружается
RAG_DEBUG_MODE=true .venv/bin/python -c "from rag_gigachat.core.llm_manager import LLMManager; m = LLMManager(); print(m.model_name)"
```

### Проблема: "Ошибка памяти на слабой машине"

```python
# Решение: Уменьшить max_new_tokens еще больше
from rag_gigachat.config import model_config
model_config.max_new_tokens = 30  # Вместо 150

# Или использовать distilgpt2 (самая легкая)
debug_config.debug_model_name = "distilgpt2"  # 82 MB
```

### Проблема: "GPU не используется в debug-режиме"

```bash
# Debug-режим намеренно использует CPU для стабильности
# Если нужен GPU:
export CUDA_VISIBLE_DEVICES=0
RAG_DEBUG_MODE=true python app.py --mode ui
```

---

## 📊 Сравнение производительности

```
╔════════════════════════╦═════════════╦═════════════╗
║       Метрика          ║ Production  ║   DEBUG     ║
╠════════════════════════╬═════════════╬═════════════╣
║ LLM параметры          ║   500M      ║    125M     ║
║ Размер модели          ║   ~1 GB     ║   ~250 MB   ║
║ Память при запуске     ║   ~1.1 GB   ║   ~400 MB   ║
║ Время загрузки         ║   ~15 сек   ║   ~1.5 сек  ║ ⚡⚡⚡
║ Время генерации        ║   ~3 сек    ║   ~1 сек    ║ ⚡⚡
║ Качество ответов       ║   ⭐⭐⭐⭐⭐  ║   ⭐⭐⭐     ║
║ Использование памяти   ║   1.1 GB    ║   400 MB    ║ 💾
╚════════════════════════╩═════════════╩═════════════╝
```

---

## 🎓 Дополнительная информация

### Почему OPT-125m?

1. **Скорость:** 4x меньше параметров = 4x быстрее на CPU
2. **Память:** ~400 MB вместо ~1.1 GB
3. **Совместимость:** Поддерживает стандартный pipeline text-generation
4. **Качество:** Достаточно для отладки и тестирования

### Как переключить на другой язык?

```python
# Если нужна поддержка русского в debug-режиме:
# Опция 1: Использовать embedding-модель + LLM
debug_config.debug_model_name = "facebook/opt-125m"  # Английская LLM
embedding_config.model_name = "intfloat/multilingual-e5-small"  # Многоязычные embeddings

# Опция 2: Полная многоязычная модель (медленнее)
debug_config.debug_model_name = "xlm-roberta-base"  # Медленнее но многоязычная
```

### Подсказки для оптимизации

```bash
# Смотреть метрики использования памяти в реальном времени
watch -n 1 'ps aux | grep python | grep debug'

# Профилировать загрузку моделей
python -m cProfile -s cumtime test_debug_mode.py | head -30

# Проверить GPU память (если есть GPU)
nvidia-smi -l 1
```
