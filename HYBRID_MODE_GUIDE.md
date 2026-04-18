# Гибридный режим работы Offline/Online

## Обзор

Система автоматически работает в режиме оффлайн по умолчанию:
- `HF_HUB_OFFLINE=1` включен
- Модели загружаются из кэша Hugging Face
- Если модель отсутствует → **автоматически временно переходит онлайн**, скачивает модель, затем возвращается в оффлайн

## Инициализация в коде

### 1. **Автоматическое переключение при загрузке LLM:**

```python
from rag_gigachat.core.llm_manager import LLMManager

# Моделька скачается автоматически если её нет в кэше
llm_manager = LLMManager(model_name="gpt2")
llm = llm_manager.get_llm()
```

**Что происходит:**
1. `load_local_model()` вызывает `check_and_download_model("gpt2")`
2. Проверяет кэш — если модель есть → используется из кэша
3. Если модель отсутствует:
   - Временно отключается `HF_HUB_OFFLINE`
   - Модель скачивается
   - `HF_HUB_OFFLINE` включается обратно

### 2. **Автоматическое переключение при инициализации эмбеддингов:**

```python
from rag_gigachat.core.vector_store import VectorStoreManager

# Эмбеддинги скачаются автоматически если их нет в кэше
vector_manager = VectorStoreManager(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_type="huggingface"
)
```

**Что происходит:**
- Аналогично загрузке LLM: проверка кэша → скачивание если нужно → восстановление оффлайн-режима

## Явное использование API

### Проверка наличия модели в кэше:

```python
from rag_gigachat.core.model_downloader import is_model_cached

# Просто проверить кэш (без скачивания)
if is_model_cached("gpt2"):
    print("✅ Модель в кэше")
else:
    print("❌ Модель отсутствует")
```

### Убедиться в доступности модели:

```python
from rag_gigachat.core.model_downloader import check_and_download_model

# Проверить кэш, скачать если нужно
if check_and_download_model("gpt2"):
    print("✅ Модель доступна")
else:
    print("❌ Ошибка загрузки")
```

### Проверка состояния оффлайн-режима:

```python
from rag_gigachat.core.model_downloader import is_offline_mode_enabled

if is_offline_mode_enabled():
    print("Режим: оффлайн")
else:
    print("Режим: онлайн")
```

## Полный workflow в приложении

```python
from rag_gigachat.core.llm_manager import LLMManager
from rag_gigachat.core.vector_store import VectorStoreManager
from rag_gigachat.core.rag_pipeline import RAGPipeline

# Инициализация всех компонентов
# Модели скачиваются автоматически при первом запуске, потом берутся из кэша

llm_manager = LLMManager(model_name="gpt2", model_type="local")
vector_manager = VectorStoreManager(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_type="huggingface"
)

rag = RAGPipeline(
    llm_manager=llm_manager,
    vector_store_manager=vector_manager
)

# Теперь можно работать в полностью оффлайн-режиме
answer = rag.query("Как дела?")
```

## Переменные окружения

| Переменная | Значение | Описание |
|-----------|---------|-----------|
| `HF_HUB_OFFLINE` | `1` (по умолчанию) | Оффлайн-режим для Hugging Face Hub |
| `TRANSFORMERS_OFFLINE` | `1` (рекомендуется) | Оффлайн-режим для transformers |
| `HF_HOME` | `~/.cache/huggingface` | Директория кэша Hugging Face (можно переопределить) |

### Установка в .env:

```bash
# .env файл
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=/custom/cache/path  # Опционально
```

## Первый запуск (требует интернет)

При первом запуске приложение скачает необходимые модели:

```bash
# Интернет нужен только для первого запуска
python app.py --mode ui
# Модели скачаются в ~/.cache/huggingface/
# Примерный размер:
#   - gpt2: ~350 MB
#   - all-MiniLM-L6-v2: ~90 MB
#   - Итого: ~450 MB
```

Последующие запуски работают полностью в оффлайне.

## Обработка ошибок

Если модель не может быть скачана:

```
❌ Ошибка скачивания модели gpt2: ...
RuntimeError: Не удалось загрузить модель gpt2. 
Проверьте интернет-соединение или скачайте модель вручную.
```

### Решение:

```bash
# Вручную скачать модель в оффлайн-режиме (требует интернет)
HF_HUB_OFFLINE=0 python -c \
  "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"

# Или использовать huggingface-cli
huggingface-cli download gpt2
```

## Отладка

Включить подробное логирование:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from rag_gigachat.core.model_downloader import check_and_download_model
check_and_download_model("gpt2")
# Выведет подробные логи процесса скачивания
```

## Примечания

- Функции вызываются автоматически при инициализации моделей
- Состояние оффлайн-режима правильно восстанавливается даже при ошибках
- Система совместима с любыми моделями Hugging Face
- Кэш может быть очищен: `rm -rf ~/.cache/huggingface/`
