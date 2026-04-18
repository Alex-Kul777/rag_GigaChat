# Реализация гибридного режима Offline/Online

## ✅ Что реализовано

### 1. **Новый модуль `src/rag_gigachat/core/model_downloader.py`**

Основные функции:

```python
is_model_cached(model_name) → bool
```
- Проверяет наличие модели в кэше Hugging Face
- Использует `try_to_load_from_cache()` для надежной проверки

```python
is_offline_mode_enabled() → bool
```
- Проверяет, включен ли оффлайн-режим (`HF_HUB_OFFLINE=1`)

```python
set_offline_mode(offline: bool) → Tuple[bool, str]
```
- Включает/отключает оффлайн-режим
- Возвращает предыдущее состояние для восстановления
- Правильно обрабатывает случаи, когда переменная была установлена/не установлена

```python
check_and_download_model(model_name) → bool
```
- **Главная функция**: проверяет кэш → скачивает если нужно → восстанавливает оффлайн-режим
- Использует `snapshot_download()` для надежного скачивания
- Обрабатывает ошибки и восстанавливает состояние в finally блоке

```python
get_hf_cache_dir() → Path
```
- Получает директорию кэша HF (из `HF_HOME` или `~/.cache/huggingface`)

### 2. **Интеграция в `llm_manager.py`**

```python
# В load_local_model():
if not check_and_download_model(self.model_name):
    raise RuntimeError(f"Не удалось загрузить модель {self.model_name}...")
```

- Вызывается **перед** загрузкой модели
- Гарантирует, что модель доступна
- Правильно обрабатывает ошибки

### 3. **Интеграция в `vector_store.py`**

```python
# В _init_embeddings():
if not check_and_download_model(self.embedding_model):
    raise RuntimeError(f"Не удалось загрузить модель эмбеддингов...")
```

- Вызывается перед инициализацией HuggingFaceEmbeddings
- Гарантирует доступность моделей эмбеддингов
- Не влияет на GigaChat эмбеддинги (они используют API)

## 🔄 Workflow по умолчанию

```
Инициализация приложения
  ↓
HF_HUB_OFFLINE=1 (оффлайн по умолчанию)
  ↓
Загрузка LLM/Embeddings
  ↓
check_and_download_model(model_name)
  ├─ is_model_cached()? 
  │  └─ ДА → используем из кэша, возвращаем True
  └─ НЕТ → 
      ├─ Временно отключаем оффлайн (HF_HUB_OFFLINE=0)
      ├─ snapshot_download() → скачиваем модель
      └─ Восстанавливаем оффлайн (HF_HUB_OFFLINE=1)
  ↓
Модель готова
```

## 📊 Примеры использования

### Автоматическое (встроено в приложение):

```python
from rag_gigachat.core.llm_manager import LLMManager

# Моделька скачается автоматически если нужно
llm = LLMManager(model_name="gpt2").get_llm()
```

### Явное использование API:

```python
from rag_gigachat.core.model_downloader import (
    is_model_cached,
    check_and_download_model,
    is_offline_mode_enabled
)

# Проверить кэш
if is_model_cached("gpt2"):
    print("Модель в кэше")

# Убедиться в доступности
if check_and_download_model("gpt2"):
    print("Модель готова")

# Проверить режим
if is_offline_mode_enabled():
    print("Оффлайн-режим включен")
```

## 🧪 Тестирование

Запустить пример:

```bash
python example_hybrid_mode.py
```

Выведет:
- ✅ Проверку наличия модели в кэше
- ✅ Статус оффлайн-режима
- ✅ Переключение между режимами
- ✅ Управление состоянием

## 🔧 Технические детали

### Обработка ошибок

- **Try-catch блоки** вокруг всех операций с HF Hub
- **Finally блок** для гарантированного восстановления оффлайн-режима
- **Специфичные исключения** (ImportError, RuntimeError)

### Совместимость

- ✅ Работает с любыми моделями Hugging Face
- ✅ Совместима с GigaChat эмбеддингами (они используют API, не затрагиваются)
- ✅ Не требует изменения конфигурации `.env`
- ✅ Назад-совместима с существующим кодом

### Переменные окружения

| Переменная | Значение | Описание |
|-----------|---------|-----------|
| `HF_HUB_OFFLINE` | `1` (по умолчанию) | Оффлайн-режим |
| `HF_HOME` | `~/.cache/huggingface` | Директория кэша |

## 📚 Документация

- `HYBRID_MODE_GUIDE.md` — полное руководство с примерами
- `example_hybrid_mode.py` — исполняемые примеры
- Встроенные docstrings во всех функциях

## 🎯 Ключевые преимущества

1. **Полностью автоматический** — не требует ручного управления
2. **Надежный** — правильно обрабатывает ошибки и восстанавливает состояние
3. **Прозрачный** — пользователь не видит переключение режимов
4. **Экономный** — модели скачиваются только один раз
5. **Расширяемый** — легко добавить новые типы моделей

## ✨ Результаты

**Первый запуск** (требует интернет):
```
python app.py --mode ui
# Скачает модели (~450 MB), потом готово к работе
```

**Последующие запуски** (полностью оффлайн):
```
HF_HUB_OFFLINE=1 python app.py --mode ui
# Все модели берутся из кэша, никакого интернета не нужно
```

---

**Статус**: ✅ Готово к использованию

**Файлы изменены**:
- ✅ `src/rag_gigachat/core/model_downloader.py` (новый)
- ✅ `src/rag_gigachat/core/llm_manager.py` (интегрирован check_and_download_model)
- ✅ `src/rag_gigachat/core/vector_store.py` (интегрирован check_and_download_model)

**Документация**:
- ✅ `HYBRID_MODE_GUIDE.md` (полное руководство)
- ✅ `example_hybrid_mode.py` (примеры)
- ✅ `IMPLEMENTATION_SUMMARY.md` (этот файл)
