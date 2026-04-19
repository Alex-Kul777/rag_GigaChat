# GigaChat Local Mode - Решение timeout ошибок

## 🐛 Проблема

При запуске RAG системы без API ключа GigaChat возникала ошибка:
```
[Errno 110] Connection timed out
```

**Причина:** Код пытался подключиться к GigaChat API даже когда нужны были локальные модели.

### Где возникала проблема

1. **rag_pipeline.py:79** - дефолтное значение `embedding_type="gigachat"`
   - При создании `RAGPipeline()` без параметров автоматически пытался использовать GigaChat
   - Это влияло на тесты и скрипты, которые использовали дефолт

2. **vector_store.py:80-97** - отсутствовала проверка конфига
   - `_init_embeddings()` не проверял `gigachat_config.enabled` и наличие API ключа
   - Выбрасывал ошибку вместо fallback на локальные модели

3. **streamlit_app.py:48** - неявная зависимость от GigaChat
   - Не указывал явно `embedding_type`, использовал дефолт "gigachat"

---

## ✅ Решение

### 1. **vector_store.py** - Умная инициализация эмбеддингов

```python
def _init_embeddings(self):
    """Инициализация модели эмбеддингов"""
    use_gigachat = (
        self.embedding_type == "gigachat"
        and gigachat_config.api_key
        and gigachat_config.enabled
    )

    if use_gigachat:
        if not GIGACHAT_AVAILABLE:
            raise ImportError("langchain-gigachat не установлен")
        return GigaChatEmbeddings(...)

    logger.info(f"Используем локальные эмбеддинги: {self.embedding_model}")
    return HuggingFaceEmbeddings(...)
```

**Логика:**
- Проверяет ТРИ условия перед попыткой подключиться к GigaChat:
  1. `embedding_type == "gigachat"` - явно требуется GigaChat
  2. `gigachat_config.api_key` - API ключ настроен
  3. `gigachat_config.enabled` - GigaChat включен в конфиге
- Если хотя бы одно условие не выполнено → fallback на HuggingFace

### 2. **rag_pipeline.py:79** - Безопасный дефолт

```python
embedding_type: str = "huggingface"  # Было: "gigachat"
```

**Эффект:**
- Все вызовы `RAGPipeline()` без параметров используют локальные эмбеддинги
- Безопасно для тестов и скриптов
- Явные вызовы с `embedding_type="gigachat"` по-прежнему работают

### 3. **streamlit_app.py:48** - Явная конфигурация

```python
st.session_state[pipeline_key] = RAGPipeline(
    embedding_model=embedding_model,
    embedding_type="huggingface",  # Явно указываем
    llm_type="gigachat",           # LLM может быть GigaChat
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
```

**Комбинация:**
- UI использует локальные embeddings (быстро, без API)
- LLM может использовать GigaChat (если API ключ есть)
- Явный код делает намерение понятным

---

## 🧪 Тестирование

### Результаты debug_query.py

```
✅ RAGPipeline инициализирован (БЕЗ timeout)
✅ 118 документов загружены из PDF
✅ Поиск выполнен: найдено 5 релевантных документов
✅ Релевантность: 0.7816
✅ Ответ сгенерирован локальной LLM
```

### Модели в использовании

- **Embeddings:** `intfloat/multilingual-e5-small` (HuggingFace)
- **LLM:** `Qwen/Qwen2.5-0.5B-Instruct` (HuggingFace)
- **Exit code:** 0 (успех)

---

## 🔄 Миграция кода

### Если вы использовали явный `embedding_type="gigachat"`

Ничего не меняется - код работает как раньше:
```python
RAGPipeline(embedding_type="gigachat", llm_type="gigachat")
```

Будет попытка подключиться к GigaChat API (если есть ключ).

### Если вы использовали `RAGPipeline()` без параметров

**Было (старое поведение):**
- Пытался подключиться к GigaChat API
- Timeout ошибка без API ключа

**Стало (новое поведение):**
- Использует локальные embeddings
- Работает без API ключа ✅

---

## 📝 Файлы измененные

| Файл | Изменение | Строки |
|------|-----------|--------|
| `src/rag_gigachat/core/vector_store.py` | Добавлена проверка конфига в `_init_embeddings()` | 81-111 |
| `src/rag_gigachat/core/rag_pipeline.py` | Изменен дефолт embedding_type | 79 |
| `src/rag_gigachat/ui/streamlit_app.py` | Явно указан embedding_type | 50 |

---

## 🎯 Результаты

✅ **Исключены timeout ошибки** при отсутствии GigaChat API ключа  
✅ **Система работает с локальными моделями** по умолчанию  
✅ **Сохранена обратная совместимость** для явных вызовов GigaChat  
✅ **Улучшена явность кода** через параметры вместо dефолтов  
✅ **Протестировано** на полном RAG цикле: загрузка документов → поиск → генерация ответа
