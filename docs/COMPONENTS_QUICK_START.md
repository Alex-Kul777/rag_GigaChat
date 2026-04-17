# 🚀 Быстрый старт с компонентами

## 1️⃣ ConfigModal

**Модальное окно с расширенными настройками**

```python
from rag_gigachat.ui.components import ConfigModal

# В боковой панели или основной области
ConfigModal.show()

# Параметры доступны в session_state:
print(st.session_state.llm_model)        # "GigaChat-2-Max"
print(st.session_state.temperature)      # 0.7
print(st.session_state.k_retrieve)       # 5
```

### Что происходит:
1. ✅ Нажимается кнопка "⚙️ Расширенные настройки"
2. 📋 Открывается модальное окно с 4 группами параметров
3. 💾 Все изменения сохраняются в `st.session_state`
4. ✕ Закрытие окна не теряет значения

### Группы параметров:
- 🤖 Модели (LLM, embedding, tokens, temperature)
- 🔍 Поиск (Top-K, max context, retrieval type)
- 📄 Чанкирование (chunk size, overlap)
- 💬 GigaChat (top-p, penalty, OCR)

---

## 2️⃣ FileListPanel

**Панель со списком PDF файлов в сайдбаре**

```python
from rag_gigachat.ui.components import FileListPanel
from rag_gigachat.config import data_config

# В боковой панели
with st.sidebar:
    FileListPanel.show(documents_dirs=data_config.documents_dirs)

# Проверить выбранный файл
if st.session_state.get("selected_file"):
    print(st.session_state.selected_file)  # "/path/to/document.pdf"
```

### Структура documents_dirs:
```python
data_config.documents_dirs = {
    "debug": Path("/data/domain_2_Debug/books"),
    "ai": Path("/data/domain_1_AI/books"),
    "UAV": Path("/data/domain_7_UAV/books"),
}
```

### Что происходит при клике на файл:
1. 📄 Выбирается файл
2. 🎯 Устанавливается `st.session_state.selected_file`
3. 🔄 Вызывается `st.rerun()`
4. 👁️ Открывается DocumentViewer (если реализовано)

### Кнопки:
```python
# 🔄 Обновить индекс
st.session_state.force_reload_index = True

# 🗑️ Очистить
st.session_state.selected_files = []
st.session_state.selected_file = None
```

---

## 3️⃣ DocumentViewer

**Просмотр PDF документа с поддержкой страниц**

```python
from rag_gigachat.ui.components import DocumentViewer

# Базовое использование
DocumentViewer.show("/path/to/document.pdf")

# С указанием страницы
DocumentViewer.show(
    file_path="/path/to/document.pdf",
    page=5  # Открыть на странице 5
)

# В модальном окне (Streamlit 1.30+)
if st.session_state.get("show_document_viewer"):
    with st.dialog("Просмотр документа", width="large"):
        DocumentViewer.show(
            st.session_state.selected_file,
            page=st.session_state.get("selected_page", 1)
        )
```

### Интерфейс:
```
📄 document.pdf | Страница: [5▼] | 2.5 MB
─────────────────────────────────────────
[PDF отображается через PDF.js]
─────────────────────────────────────────
📋 Информация о документе
  Размер файла: 2.5 MB
  Всего страниц: 42
  Путь: books
  Создан: 2026-04-16
```

### Технология:
- 🔗 PDF.js из CDN
- 🔐 Base64 кодирование
- 📏 Масштаб: 1.5x
- 📜 Высота: 800px

---

## 4️⃣ HighlightedAnswer

**Ответ с автоматическими ссылками на источники**

```python
from rag_gigachat.ui.components import HighlightedAnswer

# Получить ответ и документы из RAG pipeline
answer, retrieved_docs = pipeline.query(
    query="Какой...",
    top_k=5
)

# Показать ответ с источниками
HighlightedAnswer.show(
    answer=answer,  # Строка с ответом LLM
    retrieved_docs=retrieved_docs,  # Список Dict с doc_id, score, text
    documents_dirs=data_config.documents_dirs,
    show_sources=True
)
```

### Формат retrieved_docs:
```python
retrieved_docs = [
    {
        "doc_id": "document_p5",      # filename_pN
        "score": 0.92,                # Релевантность 0-1
        "text": "Текст отрывка..."    # Контекст
    },
    {
        "doc_id": "guide_p15",
        "score": 0.85,
        "text": "..."
    },
]
```

### Вывод:
```
🤖 Ответ
────────────────────────────────────
[Полный ответ от LLM]

**Источники:**
1. [document.pdf, стр. 5](file=document|page=5) (релевантность: 0.92)
2. [guide.pdf, стр. 15](file=guide|page=15) (релевантность: 0.85)

────────────────────────────────────
📚 Источники и релевантные отрывки

#1. document.pdf, страница 5
Релевантность: 0.92
[👁️ Открыть документ]

Отрывок:
[Жёлтая подсветка текста из документа...]

────────────────────────────────────
```

### При клике на ссылку:
```python
# Автоматически:
# 1. Парсит "document_p5" → filename="document", page=5
# 2. Ищет файл в documents_dirs
# 3. Открывает DocumentViewer на странице 5
```

---

## 5️⃣ AnswerInteraction

**Кнопки для работы с ответом**

```python
from rag_gigachat.ui.components import AnswerInteraction

# После вывода ответа
AnswerInteraction.show_actions(
    answer="Текст ответа...",
    answer_id="answer_42"  # Уникальный ID
)
```

### Что происходит при нажатии:

| Кнопка | Код | Результат |
|--------|-----|-----------|
| 📋 Копировать | `copy(answer)` | Toast: "✓ Скопировано в буфер обмена" |
| 👍 Полезно | `st.session_state.feedback = ("helpful", answer_id)` | Записывается feedback |
| 👎 Не полезно | `st.session_state.feedback = ("unhelpful", answer_id)` | Записывается feedback |
| 💾 Сохранить | `saved_answers.append(answer)` | Добавляется в saved_answers |

### Использование feedback:
```python
if st.session_state.get("feedback"):
    status, answer_id = st.session_state.feedback
    if status == "helpful":
        # Отправить метрику телеметрии
        log_helpful_answer(answer_id)
    elif status == "unhelpful":
        # Пересчитать ранжирование
        retrain_ranker()
```

---

## 📦 Полный пример: Главное меню

```python
import streamlit as st
from pathlib import Path
from rag_gigachat.config import model_config, data_config
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.ui.components import (
    ConfigModal,
    FileListPanel,
    DocumentViewer,
    HighlightedAnswer,
    AnswerInteraction
)

# ═══════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════

st.set_page_config(page_title="RAG Chat", layout="wide")

# Инициализировать session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.llm_model = model_config.llm_model_name
    st.session_state.temperature = model_config.temperature
    st.session_state.k_retrieve = model_config.default_k_retrieve


# ═══════════════════════════════════════════════════════════
# БОКОВАЯ ПАНЕЛЬ
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📚 RAG GigaChat")
    st.divider()
    
    # Расширенные настройки
    ConfigModal.show()
    st.divider()
    
    # Список файлов
    FileListPanel.show(data_config.documents_dirs)


# ═══════════════════════════════════════════════════════════
# ОСНОВНОЙ ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════

st.markdown("# 🤖 RAG Chat")
st.write("Задайте вопрос о документах →")

# История чата
with st.container(height=300, border=True):
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# Ввод вопроса
col_input, col_send = st.columns([5, 1])

with col_input:
    user_input = st.text_area(
        "Ваш вопрос",
        placeholder="Спросите о чём-нибудь из документов...",
        height=60
    )

with col_send:
    st.write("")
    st.write("")
    send_button = st.button("🚀 Отправить")

# Обработать запрос
if send_button and user_input.strip():
    # Добавить в историю
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Получить ответ
    try:
        pipeline = RAGPipeline(
            llm_model_name=st.session_state.llm_model
        )
        
        answer, docs = pipeline.query(
            user_input,
            top_k=st.session_state.k_retrieve
        )
        
        # Добавить ответ
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "docs": docs
        })
        
        # Показать ответ с источниками
        st.markdown("---")
        HighlightedAnswer.show(answer, docs, data_config.documents_dirs)
        
        # Интерактивные кнопки
        st.markdown("---")
        AnswerInteraction.show_actions(answer, f"answer_{len(st.session_state.messages)}")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")


# ═══════════════════════════════════════════════════════════
# ПРОСМОТР ДОКУМЕНТА (В МОДАЛЬНОМ ОКНЕ)
# ═══════════════════════════════════════════════════════════

if st.session_state.get("show_document_viewer"):
    file_path = st.session_state.get("selected_file")
    
    # Найти полный путь
    for domain_dir in data_config.documents_dirs.values():
        candidate = domain_dir / f"{Path(file_path).stem}.pdf"
        if candidate.exists():
            file_path = str(candidate)
            break
    
    # Показать в диалоге
    with st.dialog("Просмотр документа", width="large"):
        DocumentViewer.show(
            file_path,
            page=st.session_state.get("selected_page", 1)
        )
        
        if st.button("✕ Закрыть"):
            st.session_state.show_document_viewer = False
            st.rerun()


# ═══════════════════════════════════════════════════════════
# СТАТИСТИКА
# ═══════════════════════════════════════════════════════════

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Вопросов", len([m for m in st.session_state.messages if m["role"] == "user"]))

with col2:
    st.metric("Ответов", len([m for m in st.session_state.messages if m["role"] == "assistant"]))

with col3:
    st.metric("Модель", st.session_state.llm_model.split("/")[-1])

with col4:
    st.metric("Top-K", st.session_state.k_retrieve)
```

---

## 🔗 Интеграция с RAG Pipeline

```python
from rag_gigachat.core.rag_pipeline import RAGPipeline

# Создать pipeline с параметрами из UI
pipeline = RAGPipeline(
    llm_model_name=st.session_state.llm_model,
    embedding_model_name=st.session_state.embedding_model,
    chunk_size=st.session_state.chunk_size,
    chunk_overlap=st.session_state.chunk_overlap,
)

# Выполнить запрос
answer, retrieved_docs = pipeline.query(
    query=user_input,
    top_k=st.session_state.k_retrieve,
    retrieval_type=st.session_state.retrieval_type  # dense, sparse, hybrid
)

# Каждый doc в retrieved_docs:
for doc in retrieved_docs:
    print(f"doc_id: {doc['doc_id']}")      # "document_p5"
    print(f"score: {doc['score']}")        # 0.92
    print(f"text: {doc['text'][:100]}")    # Первые 100 символов
    print(f"metadata: {doc.get('metadata', {})}")
```

---

## 📋 Чек-лист интеграции

- [ ] Импортированы все компоненты в приложение
- [ ] Session state инициализирован
- [ ] Боковая панель включает ConfigModal и FileListPanel
- [ ] Основной интерфейс выводит HighlightedAnswer и AnswerInteraction
- [ ] DocumentViewer открывается в модальном окне (st.dialog)
- [ ] Ссылки на источники открывают правильные страницы PDF
- [ ] Кнопки AnswerInteraction сохраняют feedback в session_state
- [ ] Тестирование с real PDF файлами
- [ ] Проверка мобильной адаптивности (если требуется)

---

## 🐛 Частые проблемы

### PDF не открывается
```python
# Проверить, что файл существует
if not Path(file_path).exists():
    st.error(f"Файл не найден: {file_path}")

# Проверить права доступа
Path(file_path).stat()
```

### Session state пуст после перезагрузки
```python
# Инициализировать в начале app.py:
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "GigaChat-2-Max"
```

### Dialog не открывается (старый Streamlit)
```python
# Для Streamlit < 1.30 использовать columns:
if st.session_state.get("show_document_viewer"):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        DocumentViewer.show(file_path)
```

### Ссылки на источники не работают
```python
# Проверить формат doc_id
# Должно быть: "filename_pN" где N = номер страницы
# Пример: "document_p5"

# Проверить, что documents_dirs корректны
print(data_config.documents_dirs)
```

---

## 📚 Дополнительная информация

- Full documentation: [COMPONENTS.md](./COMPONENTS.md)
- Example app: `src/rag_gigachat/ui/app_example.py`
- Config: `src/rag_gigachat/config.py`
- RAG Pipeline: `src/rag_gigachat/core/rag_pipeline.py`
