# 🏗️ Архитектура компонентов UI

Визуальное описание архитектуры, потока данных и взаимодействия компонентов.

---

## 📐 Макет приложения (Wire Frame)

```
┌─────────────────────────────────────────────────────────────────┐
│                      🤖 RAG Chat                                │
│                  Интеллектуальный поиск                         │
├─────────────────────┬─────────────────────────────────────────┤
│                     │                                           │
│   SIDEBAR (300px)   │          MAIN AREA (1200px+)            │
│                     │                                           │
│ ┌─────────────────┐ │ ┌─────────────────────────────────────┐  │
│ │ 📚 RAG GigaChat │ │ │ 💬 Диалог (history, height=400px)  │  │
│ │                 │ │ │ ┌─────────────────────────────────┐ │  │
│ │ ⚙️ Настройки    │ │ │ │ USER: Какой метод поиска...    │ │  │
│ │ (ConfigModal)   │ │ │ │                                 │ │  │
│ │                 │ │ │ │ ASSISTANT: Существует три      │ │  │
│ │ ─────────────── │ │ │ │ основных метода...              │ │  │
│ │ 📁 Документы    │ │ │ └─────────────────────────────────┘ │  │
│ │ (FileListPanel) │ │ │ ─────────────────────────────────── │  │
│ │                 │ │ │                                     │  │
│ │ 📄 document.pdf │ │ │ 🤖 Ответ                           │  │
│ │ 📄 guide.pdf    │ │ │ ┌─────────────────────────────────┐ │  │
│ │ 📄 report.pdf   │ │ │ │ Вот три основных метода:        │ │  │
│ │                 │ │ │ │ 1. Dense (эмбеддинги)          │ │  │
│ │ 🔍 Поиск: [___] │ │ │ │ 2. Sparse (BM25)               │ │  │
│ │                 │ │ │ │ 3. Hybrid (комбинированный)    │ │  │
│ │ 🔄 Обновить     │ │ │ │                                 │ │  │
│ │ 🗑️ Очистить    │ │ │ │ Источники:                      │ │  │
│ │                 │ │ │ │ [1. document.pdf, стр. 5]     │ │  │
│ │ ─────────────── │ │ │ │ [2. guide.pdf, стр. 15]       │ │  │
│ │ ⚙️ Дополнено    │ │ │ └─────────────────────────────────┘ │  │
│ │ Настройки       │ │ │ 📋 👍 👎 💾                        │  │
│ │ (Слайдеры...)   │ │ │ ─────────────────────────────────── │  │
│ │                 │ │ │ Ваш вопрос:                         │  │
│ │                 │ │ │ [________________________] 🚀       │  │
│ │                 │ │ └─────────────────────────────────────┘  │
│ │                 │ │                                           │
│ │                 │ │ 📊 Статистика                            │
│ │                 │ │ Вопросов: 5 | Ответов: 5 | Модель: ... │
│ │                 │ │                                           │
│ └─────────────────┘ └─────────────────────────────────────────┘
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Поток взаимодействия пользователя

```
ПОЛЬЗОВАТЕЛЬ
    │
    ├─────────────────────────────────────────────────┐
    │                                                  │
    v                                                  │
┌─────────────────────────────────────────────────┐   │
│ 1. Нажимает "⚙️ Расширенные настройки"          │   │
└─────────────────────────────────────────────────┘   │
    │                                                  │
    v                                                  │
┌──────────────────────────────────────────────────┐  │
│ ConfigModal.show()                               │  │
│ ├─ Открывает диалоговое окно                     │  │
│ ├─ Показывает 4 группы параметров               │  │
│ └─ Сохраняет в st.session_state                 │  │
└──────────────────────────────────────────────────┘  │
    │                                                  │
    ├─────────────────────────────────────────────────┤
    │                                                  │
    v                                                  │
┌──────────────────────────────────────────────────┐  │
│ 2. Выбирает файл в FileListPanel                │  │
│    ├─ Вводит поиск: "документ"                  │  │
│    ├─ Видит отфильтрованный список              │  │
│    └─ Нажимает на файл                          │  │
└──────────────────────────────────────────────────┘  │
    │                                                  │
    v                                                  │
┌──────────────────────────────────────────────────┐  │
│ session_state.selected_file = "/path/document" │  │
│ session_state.show_document_viewer = True       │  │
│ st.rerun()                                       │  │
└──────────────────────────────────────────────────┘  │
    │                                                  │
    v                                                  │
┌──────────────────────────────────────────────────┐  │
│ 3. DocumentViewer.show(file_path, page=1)       │  │
│    ├─ Отображает PDF через PDF.js               │  │
│    ├─ Показывает информацию о документе         │  │
│    └─ Позволяет менять страницу                 │  │
└──────────────────────────────────────────────────┘  │
    │                                                  │
    │<─────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────┐
│ 4. Пишет вопрос в чат: "Какой метод поиска?"  │
└──────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────┐
│ pipeline.query(question)                         │
│ ├─ Получает ответ от LLM                         │
│ ├─ Получает релевантные документы                │
│ └─ Возвращает (answer, retrieved_docs)          │
└──────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────┐
│ 5. HighlightedAnswer.show(answer, docs)         │
│    ├─ Показывает ответ                           │
│    ├─ Добавляет ссылки на источники             │
│    └─ Выделяет релевантные отрывки жёлтым     │
└──────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────┐
│ AnswerInteraction.show_actions(answer)           │
│ ├─ Показывает кнопки: 📋 👍 👎 💾              │
│ └─ Сохраняет feedback в session_state           │
└──────────────────────────────────────────────────┘
    │
    v
┌──────────────────────────────────────────────────┐
│ 6. Нажимает на ссылку "document.pdf, стр. 5"   │
│    ├─ Устанавливает selected_page = 5           │
│    ├─ Открывает st.dialog()                     │
│    └─ Показывает DocumentViewer на странице 5   │
└──────────────────────────────────────────────────┘
    │
    v
ПОВТОР С ШАГА 3
```

---

## 🔌 Архитектура данных (Session State)

```python
st.session_state = {
    
    # ════ ConfigModal параметры ════
    "show_config_modal": False,
    
    # Модели
    "llm_model": "GigaChat-2-Max",
    "embedding_model": "GigaChat-2-Max",
    "max_tokens": 2000,
    "temperature": 0.7,
    
    # Поиск
    "k_retrieve": 5,
    "max_context": 2000,
    "retrieval_type": "hybrid",  # dense, sparse, hybrid
    
    # Чанкирование
    "chunk_size": 500,
    "chunk_overlap": 80,
    
    # GigaChat
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "ocr_enabled": True,
    
    # ════ FileListPanel параметры ════
    "selected_domain": "UAV",
    "file_search": "",
    "selected_files": [],
    "force_reload_index": False,
    
    # ════ DocumentViewer параметры ════
    "show_document_viewer": False,
    "selected_file": "/path/to/document.pdf",
    "selected_page": 5,
    
    # ════ Chat параметры ════
    "messages": [
        {
            "role": "user",
            "content": "Какой метод поиска лучше?"
        },
        {
            "role": "assistant",
            "content": "Ответ модели...",
            "docs": [
                {
                    "doc_id": "document_p5",
                    "score": 0.92,
                    "text": "..."
                }
            ]
        }
    ],
    
    # ════ AnswerInteraction параметры ════
    "feedback": ("helpful", "answer_42"),  # или ("unhelpful", ...)
    "saved_answers": [
        "Сохранённый ответ 1",
        "Сохранённый ответ 2"
    ],
}
```

---

## 🔗 Зависимости компонентов

```
┌─────────────────────────────────────────────────────────┐
│                     RAG Chat App                        │
└────────┬────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────────────────────┐
         │                                                     │
         v                                                     │
    ┌──────────────┐                                          │
    │ ConfigModal  │  (independent)                            │
    │              │  Сохраняет параметры в session_state     │
    │ ├─ 🤖 Models │  Используется: RAGPipeline               │
    │ ├─ 🔍 Search │                                          │
    │ ├─ 📄 Chunk  │                                          │
    │ └─ 💬 GigaChat│                                          │
    └──────────────┘                                          │
         │                                                     │
         ├─────────────────────────────────────────────────────┤
         │                                                     │
         v                                                     │
    ┌──────────────────┐                                      │
    │  FileListPanel   │  (independent)                       │
    │                  │  Зависит от: data_config            │
    │ ├─ Domain picker │  Использует: session_state          │
    │ ├─ File search   │  Отправляет: selected_file          │
    │ ├─ File list     │               show_document_viewer   │
    │ └─ Index control │                                      │
    └──────────────────┘                                      │
         │                                                     │
         │  selected_file                                      │
         │  selected_page                                      │
         └──────────────┐                                      │
                        v                                      │
                   ┌──────────────────┐                       │
                   │ DocumentViewer   │  (depends on files)   │
                   │                  │  Зависит от: Path    │
                   │ ├─ PDF display   │  Использует: PDF.js  │
                   │ ├─ Page selector │  Отправляет: None    │
                   │ └─ Info panel    │  (read-only)         │
                   └──────────────────┘                       │
                                                              │
    ┌──────────────────────────┐                             │
    │   RAG Pipeline Query      │                             │
    │                           │                             │
    │ Зависит от:             │                             │
    │ ├─ ConfigModal (params)  │ ◄─────────────────┐        │
    │ ├─ Documents (from files)│                   │        │
    │ └─ RAGPipeline           │                   │        │
    │                           │                   │        │
    │ Выводит:                 │                   │        │
    │ ├─ answer (str)          │                   │        │
    │ └─ retrieved_docs (list) │                   │        │
    └──────────────────────────┘                   │        │
         │                                          │        │
         │  answer                                  │        │
         │  retrieved_docs                          │        │
         │                                          │        │
         v                                          │        │
    ┌──────────────────────┐                       │        │
    │ HighlightedAnswer    │  (depends on query)   │        │
    │                      │                        │        │
    │ ├─ Answer text       │                       │        │
    │ ├─ Source links      │──┐                   │        │
    │ └─ Highlighted text  │  │                   │        │
    └──────────────────────┘  │                   │        │
         │                     │                   │        │
         │  open_document      │                   │        │
         │  open_page          │                   │        │
         │  (ссылки)           │                   │        │
         └─────────────────────┼──────────────────┤        │
                               v                  │        │
                          Циклический:            │        │
                          - Открыть документ     │        │
                          - Прочитать            │        │
                          - Задать новый вопрос ─┘        │
         │                                                  │
         │                                                  │
         v                                                  │
    ┌──────────────────────┐                               │
    │ AnswerInteraction    │  (depends on answer)          │
    │                      │                                │
    │ ├─ 📋 Copy           │  Сохраняет feedback в:       │
    │ ├─ 👍 Helpful        │  ├─ st.session_state         │
    │ ├─ 👎 Unhelpful      │  └─ (optional) database      │
    │ └─ 💾 Save           │                                │
    └──────────────────────┘                               │
                        │                                   │
                        └───────────────────────────────────┘
```

---

## 📊 Структура retrieved_docs

```
retrieved_docs = [
    {
        "doc_id": "document_p5",              ← filename_pN
        "score": 0.92,                        ← 0-1, релевантность
        "text": "Полный текст отрывка...",   ← До 300 символов
        "metadata": {                         ← Optional
            "source": "document.pdf",
            "page": 5,
            "date_created": "2026-04-16"
        }
    },
    {
        "doc_id": "guide_p15",
        "score": 0.85,
        "text": "...",
        "metadata": { ... }
    },
    # ... и так далее
]
```

---

## 🎨 Стили CSS

```css
/* Основные цвета */
:root {
    --primary: #1E88E5;        /* Синий */
    --highlight: #ffeb3b;      /* Жёлтый */
    --background: #f0f2f6;     /* Светло-серый */
    --error: #d32f2f;          /* Красный */
    --success: #388e3c;        /* Зелёный */
}

/* Компоненты */
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: var(--primary);
    text-align: center;
}

.source-container {
    border-left: 4px solid var(--primary);
    padding-left: 10px;
    margin: 10px 0;
    background: var(--background);
    border-radius: 4px;
}

.highlighted-text {
    background-color: var(--highlight);
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: 500;
}

.stats-container {
    background-color: var(--background);
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
}

.pdf-container {
    background: white;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
    border-radius: 8px;
    padding: 20px;
    max-width: 100%;
}
```

---

## 🚀 Инициализация приложения

```python
# app.py

import streamlit as st
from pathlib import Path
from rag_gigachat.config import model_config, data_config
from rag_gigachat.ui.components import (
    ConfigModal,
    FileListPanel,
    DocumentViewer,
    HighlightedAnswer,
    AnswerInteraction
)

# ════ 1. КОНФИГУРАЦИЯ ════
st.set_page_config(
    page_title="RAG Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════ 2. ИНИЦИАЛИЗАЦИЯ STATE ════
def init_session_state():
    defaults = {
        "messages": [],
        "llm_model": model_config.llm_model_name,
        "temperature": model_config.temperature,
        "k_retrieve": model_config.default_k_retrieve,
        # ... остальные
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ════ 3. РЕНДЕР ════
# Боковая панель
with st.sidebar:
    st.markdown("### 📚 RAG Chat")
    ConfigModal.show()
    FileListPanel.show(data_config.documents_dirs)

# Основной контент
st.title("🤖 RAG Chat")
# ... чат интерфейс

# Документ
if st.session_state.get("show_document_viewer"):
    with st.dialog("Просмотр"):
        DocumentViewer.show(st.session_state.selected_file)
```

---

## 📈 Масштабируемость

### Для больших объёмов документов:

```python
# Кэширование с TTL
@st.cache_data(ttl=3600)
def get_pdf_files(domain_path):
    return sorted(domain_path.rglob("*.pdf"))

# Пагинация в FileListPanel
page_size = 20
total_files = len(pdf_files)
total_pages = (total_files + page_size - 1) // page_size

page = st.selectbox("Страница", range(1, total_pages + 1))
start_idx = (page - 1) * page_size
end_idx = start_idx + page_size

for file in pdf_files[start_idx:end_idx]:
    # ... показать файл
```

### Для больших ответов:

```python
# Потоковая генерация
def stream_answer(query):
    pipeline = RAGPipeline()
    
    for chunk in pipeline.stream_query(query):
        yield chunk

# В UI
answer_placeholder = st.empty()
full_answer = ""

for chunk in stream_answer(user_query):
    full_answer += chunk
    answer_placeholder.write(full_answer)
```

---

## 🔐 Безопасность

### 1. Валидация input
```python
def validate_file_path(file_path: str) -> bool:
    # Проверить, что путь внутри разрешённой директории
    safe_dir = data_config.documents_dirs["UAV"]
    return Path(file_path).resolve().is_relative_to(safe_dir)
```

### 2. Base64 для PDF
```python
# PDF загружается как Base64 → HTML → безиспользование iframe
import base64
pdf_data = base64.b64encode(pdf_bytes).decode()
# Защита от XSS
```

### 3. Session state
```python
# Session state не доступен между пользователями
# Каждая сессия изолирована
```

---

## 📝 Примечания разработчика

1. **Streamlit 1.30+** требуется для `st.dialog()`
2. **PDF.js** из CDN для экономии памяти
3. **Session state** автоматически управляется Streamlit
4. **Rerun** нужен после изменения state для обновления UI
5. **Caching** для оптимизации производительности

