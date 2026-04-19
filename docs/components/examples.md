# 📖 Примеры кода для компонентов

Готовые копипаст примеры для каждого компонента.

---

## 🟦 ConfigModal - Модальное окно настроек

### Минимальный пример
```python
import streamlit as st
from rag_gigachat.ui.components import ConfigModal

st.set_page_config(layout="wide")

# В боковой панели
with st.sidebar:
    ConfigModal.show()

# Использовать параметры
st.write(f"Temperature: {st.session_state.temperature}")
st.write(f"Top-K: {st.session_state.k_retrieve}")
```

### С инициализацией значений
```python
import streamlit as st
from rag_gigachat.config import model_config
from rag_gigachat.ui.components import ConfigModal

# Инициализировать из конфига
if "temperature" not in st.session_state:
    st.session_state.temperature = model_config.temperature
if "k_retrieve" not in st.session_state:
    st.session_state.k_retrieve = model_config.default_k_retrieve

with st.sidebar:
    ConfigModal.show()
```

### С применением настроек в pipeline
```python
import streamlit as st
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.ui.components import ConfigModal

with st.sidebar:
    ConfigModal.show()

# Создать pipeline с параметрами из UI
if st.button("Инициализировать"):
    pipeline = RAGPipeline(
        llm_model_name=st.session_state.llm_model,
        embedding_model_name=st.session_state.embedding_model,
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
    )
    
    # Использовать pipeline
    answer, docs = pipeline.query(
        "Ваш вопрос",
        top_k=st.session_state.k_retrieve,
        retrieval_type=st.session_state.retrieval_type
    )
    
    st.success("✓ Pipeline готов!")
```

### Сохранение настроек в файл
```python
import streamlit as st
import json
from pathlib import Path
from rag_gigachat.ui.components import ConfigModal

CONFIG_FILE = Path("config_ui.json")

with st.sidebar:
    col1, col2 = st.columns(2)
    
    with col1:
        ConfigModal.show()
    
    with col2:
        if st.button("💾 Сохранить"):
            config = {
                "llm_model": st.session_state.llm_model,
                "temperature": st.session_state.temperature,
                "k_retrieve": st.session_state.k_retrieve,
                "chunk_size": st.session_state.chunk_size,
                # ... остальные параметры
            }
            
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            
            st.success("✓ Параметры сохранены")

# Загрузить при запуске
if CONFIG_FILE.exists():
    with open(CONFIG_FILE) as f:
        saved_config = json.load(f)
        for key, value in saved_config.items():
            if key not in st.session_state:
                st.session_state[key] = value
```

---

## 🟩 FileListPanel - Панель списка файлов

### Минимальный пример
```python
import streamlit as st
from rag_gigachat.config import data_config
from rag_gigachat.ui.components import FileListPanel

with st.sidebar:
    FileListPanel.show(data_config.documents_dirs)

# Проверить выбранный файл
if st.session_state.get("selected_file"):
    st.write(f"Выбран файл: {st.session_state.selected_file}")
```

### С открытием DocumentViewer
```python
import streamlit as st
from pathlib import Path
from rag_gigachat.config import data_config
from rag_gigachat.ui.components import FileListPanel, DocumentViewer

# Инициализировать
if "show_document_viewer" not in st.session_state:
    st.session_state.show_document_viewer = False

with st.sidebar:
    FileListPanel.show(data_config.documents_dirs)

# Показать документ если выбран
if st.session_state.get("show_document_viewer") and st.session_state.get("selected_file"):
    with st.dialog("Просмотр документа", width="large"):
        # Найти полный путь
        file_name = st.session_state.selected_file
        file_path = None
        
        for domain_dir in data_config.documents_dirs.values():
            candidate = domain_dir / f"{file_name}.pdf"
            if candidate.exists():
                file_path = str(candidate)
                break
        
        if file_path:
            DocumentViewer.show(file_path, st.session_state.get("selected_page", 1))
        
        if st.button("Закрыть"):
            st.session_state.show_document_viewer = False
            st.rerun()
```

### С фильтрацией и сортировкой
```python
import streamlit as st
from pathlib import Path
from rag_gigachat.config import data_config
from rag_gigachat.ui.components import FileListPanel

# Расширенная версия FileListPanel
with st.sidebar:
    st.subheader("📁 Документы")
    
    # Выбор домена
    selected_domain = st.selectbox(
        "Домен",
        options=list(data_config.documents_dirs.keys())
    )
    
    domain_path = data_config.documents_dirs[selected_domain]
    
    # Дополнительные фильтры
    col1, col2 = st.columns(2)
    
    with col1:
        search = st.text_input("🔍 Поиск")
    
    with col2:
        sort_by = st.selectbox(
            "Сортировка",
            ["Имя", "Размер", "Дата"]
        )
    
    # Получить файлы
    pdf_files = sorted(domain_path.rglob("*.pdf"))
    
    if search:
        pdf_files = [f for f in pdf_files if search.lower() in f.name.lower()]
    
    # Сортировка
    if sort_by == "Размер":
        pdf_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    elif sort_by == "Дата":
        pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Показать список
    for file in pdf_files[:10]:  # Топ 10
        if st.button(f"📄 {file.name}", use_container_width=True):
            st.session_state.selected_file = str(file)
            st.rerun()
```

### С удалением и переименованием
```python
import streamlit as st
from pathlib import Path
from rag_gigachat.config import data_config
from rag_gigachat.ui.components import FileListPanel

with st.sidebar:
    FileListPanel.show(data_config.documents_dirs)
    
    st.markdown("---")
    st.subheader("⚙️ Управление файлами")
    
    # Выбор файла для редактирования
    domain_path = data_config.documents_dirs["UAV"]
    pdf_files = list(domain_path.rglob("*.pdf"))
    
    if pdf_files:
        selected = st.selectbox(
            "Выберите файл",
            [f.name for f in pdf_files]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Удалить"):
                Path(selected).unlink()
                st.success("✓ Файл удалён")
                st.rerun()
        
        with col2:
            new_name = st.text_input("Новое имя")
            if new_name and st.button("✏️ Переименовать"):
                old_path = domain_path / selected
                new_path = domain_path / new_name
                old_path.rename(new_path)
                st.success("✓ Файл переименован")
                st.rerun()
```

---

## 🟦 DocumentViewer - Просмотр PDF

### Минимальный пример
```python
import streamlit as st
from rag_gigachat.ui.components import DocumentViewer

DocumentViewer.show("/path/to/document.pdf")
```

### С выбором страницы
```python
import streamlit as st
from pathlib import Path
from rag_gigachat.ui.components import DocumentViewer

file_path = "/path/to/document.pdf"

# Получить количество страниц
try:
    import PyPDF2
    with open(file_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        total_pages = len(pdf.pages)
except:
    total_pages = 1

# Выбор страницы
page = st.slider("Страница", 1, total_pages, 1)

# Показать документ
DocumentViewer.show(file_path, page)
```

### В модальном окне с кнопкой закрытия
```python
import streamlit as st
from rag_gigachat.ui.components import DocumentViewer

if "show_viewer" not in st.session_state:
    st.session_state.show_viewer = False

if st.button("📄 Открыть документ"):
    st.session_state.show_viewer = True

if st.session_state.show_viewer:
    with st.dialog("Просмотр PDF", width="large"):
        DocumentViewer.show("/path/to/document.pdf")
        
        if st.button("Закрыть"):
            st.session_state.show_viewer = False
            st.rerun()
```

### С аннотациями (готовая структура)
```python
import streamlit as st
from rag_gigachat.ui.components import DocumentViewer

file_path = "/path/to/document.pdf"

col1, col2 = st.columns([3, 1])

with col1:
    DocumentViewer.show(file_path)

with col2:
    st.subheader("📝 Заметки")
    
    notes = st.text_area("Добавьте заметку")
    
    if st.button("💾 Сохранить"):
        # Сохранить в базу
        st.success("✓ Заметка сохранена")
    
    # Показать сохранённые заметки
    st.write("### Ваши заметки:")
    for i, note in enumerate(["Заметка 1", "Заметка 2"], 1):
        st.caption(f"{i}. {note}")
```

### С поиском текста в PDF
```python
import streamlit as st
from rag_gigachat.ui.components import DocumentViewer

file_path = "/path/to/document.pdf"

col1, col2 = st.columns([3, 1])

with col1:
    DocumentViewer.show(file_path)

with col2:
    st.subheader("🔍 Поиск")
    
    search_text = st.text_input("Ищите в документе")
    
    if search_text:
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                results = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if search_text.lower() in text.lower():
                        results.append(page_num)
            
            if results:
                st.success(f"Найдено на страницах: {', '.join(map(str, results))}")
                
                if st.button(f"→ На страницу {results[0]}"):
                    st.session_state.selected_page = results[0]
                    st.rerun()
            else:
                st.info("Текст не найден")
        except Exception as e:
            st.error(f"Ошибка поиска: {e}")
```

---

## 🟦 HighlightedAnswer - Ответ с источниками

### Минимальный пример
```python
import streamlit as st
from rag_gigachat.ui.components import HighlightedAnswer
from rag_gigachat.config import data_config

# Пример результатов
retrieved_docs = [
    {
        "doc_id": "document_p5",
        "score": 0.92,
        "text": "Текст из документа..."
    },
]

HighlightedAnswer.show(
    answer="Ответ модели...",
    retrieved_docs=retrieved_docs,
    documents_dirs=data_config.documents_dirs
)
```

### Интеграция с RAG Pipeline
```python
import streamlit as st
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.config import data_config
from rag_gigachat.ui.components import HighlightedAnswer

pipeline = RAGPipeline()

user_query = st.text_input("Ваш вопрос")

if user_query:
    # Получить ответ
    answer, retrieved_docs = pipeline.query(
        user_query,
        top_k=5
    )
    
    # Показать результат
    HighlightedAnswer.show(
        answer=answer,
        retrieved_docs=retrieved_docs,
        documents_dirs=data_config.documents_dirs,
        show_sources=True
    )
```

### С кастомной постобработкой
```python
import streamlit as st
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.config import data_config
from rag_gigachat.ui.components import HighlightedAnswer

pipeline = RAGPipeline()

user_query = st.text_input("Ваш вопрос")

if user_query:
    with st.spinner("🔄 Ищу ответ..."):
        answer, retrieved_docs = pipeline.query(user_query, top_k=5)
    
    # Фильтрация по релевантности
    min_score = st.slider("Минимальная релевантность", 0.0, 1.0, 0.7)
    filtered_docs = [d for d in retrieved_docs if d.get("score", 0) >= min_score]
    
    if filtered_docs:
        HighlightedAnswer.show(
            answer=answer,
            retrieved_docs=filtered_docs,
            documents_dirs=data_config.documents_dirs
        )
    else:
        st.warning("⚠️ Нет документов с достаточной релевантностью")
```

### С кэшированием ответов
```python
import streamlit as st
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.config import data_config
from rag_gigachat.ui.components import HighlightedAnswer

@st.cache_data
def get_answer(query, top_k=5):
    pipeline = RAGPipeline()
    answer, docs = pipeline.query(query, top_k=top_k)
    return answer, docs

user_query = st.text_input("Ваш вопрос")

if user_query:
    answer, retrieved_docs = get_answer(user_query)
    
    HighlightedAnswer.show(
        answer=answer,
        retrieved_docs=retrieved_docs,
        documents_dirs=data_config.documents_dirs
    )
```

### С экспортом в Markdown
```python
import streamlit as st
from rag_gigachat.ui.components import HighlightedAnswer
from rag_gigachat.config import data_config

answer = "Ответ модели..."
retrieved_docs = [...]

HighlightedAnswer.show(answer, retrieved_docs, data_config.documents_dirs)

# Кнопка экспорта
col1, col2 = st.columns(2)

with col1:
    if st.button("📋 Копировать"):
        st.toast("✓ Скопировано")

with col2:
    if st.download_button(
        label="📥 Скачать как Markdown",
        data=f"# Ответ\n\n{answer}\n\n## Источники\n" + 
              "\n".join([f"- {d['doc_id']}: {d['text'][:100]}" for d in retrieved_docs]),
        file_name="answer.md",
        mime="text/markdown"
    ):
        st.success("✓ Файл скачан")
```

---

## 🟦 AnswerInteraction - Интерактивные кнопки

### Минимальный пример
```python
import streamlit as st
from rag_gigachat.ui.components import AnswerInteraction

answer = "Это ответ модели..."

AnswerInteraction.show_actions(answer, answer_id="answer_1")

# Проверить feedback
if st.session_state.get("feedback"):
    status, answer_id = st.session_state.feedback
    st.write(f"Feedback: {status} для {answer_id}")
```

### С обработкой feedback
```python
import streamlit as st
from rag_gigachat.ui.components import AnswerInteraction

answer = "Ответ модели..."

AnswerInteraction.show_actions(answer, answer_id="answer_42")

# Обработка feedback
if st.session_state.get("feedback"):
    status, answer_id = st.session_state.feedback
    
    if status == "helpful":
        st.balloons()
        st.success("✓ Спасибо за положительный отзыв!")
        # Отправить телеметрию
        log_helpful_answer(answer_id)
    
    elif status == "unhelpful":
        st.warning("Мы улучшим ответы!")
        # Отправить телеметрию
        log_unhelpful_answer(answer_id)
    
    # Очистить feedback
    st.session_state.feedback = None
    st.rerun()
```

### С сохранением в базу
```python
import streamlit as st
import json
from pathlib import Path
from rag_gigachat.ui.components import AnswerInteraction

SAVED_FILE = Path("saved_answers.json")

answer = "Ответ модели..."

AnswerInteraction.show_actions(answer, answer_id="answer_1")

# Обработка сохранения
if st.session_state.get("saved_answers"):
    saved = []
    
    if SAVED_FILE.exists():
        with open(SAVED_FILE) as f:
            saved = json.load(f)
    
    # Добавить новый ответ
    saved.append({
        "answer_id": "answer_1",
        "answer": answer,
        "timestamp": st.session_state.get("timestamp")
    })
    
    with open(SAVED_FILE, "w") as f:
        json.dump(saved, f, indent=2)

# Показать сохранённые ответы
if SAVED_FILE.exists():
    with open(SAVED_FILE) as f:
        saved_list = json.load(f)
    
    st.write(f"### Сохранено: {len(saved_list)} ответов")
    for item in saved_list[-5:]:  # Последние 5
        st.caption(f"- {item['answer'][:100]}...")
```

### С шарингом
```python
import streamlit as st
from rag_gigachat.ui.components import AnswerInteraction

answer = "Ответ модели..."

col1, col2 = st.columns(2)

with col1:
    AnswerInteraction.show_actions(answer, answer_id="answer_1")

with col2:
    st.subheader("🔗 Шаринг")
    
    if st.button("📨 Отправить по email"):
        st.write("mailto:?body=" + answer[:100])
        st.toast("✓ Ссылка скопирована")
    
    if st.button("📱 Поделиться"):
        st.toast("✓ QR код сгенерирован")
    
    if st.button("🔗 Скопировать ссылку"):
        st.toast("✓ Ссылка скопирована")
```

---

## 🎯 Полный пример с всеми компонентами

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

# ════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ
# ════════════════════════════════════════════════

st.set_page_config(page_title="RAG Chat", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.llm_model = model_config.llm_model_name
    st.session_state.k_retrieve = model_config.default_k_retrieve
    st.session_state.show_document_viewer = False

# ════════════════════════════════════════════════
# БОКОВАЯ ПАНЕЛЬ
# ════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📚 RAG Chat")
    st.divider()
    
    ConfigModal.show()
    st.divider()
    
    FileListPanel.show(data_config.documents_dirs)

# ════════════════════════════════════════════════
# ОСНОВНОЙ ИНТЕРФЕЙС
# ════════════════════════════════════════════════

st.title("🤖 RAG Chat с GigaChat")

# История
with st.container(height=300, border=True):
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# Ввод
col_input, col_send = st.columns([5, 1])

with col_input:
    user_input = st.text_area("Ваш вопрос", height=80)

with col_send:
    st.write("")
    st.write("")
    is_sent = st.button("🚀")

if is_sent and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        with st.spinner("⏳ Обработка..."):
            pipeline = RAGPipeline(llm_model_name=st.session_state.llm_model)
            answer, docs = pipeline.query(
                user_input,
                top_k=st.session_state.k_retrieve
            )
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "docs": docs
        })
        
        st.markdown("---")
        HighlightedAnswer.show(answer, docs, data_config.documents_dirs)
        st.markdown("---")
        AnswerInteraction.show_actions(answer, f"answer_{len(st.session_state.messages)}")
        
        st.rerun()
    
    except Exception as e:
        st.error(f"❌ {e}")

# ════════════════════════════════════════════════
# ПРОСМОТР ДОКУМЕНТА
# ════════════════════════════════════════════════

if st.session_state.show_document_viewer and st.session_state.get("selected_file"):
    file_path = st.session_state.selected_file
    
    for domain_dir in data_config.documents_dirs.values():
        candidate = domain_dir / f"{Path(file_path).stem}.pdf"
        if candidate.exists():
            file_path = str(candidate)
            break
    
    with st.dialog("📖", width="large"):
        DocumentViewer.show(file_path, st.session_state.get("selected_page", 1))
        
        if st.button("✕"):
            st.session_state.show_document_viewer = False
            st.rerun()
```

---

## 💡 Советы и трюки

### 1. Кэширование результатов
```python
@st.cache_data(ttl=3600)
def get_answer(query):
    pipeline = RAGPipeline()
    return pipeline.query(query, top_k=5)
```

### 2. Отслеживание метрик
```python
col1, col2, col3 = st.columns(3)
col1.metric("Запросов", len([m for m in st.session_state.messages if m["role"] == "user"]))
col2.metric("Модель", st.session_state.llm_model.split("/")[-1])
col3.metric("Top-K", st.session_state.k_retrieve)
```

### 3. Обработка ошибок
```python
try:
    answer, docs = pipeline.query(query)
except ValueError as e:
    st.error(f"❌ Ошибка валидации: {e}")
except TimeoutError:
    st.warning("⏱️ Запрос слишком долгий, сокращаю контекст...")
except Exception as e:
    st.error(f"❌ Неожиданная ошибка: {e}")
    logger.error(f"Error: {e}", exc_info=True)
```

### 4. Прогресс-бар
```python
with st.spinner("⏳ Загрузка..."):
    progress_bar = st.progress(0)
    
    for i in range(100):
        progress_bar.progress(i)
        # ... работа
```

### 5. Компактный layout
```python
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
```
