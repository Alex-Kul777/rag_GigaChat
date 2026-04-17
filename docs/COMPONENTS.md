# 🎨 Компоненты Streamlit UI

Полный набор переиспользуемых компонентов для RAG Chat приложения.

## 📋 Содержание

1. [ConfigModal](#configmodal) - Модальное окно с настройками
2. [FileListPanel](#filelistpanel) - Панель списка файлов
3. [DocumentViewer](#documentviewer) - Просмотр PDF документов
4. [HighlightedAnswer](#highlightedanswer) - Ответ с подсветкой источников
5. [AnswerInteraction](#answerinteraction) - Интерактивные действия

---

## ConfigModal

### Назначение
Модальное окно с расширенными настройками. Открывается по кнопке "⚙️ Расширенные настройки" на боковой панели.

### Группы параметров

#### 1️⃣ Модели
```python
ConfigModal.show()
```

- **LLM модель** - выбор модели для генерации ответов
- **Embedding модель** - модель для создания эмбеддингов
- **Max tokens** - максимальное количество токенов в ответе (100-4000)
- **Temperature** - степень "творчества" модели (0-2.0)

#### 2️⃣ Поиск
- **Top-K документов** - количество релевантных документов (1-20)
- **Max контекст** - максимальный размер контекста в символах (500-5000)
- **Тип поиска** - радиокнопки: dense, sparse, hybrid

#### 3️⃣ Чанкирование
- **Размер чанка** - длина текстового фрагмента (100-2000)
- **Перекрытие чанков** - количество перекрывающихся символов (0-500)

#### 4️⃣ GigaChat
- **Top-P** - разнообразие ответов (0-1.0)
- **Penalty повторений** - штраф за повторы (1.0-2.0)
- **OCR для PDF** - включить OCR для сканированных документов (checkbox)

### Состояние
Все параметры сохраняются в `st.session_state`:

```python
st.session_state.llm_model
st.session_state.temperature
st.session_state.k_retrieve
st.session_state.chunk_size
# и т.д.
```

### Действия
- **✅ Применить** - сохранить и применить настройки
- **🔄 Сброс** - вернуть значения по умолчанию
- **✕ Закрыть** - закрыть окно без сохранения

---

## FileListPanel

### Назначение
Правая боковая панель со списком загруженных PDF файлов. Ширина ~300px.

### Функциональность

```python
FileListPanel.show(documents_dirs=data_config.documents_dirs)
```

#### Выбор домена
Dropbox для выбора категории документов:
- debug
- ai  
- UAV

#### Поиск файлов
Текстовое поле для поиска по имени файла:
```
🔍 Поиск файла: "документ"
```

#### Список файлов
Каждый файл показывается как кнопка с метаинформацией:
```
📄 document.pdf          5p
📄 guide.pdf            42p
📄 report.pdf           12p
```

При клике на файл:
1. Файл открывается в `DocumentViewer`
2. Устанавливается `st.session_state.selected_file`
3. Интерфейс переходит на вкладку просмотра

#### Кнопки действий
- **🔄 Обновить индекс** - пересчитать векторный индекс
- **🗑️ Очистить** - очистить выделение и выбранные файлы

#### Статистика
- Количество найденных файлов
- Текущий домен (путь)
- Размер файла (MB) при выборе

---

## DocumentViewer

### Назначение
Интерактивный просмотр PDF документов с поддержкой переходов между страницами.

### Использование

```python
DocumentViewer.show(
    file_path="/path/to/document.pdf",
    page=5  # Открыть на странице 5
)
```

### Компоненты интерфейса

#### Заголовок
```
📄 document.pdf    | Страница: [5] | 2.5 MB
```

#### Просмотр PDF
- Рендеринг через PDF.js
- Base64 кодирование для безопасности
- Масштабирование под размер экрана
- Высота: 800px, прокрутка включена

#### Информация о документе
Expandable секция с:
- Размер файла (MB)
- Всего страниц
- Путь к файлу
- Дата создания

### Технические детали

PDF отображается через HTML с использованием **PDF.js** (CDN):
- Worker: `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js`
- Масштаб: 1.5x
- Кодирование: Base64

```python
st.components.v1.html(pdf_display, height=800, scrolling=True)
```

---

## HighlightedAnswer

### Назначение
Отображение ответа LLM с автоматическими ссылками на источники и подсветкой релевантных отрывков.

### Использование

```python
HighlightedAnswer.show(
    answer="Ответ модели...",
    retrieved_docs=[
        {
            "doc_id": "document_p5",
            "score": 0.92,
            "text": "Релевантный текст..."
        },
        # ...
    ],
    documents_dirs=data_config.documents_dirs,
    show_sources=True
)
```

### Структура вывода

#### 1. Основной ответ
```
🤖 Ответ
[Текст ответа от LLM]

**Источники:**
1. [document.pdf, стр. 5](file=document|page=5) (релевантность: 0.92)
2. [guide.pdf, стр. 15](file=guide|page=15) (релевантность: 0.85)
3. [report.pdf, стр. 8](file=report|page=8) (релевантность: 0.78)
```

#### 2. Развёрнутая секция источников

Для каждого документа:

```
┌─────────────────────────────────────┐
│ #1. document.pdf, страница 5        │
│ Релевантность: 0.92                 │
├─────────────────────────────────────┤
│ [👁️ Открыть документ]               │
├─────────────────────────────────────┤
│ Отрывок:                            │
│ ┌───────────────────────────────────┤
│ │ [Жёлтая подсветка текста...]     │
│ └───────────────────────────────────┤
│ 📌 Дополнение к ответу на основе... │
└─────────────────────────────────────┘
```

### Обработка ссылок

При клике на кнопку "👁️ Открыть документ":
1. Извлекается имя файла из `doc_id` (формат: `filename_pN`)
2. Ищется полный путь в `documents_dirs`
3. Открывается `DocumentViewer` на нужной странице

```python
doc_id: "document_p5"
→ filename: "document"
→ page: 5
→ file_path: "/home/kap/projects/data/domain_7_UAV/books/document.pdf"
→ DocumentViewer.show(file_path, page=5)
```

### Подсветка текста

Релевантные отрывки выделяются жёлтым:
```html
<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">
    [Текст с подсветкой]
</mark>
```

---

## AnswerInteraction

### Назначение
Интерактивные кнопки для работы с ответами (копирование, оценка, сохранение).

### Использование

```python
AnswerInteraction.show_actions(
    answer="Текст ответа...",
    answer_id="answer_42"
)
```

### Кнопки

| Кнопка | Действие | Результат |
|--------|----------|-----------|
| 📋 Копировать | Копирование в буфер обмена | Toast: "✓ Скопировано в буфер обмена" |
| 👍 Полезно | Отметить ответ как полезный | `st.session_state.feedback = ("helpful", answer_id)` |
| 👎 Не полезно | Отметить как неполезный | `st.session_state.feedback = ("unhelpful", answer_id)` |
| 💾 Сохранить | Добавить в сохранённые ответы | `st.session_state.saved_answers.append(answer)` |

---

## 🔌 Интеграция в приложение

### Основной файл приложения

```python
from rag_gigachat.ui.components import (
    ConfigModal,
    FileListPanel, 
    DocumentViewer,
    HighlightedAnswer,
    AnswerInteraction
)

# В боковой панели
with st.sidebar:
    ConfigModal.show()
    FileListPanel.show(data_config.documents_dirs)

# В основной области
if user_query:
    answer, docs = pipeline.query(user_query)
    HighlightedAnswer.show(answer, docs, data_config.documents_dirs)
    AnswerInteraction.show_actions(answer)

# Просмотр документа
if st.session_state.get("show_document_viewer"):
    with st.dialog("Просмотр документа"):
        DocumentViewer.show(st.session_state.selected_file)
```

### Session State структура

```python
st.session_state = {
    # ConfigModal
    "show_config_modal": False,
    "llm_model": "GigaChat-2-Max",
    "temperature": 0.7,
    "k_retrieve": 5,
    # ... остальные параметры
    
    # FileListPanel
    "selected_domain": "UAV",
    "selected_files": [],
    "file_search": "",
    "force_reload_index": False,
    
    # DocumentViewer
    "show_document_viewer": False,
    "selected_file": "document.pdf",
    "selected_page": 1,
    
    # Chat
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ],
    
    # Feedback
    "feedback": ("helpful", "answer_42"),
    "saved_answers": ["Ответ 1", "Ответ 2"],
}
```

---

## 📐 Макет приложения

```
┌─────────────────────────────────────────────────────┐
│                      🤖 RAG Chat                    │
├──────────────────────┬──────────────────────────────┤
│                      │                              │
│    FileListPanel     │      Основной чат            │
│  (FileList.show)     │   (render_main_interface)   │
│                      │                              │
│  📁 Домен: [v]       │  💬 Диалог                   │
│  🔍 Поиск: [____]    │  ┌──────────────────────┐   │
│  📄 document.pdf   5p│  │ Привет! Как дела?    │   │
│  📄 guide.pdf     42p│  │ Ассистент: ...       │   │
│  📄 report.pdf    12p│  └──────────────────────┘   │
│                      │  Ваш вопрос:                 │
│  🔄 Обновить         │  [__________________]       │
│  🗑️ Очистить         │  [🚀 Отправить]             │
│                      │  ──────────────────────     │
│  ⚙️ Расширенные      │  🤖 Ответ                   │
│     настройки        │  [...ответ...]              │
│                      │                              │
│                      │  Источники (3):              │
│                      │  #1. doc.pdf, стр. 5  [👁️] │
│                      │  #2. guide.pdf, стр. 15 [👁️]│
│                      │                              │
│                      │  [📋] [👍] [👎] [💾]       │
└──────────────────────┴──────────────────────────────┘
```

---

## 🎨 Стили и цвета

### Основные цвета
- **Primary**: `#1E88E5` (синий)
- **Highlight**: `#ffeb3b` (жёлтый)
- **Background**: `#f0f2f6` (светло-серый)
- **Error**: `#d32f2f` (красный)
- **Success**: `#388e3c` (зелёный)

### CSS классы

```css
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1E88E5;
}

.source-container {
    border-left: 4px solid #1E88E5;
    padding-left: 10px;
    margin: 10px 0;
}

.stats-container {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
}
```

---

## ⚙️ Требования

- **Streamlit**: >= 1.30 (для `st.dialog`)
- **PyPDF2**: для получения информации о страницах
- **pathlib**: встроенный модуль

```bash
pip install streamlit>=1.30 PyPDF2
```

---

## 📚 Примеры использования

### Полный пример интеграции

See: `src/rag_gigachat/ui/app_example.py`

### Быстрый старт

```python
import streamlit as st
from rag_gigachat.config import data_config
from rag_gigachat.ui.components import FileListPanel, ConfigModal

st.set_page_config(layout="wide")

# Инициализировать session state
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "GigaChat-2-Max"

# Боковая панель
with st.sidebar:
    ConfigModal.show()
    FileListPanel.show(data_config.documents_dirs)

# Основной контент
st.write("Основной контент здесь...")
```

---

## 🐛 Отладка

### Логирование

Компоненты используют `logging`:

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("FileListPanel инициализирована")
logger.error("Ошибка при загрузке PDF")
```

### Session State

Проверить состояние:

```python
st.write(st.session_state)  # Показать всё состояние
st.write(st.session_state.selected_file)  # Конкретное значение
```

### Переинициализация

```python
if st.button("🔄 Очистить состояние"):
    st.session_state.clear()
    st.rerun()
```

---

## 📝 Заметки разработчика

1. **Dialog vs Columns**: `st.dialog` (Streamlit 1.30+) для модальных окон. Для старых версий использовать `st.columns` и условное отображение.

2. **PDF.js**: Используется CDN версия. Для offline — скачать локально и обновить URL.

3. **Безопасность**: Base64 кодирование PDF для безопасной передачи в HTML.

4. **Производительность**: Кэширование PDF списков с помощью `@st.cache_data`.

5. **Адаптивность**: Макет автоматически адаптируется к размеру окна (использует `st.columns`).

---

## 🚀 Будущие улучшения

- [ ] Экспорт переписки в PDF/Word
- [ ] Сохранение сессий
- [ ] Темная тема
- [ ] Поддержка изображений в ответах
- [ ] Кэширование PDF просмотров
- [ ] Поддержка других форматов (DOCX, PPT, etc.)
