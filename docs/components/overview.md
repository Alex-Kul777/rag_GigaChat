# 🎨 RAG Chat Streamlit Components

Полный набор переиспользуемых компонентов для построения интеллектуального интерфейса RAG чата.

## 📦 Что включено

### Компоненты (5 штук)
1. **ConfigModal** - Модальное окно с расширенными настройками
2. **FileListPanel** - Панель со списком PDF файлов в сайдбаре
3. **DocumentViewer** - Интерактивный просмотр PDF с поддержкой страниц
4. **HighlightedAnswer** - Ответ с подсветкой источников и ссылками
5. **AnswerInteraction** - Интерактивные кнопки (копирование, оценка, сохранение)

### Файлы проекта

| Файл | Описание | Размер |
|------|---------|--------|
| `src/rag_gigachat/ui/components.py` | Основной модуль с компонентами | ~600 строк |
| `src/rag_gigachat/ui/app_example.py` | Пример полной интеграции | ~400 строк |
| `docs/components/overview.md` | Подробная документация | 📖 |
| `docs/components/quick-start.md` | Быстрый старт с примерами | 🚀 |
| `docs/components/examples.md` | Готовые копипаст примеры | 💡 |
| `docs/components/architecture.md` | Архитектура и диаграммы | 📐 |

---

## 🚀 Быстрый старт (30 секунд)

### Шаг 1: Импортировать компоненты
```python
from rag_gigachat.ui.components import (
    ConfigModal,
    FileListPanel,
    DocumentViewer,
    HighlightedAnswer,
    AnswerInteraction
)
from rag_gigachat.config import data_config
```

### Шаг 2: Добавить в боковую панель
```python
with st.sidebar:
    st.markdown("### 📚 RAG Chat")
    ConfigModal.show()
    FileListPanel.show(data_config.documents_dirs)
```

### Шаг 3: Показать ответ с источниками
```python
# После получения ответа от pipeline
answer, docs = pipeline.query(user_input)

HighlightedAnswer.show(answer, docs, data_config.documents_dirs)
AnswerInteraction.show_actions(answer)
```

### Шаг 4: Включить просмотр документов
```python
if st.session_state.get("show_document_viewer"):
    with st.dialog("Просмотр"):
        DocumentViewer.show(st.session_state.selected_file)
```

**Готово!** 🎉

---

## 📊 Структура макета

```
┌─────────────────────────────────────────────────┐
│           🤖 RAG Chat приложение              │
├──────────────┬──────────────────────────────────┤
│ SIDEBAR      │  MAIN CONTENT                    │
│ (300px)      │  (100% width - 300px)           │
│              │                                  │
│ ⚙️ Config    │  💬 Chat history                │
│ 📁 Files     │  🤖 Answer                      │
│ 🔍 Search    │  📚 Sources                     │
│              │  [📋👍👎💾] Actions             │
└──────────────┴──────────────────────────────────┘
```

---

## 🎨 Компоненты подробно

### 1. ConfigModal
**Модальное окно с настройками**

```python
ConfigModal.show()
```

Параметры:
- 🤖 Модели (LLM, embedding, tokens, temperature)
- 🔍 Поиск (Top-K, max context, retrieval type)
- 📄 Чанкирование (size, overlap)
- 💬 GigaChat (top-p, penalty, OCR)

#### Назначение
Модальное окно с расширенными настройками. Открывается по кнопке "⚙️ Расширенные настройки" на боковой панели.

#### Группы параметров

##### 1️⃣ Модели
- **LLM модель** - выбор модели для генерации ответов
- **Embedding модель** - модель для создания эмбеддингов
- **Max tokens** - максимальное количество токенов в ответе (100-4000)
- **Temperature** - степень "творчества" модели (0-2.0)

##### 2️⃣ Поиск
- **Top-K документов** - количество релевантных документов (1-20)
- **Max контекст** - максимальный размер контекста в символах (500-5000)
- **Тип поиска** - радиокнопки: dense, sparse, hybrid

##### 3️⃣ Чанкирование
- **Размер чанка** - длина текстового фрагмента (100-2000)
- **Перекрытие чанков** - количество перекрывающихся символов (0-500)

##### 4️⃣ GigaChat
- **Top-P** - разнообразие ответов (0-1.0)
- **Penalty повторений** - штраф за повторы (1.0-2.0)
- **OCR для PDF** - включить OCR для сканированных документов (checkbox)

#### Состояние
Все параметры сохраняются в `st.session_state`:

```python
st.session_state.llm_model
st.session_state.temperature
st.session_state.k_retrieve
st.session_state.chunk_size
# и т.д.
```

#### Действия
- **✅ Применить** - сохранить и применить настройки
- **🔄 Сброс** - вернуть значения по умолчанию
- **✕ Закрыть** - закрыть окно без сохранения

---

### 2. FileListPanel

**Правая боковая панель со списком загруженных PDF файлов**

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

---

Полная документация см. в `docs/components/architecture.md`, `docs/components/examples.md`, `docs/components/quick-start.md`.
