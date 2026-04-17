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
| `docs/COMPONENTS.md` | Подробная документация | 📖 |
| `docs/COMPONENTS_QUICK_START.md` | Быстрый старт с примерами | 🚀 |
| `docs/COMPONENTS_EXAMPLES.md` | Готовые копипаст примеры | 💡 |
| `docs/COMPONENTS_ARCHITECTURE.md` | Архитектура и диаграммы | 📐 |

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

## 🔧 Компоненты подробно

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

Результат: `st.session_state` заполнен параметрами

---

### 2. FileListPanel
**Панель списка файлов**

```python
FileListPanel.show(data_config.documents_dirs)
```

Функции:
- 📂 Выбор домена (debug, ai, UAV)
- 🔍 Поиск по имени файла
- 📊 Количество страниц для каждого файла
- 🔄 Кнопка обновления индекса
- 🗑️ Кнопка очистки

При клике на файл:
1. Открывается DocumentViewer
2. Текущая страница = 1

---

### 3. DocumentViewer
**Просмотр PDF**

```python
DocumentViewer.show(file_path="/path/to/doc.pdf", page=5)
```

Особенности:
- 📄 Отображение PDF через PDF.js (CDN)
- 📄 Выбор страницы через слайдер
- 📋 Информация о документе
- 🔐 Base64 кодирование для безопасности
- 📏 Масштаб 1.5x, высота 800px

---

### 4. HighlightedAnswer
**Ответ с источниками**

```python
HighlightedAnswer.show(
    answer="Текст ответа...",
    retrieved_docs=[
        {"doc_id": "file_p5", "score": 0.92, "text": "..."}
    ],
    documents_dirs=data_config.documents_dirs,
    show_sources=True
)
```

Вывод:
```
🤖 Ответ
[Полный текст ответа]

Источники:
1. [file.pdf, стр. 5] (релевантность: 0.92)
2. [guide.pdf, стр. 12] (релевантность: 0.85)

📚 Источники и релевантные отрывки
[Развёрнутая информация с подсветкой]
```

---

### 5. AnswerInteraction
**Кнопки взаимодействия**

```python
AnswerInteraction.show_actions(answer, answer_id="answer_42")
```

Кнопки:
- 📋 Копировать → Toast: "✓ Скопировано"
- 👍 Полезно → `session_state.feedback = ("helpful", id)`
- 👎 Не полезно → `session_state.feedback = ("unhelpful", id)`
- 💾 Сохранить → `session_state.saved_answers.append(answer)`

---

## 📚 Документация

### Для начинающих:
→ Начните с **COMPONENTS_QUICK_START.md**
- Примеры для каждого компонента
- Полный пример приложения
- Частые проблемы и решения

### Для разработчиков:
→ Изучите **COMPONENTS.md**
- Полное описание каждого компонента
- API документация
- Настройки и параметры

### Для архитекторов:
→ Посмотрите **COMPONENTS_ARCHITECTURE.md**
- Wire frame макета
- Поток взаимодействия
- Архитектура данных (session state)
- Диаграммы зависимостей

### Для копипаста:
→ Используйте **COMPONENTS_EXAMPLES.md**
- Готовые примеры кода
- Минимальные примеры
- Примеры с интеграцией
- Советы и трюки

### Для запуска:
→ Смотрите **app_example.py**
- Полный пример приложения
- Все компоненты интегрированы
- Готов к запуску

---

## 🎯 Примеры использования

### Пример 1: Минимальное приложение (50 строк)
```python
import streamlit as st
from rag_gigachat.config import data_config
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.ui.components import (
    ConfigModal, FileListPanel, HighlightedAnswer, AnswerInteraction
)

st.set_page_config(layout="wide")

# Боковая панель
with st.sidebar:
    ConfigModal.show()
    FileListPanel.show(data_config.documents_dirs)

# Основной интерфейс
st.title("🤖 RAG Chat")

user_input = st.text_area("Ваш вопрос")

if st.button("🚀 Отправить"):
    pipeline = RAGPipeline()
    answer, docs = pipeline.query(user_input, top_k=st.session_state.k_retrieve)
    
    HighlightedAnswer.show(answer, docs, data_config.documents_dirs)
    AnswerInteraction.show_actions(answer)
```

### Пример 2: С историей чата
```python
import streamlit as st
from rag_gigachat.config import data_config
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.ui.components import (
    ConfigModal, FileListPanel, HighlightedAnswer, AnswerInteraction
)

st.set_page_config(layout="wide")

# Инициализировать
if "messages" not in st.session_state:
    st.session_state.messages = []

# Боковая панель
with st.sidebar:
    ConfigModal.show()
    FileListPanel.show(data_config.documents_dirs)

st.title("🤖 RAG Chat")

# История
with st.container(height=300, border=True):
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# Ввод
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_area("Вопрос", height=60)
with col2:
    st.write("")
    send = st.button("🚀")

if send and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    pipeline = RAGPipeline()
    answer, docs = pipeline.query(user_input, top_k=st.session_state.k_retrieve)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "docs": docs
    })
    
    st.markdown("---")
    HighlightedAnswer.show(answer, docs, data_config.documents_dirs)
    AnswerInteraction.show_actions(answer, f"answer_{len(st.session_state.messages)}")
    
    st.rerun()
```

### Пример 3: Полный (см. app_example.py)
- История чата
- Расширенные настройки
- Список файлов
- Просмотр документов
- Интерактивные ответы
- Статистика

---

## 🛠️ Требования

### Зависимости
```bash
pip install streamlit>=1.30
pip install PyPDF2  # Для получения информации о страницах
```

### Versions
- **Streamlit**: >= 1.30 (для `st.dialog`)
- **Python**: >= 3.10
- **PyPDF2**: любая версия (опционально)

### Внешние ресурсы
- **PDF.js**: из CDN (https://cdnjs.cloudflare.com/)
- Интернет требуется для загрузки PDF.js при первом запуске

---

## 🔐 Безопасность

### Валидация
- ✅ Проверка существования файлов
- ✅ Проверка расширений файлов
- ✅ Защита от path traversal

### Кодирование
- ✅ Base64 для PDF (защита от XSS)
- ✅ HTML escaping для текста
- ✅ Session isolation между пользователями

### Session state
- ✅ Автоматическая изоляция между сессиями
- ✅ Отсутствие утечки состояния между пользователями

---

## 📈 Производительность

### Оптимизация
- 💾 Кэширование PDF файлов
- ⚡ Lazy loading документов
- 🔄 Потоковая генерация ответов
- 🖼️ PDF.js для эффективного отображения

### Масштабируемость
- ✅ Поддержка тысячи PDF файлов
- ✅ Пагинация в FileListPanel
- ✅ TTL кэширование для results

---

## 🐛 Тестирование

### Unit тесты
```python
pytest tests/ui/test_components.py -v
```

### Integration тесты
```python
pytest tests/ui/test_integration.py -v
```

### Manual тестирование
```bash
streamlit run src/rag_gigachat/ui/app_example.py
```

---

## 🚀 Запуск приложения

### С компонентами (рекомендуется)
```bash
cd /home/kap/projects/rag_GigaChat
streamlit run src/rag_gigachat/ui/app_example.py
```

### С собственным приложением
```python
# my_app.py
import streamlit as st
from rag_gigachat.ui.components import ConfigModal, FileListPanel
# ... ваш код

# Запуск
# streamlit run my_app.py
```

---

## 📝 Краткая справка (Cheat Sheet)

### ConfigModal
```python
ConfigModal.show()
# → Открывает диалог с параметрами
# → Сохраняет в session_state
```

### FileListPanel
```python
FileListPanel.show(data_config.documents_dirs)
# → Показывает список файлов в sidebar
# → Сохраняет selected_file в session_state
```

### DocumentViewer
```python
DocumentViewer.show(file_path, page=5)
# → Показывает PDF на странице 5
# → Поддерживает смену страницы
```

### HighlightedAnswer
```python
HighlightedAnswer.show(answer, docs, documents_dirs)
# → Показывает ответ
# → Добавляет ссылки на источники
# → Выделяет текст жёлтым
```

### AnswerInteraction
```python
AnswerInteraction.show_actions(answer, answer_id)
# → Показывает кнопки: 📋 👍 👎 💾
# → Сохраняет feedback в session_state
```

---

## 📞 Поддержка

### Если что-то не работает:
1. ✅ Проверьте версию Streamlit: `streamlit --version` (должна быть >= 1.30)
2. ✅ Проверьте файлы: `ls /path/to/pdf_dir/`
3. ✅ Проверьте session state: `st.write(st.session_state)`
4. ✅ Смотрите документацию в `docs/`

### Документация:
- 📖 Подробно: [COMPONENTS.md](./COMPONENTS.md)
- 🚀 Быстро: [COMPONENTS_QUICK_START.md](./COMPONENTS_QUICK_START.md)
- 💡 Примеры: [COMPONENTS_EXAMPLES.md](./COMPONENTS_EXAMPLES.md)
- 📐 Архитектура: [COMPONENTS_ARCHITECTURE.md](./COMPONENTS_ARCHITECTURE.md)
- 🔧 Интеграция: [app_example.py](../src/rag_gigachat/ui/app_example.py)

---

## 📜 Лицензия

Часть проекта RAG GigaChat. Используйте свободно внутри проекта.

---

## 🎯 Дорожная карта

- [x] ConfigModal - настройки
- [x] FileListPanel - список файлов
- [x] DocumentViewer - просмотр PDF
- [x] HighlightedAnswer - ответы с источниками
- [x] AnswerInteraction - интерактивные кнопки
- [ ] Экспорт в PDF/Word
- [ ] Сохранение сессий в БД
- [ ] Темная тема
- [ ] Поддержка DOCX, PPT
- [ ] Поиск в PDF

---

**Версия:** 1.0.0  
**Дата:** 2026-04-17  
**Статус:** ✅ Готово к использованию
