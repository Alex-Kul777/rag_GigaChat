---
name: improvements_codemurt.md
description: Улучшить существующие RAG решение на основе github.com/codemurt/rag_system
---

# Список улучшений для RAG

## Очередь улучшений (добавляйте сюда замечания):

### Приоритет 1 (критические):
- [x] Добавить Docker-упаковку — создать Dockerfile и docker-compose.yml для упрощения развёртывания (как у codemurt)
- [x] Добавить поддержку OCR для сканированных PDF — использовать Docling или pytesseract + pdf2image (аналог функции из codemurt)

### Приоритет 2 (важные):
- [x] Добавить прогресс-бар при загрузке больших PDF в Streamlit UI (асинхронная обработка с st.progress)
- [x] Включить логирование в файл — активировать logging_config.log_to_file = True в config.py (уже есть, но выключено)
- [x] Добавить CI/CD через GitHub Actions — автоматический запуск тестов и проверка покрытия при push

### Приоритет 3 (косметические):
- [x] Добавить бейджики в README (лицензия, версия Python, тесты, coverage)
- [x] Указать номера страниц в ответе — при ответе показывать не только имя файла, но и страницу источника
- [ ] Добавить скриншоты интерфейса в README (чат, эксперименты, отчёты)

## Примеры замечаний:
- Увеличить chunk_size для лучшего контекста — сейчас 800, для длинных документов нужно 1000-1200
- Сдвинуть кнопку "Очистить историю" в правую часть чата
- Добавить возможность выбора модели эмбеддингов через UI (сейчас только через config)

---

## Детальные предложения с кодом

### 1. Docker-упаковка

Было: отсутствие Docker.

Стало: добавлены файлы для контейнеризации.

Новые файлы:

Файл: Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей (для PDF-обработки)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ui_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]


Файл: docker-compose.yml

version: '3.8'

services:
  rag-gigachat:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./experiments:/app/experiments
      - ./logs:/app/logs
    restart: unless-stopped


### 2. OCR для сканированных PDF

Было: только текстовые PDF.

Стало: поддержка OCR через docling.

Изменения в data_loader.py (добавить функцию):

try:
    from docling.document_converter import DocumentConverter
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def load_pdf_with_ocr(pdf_path: Path) -> str:
    if not OCR_AVAILABLE:
        logger.warning("Docling not installed, OCR disabled")
        return ""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_text()


В методе CorpusLoader.load_from_pdf_directory добавить проверку:

# Если обычный PyPDFLoader не дал текста, пробуем OCR
if not text.strip() and OCR_AVAILABLE:
    text = load_pdf_with_ocr(pdf_path)


### 3. Прогресс-бар для загрузки PDF

Было: загрузка без индикации прогресса (долго).

Стало: добавлен st.progress в ui_streamlit.py.

Изменения в ui_streamlit.py (внутри with st.spinner):

progress_bar = st.progress(0)
for idx, pdf_file in enumerate(uploaded_files):
    # обработка файла
    progress_bar.progress((idx + 1) / len(uploaded_files))
progress_bar.empty()


### 4. Включение логирования в файл

Было: в config.py параметры есть, но не активированы (логи только в консоль).

Стало: изменён config.py.

Файл: config.py (изменённая секция)

@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    log_level: str = "DEBUG"
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_date_format: str = '%Y-%m-%d %H:%M:%S'
    log_to_file: bool = True   # было False
    log_to_console: bool = True
    log_file_name: str = "rag_app.log"


### 5. CI/CD через GitHub Actions

Было: отсутствует.

Стало: добавлен файл .github/workflows/test.yml.

name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-fail-under=60
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml
        fail_ci_if_error: false


### 6. Номера страниц в ответе

Было: в ответе только имя файла.

Стало: в rag_core.py при создании retrieved_docs сохранять номер страницы.

Изменение в _build_graph (функция retrieve):

# Сохраняем номер страницы из метаданных документа
page_num = doc.metadata.get('page', 0)
retrieved_docs.append({
    'doc_id': doc_id,
    'score': score,
    'text': doc.page_content,
    'page': page_num
})


В ui_streamlit.py при отображении источников:

st.markdown(f"**Источник {i}:** `{doc['doc_id']}` (стр. {doc.get('page', '?')}, score: {doc['score']:.3f})")


## Сравнение ключевых параметров (было / стало)

| Параметр | Было | Стало |
|----------|------|-------|
| Docker | отсутствует | Dockerfile + docker-compose.yml |
| OCR PDF | нет | поддержка Docling (опционально) |
| Прогресс-бар | нет | есть в Streamlit UI |
| Логирование в файл | выключено | включено (log_to_file=True) |
| CI | нет | GitHub Actions + coverage |
| Номера страниц | не отображаются | отображаются в источниках |
| chunk_size (пример) | 800 | 1000 (рекомендуется для длинных документов) |

## Заключение

Предложенные улучшения закрывают основные недостатки, выявленные при сравнении с codemurt/rag_system, и усиливают сильные стороны вашего проекта. Приоритет 1 — Docker и OCR — критичны для портативности и полноты обработки документов. Приоритет 2 повышает удобство и надёжность. Косметические улучшения (приоритет 3) сделают проект более привлекательным для пользователей.
