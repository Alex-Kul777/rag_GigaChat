# RAG Система - Система Извлечения и Генерации Информации

![Tests](https://github.com/Alex-Kul777/rag_GigaChat/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-60%25%2B-yellowgreen)

## 📋 Описание

Этот проект реализует систему **Retrieval-Augmented Generation (RAG)** для ответов на вопросы на основе PDF документов. Система объединяет поиск документов с генерацией на основе большой языковой модели для предоставления точных, контекстуальных ответов.

## 🚀 Возможности

- **Обработка документов**: Загрузка PDF документов с извлечением метаданных и кэшированием
- **Векторный поиск**: Семантический поиск на основе FAISS с настраиваемыми эмбедингами
- **Гибридный поиск**: Поддержка плотного, разреженного и гибридного поиска
- **Интеграция LLM**: Поддержка локальных моделей (HuggingFace) и GigaChat
- **Фреймворк экспериментов**: Запуск экспериментов с различными конфигурациями
- **Метрики оценки**: 
  - Поиск: MAP, MRR, Precision@k, Recall@k, NDCG@k
  - Генерация: ROUGE, BLEU, BERTScore
  - Продвинутые: Верность, Релевантность ответа, Релевантность контекста (RAGAS)
- **Поддержка OCR**: Автоматический откат на Docling OCR для сканированных PDF с кэшированием и ограничениями размера
  - Настраиваемые ограничения размера файла и пороги символов
  - Кэширование на основе MD5 для избежания повторной обработки
- **Модульная архитектура**: Чистое разделение ответственности с паттернами Strategy и Dependency Injection
  - `vector_store.py`: Абстракция хранилища векторов с бэкэндом FAISS и валидацией входных данных
  - `llm_manager.py`: Абстракция поставщика LLM (GigaChat, HuggingFace, OpenAI)
  - `retriever.py`: Подключаемые стратегии поиска (Dense, Sparse, Hybrid)
  - `token_counter.py`: Независимый трекинг токенов с управлением балансом
- **Валидация входных данных**: Проверки перед запросами, файлами и конфигурацией API
- **Набор тестов**: 33 автоматизированных теста с pytest, mock'ами и отчётами о покрытии
- **Docker**: Развёртывание одной командой через `docker-compose up`
- **CI/CD**: GitHub Actions — автоматический запуск тестов и отчёты о покрытии при каждом push
- **Веб-интерфейс**: Streamlit-интерфейс с возможностью загрузки PDF и прогресс-баром
- **UI компоненты**: Переиспользуемые Streamlit компоненты для RAG чата
  - ConfigModal для расширенных настроек
  - FileListPanel для управления документами
  - DocumentViewer для просмотра PDF
  - HighlightedAnswer для подсветки источников
  - AnswerInteraction для обратной связи пользователя
- **Отчёты**: Экспорт результатов в Excel с резюме экспериментов

## 📁 Структура проекта

```
.
├── app.py                      # Точка входа (режимы UI, query, experiment)
├── config.py                   # Централизованная конфигурация
├── data_loader.py              # Загрузка документов с кэшированием и OCR
├── rag_core.py                 # Основной RAG пайплайн с LangGraph
├── vector_store.py             # Менеджер хранилища векторов (FAISS)
├── llm_manager.py              # Абстракция поставщика LLM
├── retriever.py                # Стратегии поиска (плотный, разреженный, гибридный)
├── token_counter.py            # Трекинг токенов и управление балансом
├── models.py                   # Модели данных (dataclasses, enums)
├── validator.py                # Валидация входных данных
├── evaluator.py                # Метрики оценки (RAGAS, custom)
├── experiment.py               # Запуск экспериментов с метриками
├── excel_reporter.py           # Генерация отчётов в Excel
├── ui_streamlit.py             # Веб-интерфейс Streamlit с загрузкой PDF
├── create_wikieval_dataset.py  # Утилита для создания датасетов
└── requirements.txt            # Зависимости Python
```

## 🛠️ Установка

### Предварительные требования

- Python 3.10+
- [API ключ GigaChat](https://developers.sber.ru/) (опционально для моделей GigaChat)

### Процесс установки

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Alex-Kul777/rag_GigaChat.git
cd rag_GigaChat
```

2. Создайте виртуальное окружение:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Настройте переменные окружения:
```bash
cp .env.example .env
# Отредактируйте .env с вашим API ключом GigaChat
```

### Конфигурация

Система использует централизованную конфигурацию в `config.py`:

```python
# Параметры модели
model_config.llm_model_name = "GigaChat-2-Max"
model_config.embedding_model_name = "GigaChat-2-Max"
model_config.temperature = 0.7
model_config.default_k_retrieve = 5

# Параметры данных
data_config.chunk_size = 800
data_config.chunk_overlap = 100
data_config.documents_dirs = {
    "debug": Path("data/domain_2_Debug/books"),
    "ai": Path("data/domain_1_AI/books"),
    "test": Path("data/test_docs")
}

# Параметры OCR
data_config.ocr_enabled = True
data_config.ocr_max_file_size_mb = 50
data_config.ocr_min_chars_per_page = 50

# Параметры GigaChat
gigachat_config.api_key = os.getenv("GIGACHAT_API_KEY", "")
gigachat_config.model = "GigaChat-2-Max"
```

## 🚦 Использование

### 1. Веб-интерфейс (рекомендуется)

```bash
python app.py --mode ui
```
Затем откройте `http://localhost:8501` в браузере.

### 2. Режим одиночного запроса

```bash
python app.py --mode query --query "Что такое RAG?" --retrieval_type dense --k 5
```

Опции:
- `--query`: Текст вопроса
- `--retrieval_type`: `dense`, `sparse` или `hybrid`
- `--k`: Количество документов для поиска
- `--output`: Сохранить результат в JSON файл

### 3. Режим экспериментов

```bash
python app.py --mode experiment --testset testset.json --experiment_name my_exp --retrieval_type hybrid
```

Опции:
- `--testset`: Путь к JSON файлу с тестовыми запросами
- `--experiment_name`: Имя для этого эксперимента
- `--retrieval_type`: Метод поиска
- `--k`: Количество документов для поиска
- `--output_dir`: Директория для результатов

## 🧪 Запуск тестов

Запустите полный набор тестов (33 теста):

```bash
# Все тесты
pytest -v
# или через Makefile
make test

# С отчётом о покрытии (цель: 60%)
make test-cov

# По категориям
make test-unit    # только unit тесты
make test-smoke   # только smoke тесты
```

## 🔧 Опции конфигурации

### Конфигурация модели

| Параметр | Описание | По умолчанию |
|----------|---------|--------------|
| `llm_model_name` | Название LLM модели | `GigaChat-2-Max` |
| `embedding_model_name` | Модель эмбедингов | `GigaChat-2-Max` |
| `max_new_tokens` | Максимум токенов для генерации | `1000` |
| `temperature` | Случайность генерации | `0.7` |
| `default_k_retrieve` | Документов для поиска по умолчанию | `5` |
| `device` | Устройство вычисления | `cpu` |

### Конфигурация данных

| Параметр | Описание | По умолчанию |
|----------|---------|--------------|
| `chunk_size` | Размер куска документа | `800` |
| `chunk_overlap` | Перекрытие между кусками | `100` |
| `force_reload` | Перезагрузить документы из источника | `False` |
| `cache_enabled` | Включить кэширование документов | `True` |
| `ocr_enabled` | Включить OCR для сканированных PDF | `True` |
| `ocr_max_file_size_mb` | Максимальный размер файла для OCR | `50` |

## 🐳 Docker

Запустите всё приложение одной командой:

```bash
# Скопируйте и заполните ваш API ключ
cp .env.example .env

# Соберите и запустите
docker-compose up --build
```

Затем откройте `http://localhost:8501` в браузере.

Директории данных (`data/`, `experiments/`, `logs/`) монтируются как volumes — ваши документы и результаты сохраняются между перезагрузками контейнера.

## 📊 Метрики оценки

### Метрики поиска
- **MAP** (Mean Average Precision): Средняя точность по всем запросам
- **MRR** (Mean Reciprocal Rank): Позиция первого релевантного документа
- **Precision@k**: Доля релевантных документов в топ-k
- **Recall@k**: Доля найденных релевантных документов в топ-k
- **NDCG@k**: Нормализованный Дисконтированный Кумулятивный Прирост

### Метрики генерации
- **ROUGE**: Пересечение с эталонными ответами
- **BLEU**: Точность N-граммов
- **BERTScore**: Семантическое сходство с BERT

### Продвинутые метрики RAGAS
- **Faithfulness**: Факт. точность ответа
- **Answer Relevancy**: Релевантность ответа вопросу
- **Context Relevancy**: Релевантность контекста вопросу

## 🔄 Рабочий процесс

1. **Загрузка документов**: PDF загружаются, разбиваются на куски и сохраняются в FAISS
2. **Обработка запроса**: Запрос пользователя конвертируется в эмбеддинг для поиска
3. **Поиск**: Топ-k релевантных кусков извлекаются из FAISS
4. **Построение контекста**: Найденные куски объединяются в контекст
5. **Генерация**: LLM генерирует ответ на основе контекста и вопроса
6. **Оценка**: Метрики рассчитываются для качества поиска и генерации

## 📝 Логирование

Система использует отфильтрованное логирование:
- **Логи консоли**: Только из модулей в `OUR_MODULES` (debug, experiment, rag_core и т.д.)
- **Логи файлов**: Все логи сохраняются в `logs/rag_app.log`
- **Логи сторонних библиотек**: Подавляются по умолчанию

Настройте логирование в `config.py`:
```python
logging_config.log_level = "DEBUG"  # или INFO, WARNING
logging_config.log_to_file = True
logging_config.log_to_console = True
```

## 👨‍💻 Разработка

### Добавление нового источника документов

1. Добавьте метод загрузки документов в `data_loader.py`
2. Обновите метод `CorpusLoader.load_from_*`
3. Добавьте поддержку в `RAGPipeline.load_from_*`

### Добавление новой LLM

1. Расширьте `LLMManager` новым методом загрузки
2. Добавьте конфигурацию в `ModelConfig`
3. Обновите `RAGPipeline` для поддержки нового типа

### Добавление новых метрик

1. Добавьте расчёт метрики в `evaluator.py`
2. Обновите `ExperimentResult` для включения новой метрики
3. Модифицируйте `excel_reporter.py` для отображения новых метрик

## 📄 Лицензия

[MIT License]

## 📧 Контакты

[Алексей К. Telegram @auditor2it]

---

**Замечание**: Эта система требует API ключ GigaChat для использования моделей GigaChat. Для локальных моделей убедитесь, что у вас достаточно памяти (рекомендуется 16GB+ для моделей 7B).
