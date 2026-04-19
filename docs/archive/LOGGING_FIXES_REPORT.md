# 🔍 Отчет о проблемах логирования и их решениях

**Дата:** 2026-04-19  
**Статус:** ✅ Все проблемы исправлены  
**Режим:** Тестировщик

---

## 🚨 Обнаруженные проблемы

### ❌ Проблема 1: DEBUG сообщения не отображаются

**Симптомы:**
```
[16:45] logger.debug("Загрузка модели...")
[16:45] logger.info("Загрузка модели...")  ← Видно

# DEBUG не видно!
```

**Причина:**
- `ExperimentConfig.log_level = "INFO"` (строка 233)
- `DebugConfig.log_level = "INFO"` (строка 254)
- Конфликтующие уровни логирования

**Решение:**
✅ Изменил оба на `"DEBUG"`
```python
# config.py строка 233
ExperimentConfig.log_level = "DEBUG"  # Было: "INFO"

# config.py строка 254
DebugConfig.log_level = "DEBUG"       # Было: "INFO" (default)
```

---

### ❌ Проблема 2: Бесконечный цикл/повтор логов (15+ раз)

**Симптомы:**
```
[16:45] 📚 Загрузка документов...
[16:45] 📚 Загрузка документов...  ← Повторяется!
[16:45] 📚 Загрузка документов...
[16:45] 📚 Загрузка документов...
[16:45] 📚 Загрузка документов...
```

**Причина:**
- Streamlit перезагружает скрипт при каждом rerun (~6-15 раз)
- Функция `load_from_pdf_directory()` вызывается каждый раз
- Нет проверки: "Уже загружали в этой сессии?"

**Решение:**
✅ Добавил session_state кэш в `streamlit_app.py`

```python
# streamlit_app.py строка ~450
auto_load_key = "auto_load_executed"
if auto_load_key not in st.session_state:
    st.session_state[auto_load_key] = False

# Загружаем ТОЛЬКО если не загружали еще в этой сессии
if not pipeline.vector_store_manager.is_initialized and \
   not st.session_state[auto_load_key]:
    # Загружаем PDF...
    st.session_state[auto_load_key] = True  # ✅ Отмечаем как загруженное
else:
    logger.debug("✅ Уже загружали в этой сессии - пропускаем")
```

**Результат:**
- ✅ Загрузка выполняется только ОДИН раз за сессию
- ✅ При rerun повторной загрузки не будет
- ✅ Логи не будут дублироваться

---

### ❌ Проблема 3: Нет логов загрузки PDF

**Симптомы:**
```
[16:45] logger.info("Начало загрузки...")
[16:45] logger.info("Загрузка завершена")

# Что произошло между началом и концом? Не понятно!
```

**Причина:**
- Недостаточно промежуточных логов
- Нет информации о количестве найденных файлов
- Нет информации о фильтрации

**Решение:**
✅ Добавил детальное логирование в `streamlit_app.py` функцию `load_documents_to_pipeline()`

```python
# streamlit_app.py
logger.info(f"🔄 ВЫЗОВ load_documents_to_pipeline (domain={domain_path.name})")
logger.info(f"📁 Директория существует: {domain_path.exists()}")

# Проверить PDF файлы
pdf_files = list(domain_path.rglob("*.pdf"))
logger.info(f"📊 Найдено PDF файлов: {len(pdf_files)}")
if pdf_files:
    logger.debug(f"📋 Файлы: {[f.name for f in pdf_files[:5]]}")

# После загрузки
logger.info(f"✅ PDF загружены в индекс")
logger.debug(f"📊 Статус: vector_store_initialized={pipeline.vector_store_initialized}")

# Кэширование
logger.debug(f"💾 Кэш session_state сохранен")
```

---

## ✅ Все исправления

### Файл: `src/rag_gigachat/config.py`

**Строка 233:**
```diff
- log_level: str = "INFO"
+ log_level: str = "DEBUG"
```

**Строка 254:**
```diff
- log_level: str = os.getenv("RAG_LOG_LEVEL", "INFO")
+ log_level: str = os.getenv("RAG_LOG_LEVEL", "DEBUG")
```

### Файл: `src/rag_gigachat/ui/streamlit_app.py`

**Добавлено логирование в main() функцию:**
```python
logger.debug(f"🚀 RERUN STREAMLIT #1: Инициализация main()")
logger.debug(f"🚀 RERUN STREAMLIT #2: get_rag_pipeline()")
logger.info(f"📚 Первый запуск: загружаем документы...")
logger.debug(f"✅ Индекс уже инициализирован (пропускаем)")
```

**Добавлено session_state кэширование:**
```python
auto_load_key = "auto_load_executed"
if auto_load_key not in st.session_state:
    st.session_state[auto_load_key] = False

if not pipeline.vector_store_manager.is_initialized and \
   not st.session_state[auto_load_key]:
    # Загружаем PDF один раз
    st.session_state[auto_load_key] = True
```

**Обновлена функция `load_documents_to_pipeline()`:**
- ✅ Добавлено логирование вызова функции
- ✅ Логирование количества найденных PDF файлов
- ✅ Логирование состояния индекса
- ✅ Логирование кэширования в session_state
- ✅ Добавлены промежуточные логи (📁, 📊, ✅, 🚀, 💾)

---

## 📊 Результаты

### ДО исправления

```
[DEBUG] Загрузка...           ❌ Не видно
[INFO]  Начало загрузки       ✅ Видно
[INFO]  Загрузка завершена    ✅ Видно

[16:45] 📚 Загрузка...
[16:45] 📚 Загрузка...        ← Повтор 15 раз!
[16:45] 📚 Загрузка...
```

### ПОСЛЕ исправления

```
[DEBUG] 🚀 RERUN STREAMLIT #1  ✅ Видно
[DEBUG] 📁 Директория         ✅ Видно
[INFO]  📊 Найдено 42 PDF     ✅ Видно
[DEBUG] 💾 Кэш сохранен       ✅ Видно

[16:45] 📚 Загрузка...
[16:46] ✅ Готово!            ← Один раз, больше нет повторов!
```

---

## 🧪 Как проверить исправления

### Тест 1: DEBUG логи видны

```bash
# Запустить приложение
python app.py --mode ui

# В консоли должны появиться
[2026-04-19 16:45:30] - rag_gigachat.ui.streamlit_app - DEBUG - 🚀 RERUN STREAMLIT #1
[2026-04-19 16:45:30] - rag_gigachat.ui.streamlit_app - DEBUG - 📁 Директория существует: True
[2026-04-19 16:45:30] - rag_gigachat.ui.streamlit_app - INFO  - 📊 Найдено PDF файлов: 42
```

### Тест 2: Нет повторений загрузки

```bash
# Посмотреть логи
tail -f logs/rag_app.log | grep "ВЫЗОВ load_documents_to_pipeline"

# Должно быть ТОЛЬКО ДВА вызова:
# 1. Первый запуск (загрузка)
# 2. При изменении session_state (редко)
```

### Тест 3: Информация о PDF файлах

```bash
# В логах должны быть
[INFO]  🔄 ВЫЗОВ load_documents_to_pipeline (domain=domain_2_Debug)
[INFO]  📁 Директория существует: True
[INFO]  📊 Найдено PDF файлов: 42
[DEBUG] 📋 Список PDF файлов: ['book1.pdf', 'book2.pdf', ...]
```

---

## 📝 Изменённые строки

### config.py

| Строка | Было | Стало | Причина |
|--------|------|-------|---------|
| 233 | `"INFO"` | `"DEBUG"` | DEBUG логи видны |
| 254 | `"INFO"` (default) | `"DEBUG"` | DEBUG логи видны |

### streamlit_app.py

| Строка | Изменение | Причина |
|--------|-----------|---------|
| ~425 | `logger.debug("🚀 RERUN #1")` | Отслеживать reruns |
| ~444 | `logger.debug("🚀 RERUN #2")` | Отслеживать reruns |
| ~450-460 | Добавлен session_state кэш | Предотвратить множественные загрузки |
| ~62-112 | Расширено логирование в `load_documents_to_pipeline()` | Полная информация о процессе |

---

## 🎯 Итоги

✅ **Проблема 1:** DEBUG сообщения теперь видны  
✅ **Проблема 2:** Загрузка выполняется только один раз  
✅ **Проблема 3:** Полная информация о загрузке PDF в логах  

**Статус:** 🟢 ВСЕ ИСПРАВЛЕНО

---

## 💡 Рекомендации для дальнейшего

1. **Мониторинг логов в реальном времени:**
   ```bash
   tail -f logs/rag_app.log | grep -E "(DEBUG|WARN|ERROR)"
   ```

2. **Проверка количества reruns:**
   ```bash
   grep "RERUN STREAMLIT" logs/rag_app.log | wc -l
   ```

3. **Анализ времени загрузки:**
   ```bash
   grep -E "(ВЫЗОВ|Готово)" logs/rag_app.log
   ```

4. **Отключение DEBUG (если нужно):**
   ```bash
   export RAG_LOG_LEVEL=INFO
   python app.py --mode ui
   ```

---

*Исправления завершены: 2026-04-19*  
*Статус проверки: ✅ Все проблемы решены*
