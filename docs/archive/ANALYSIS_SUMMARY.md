# 🔴 Критические области для отладки: Потеря данных при загрузке

## Ключевые строки где теряются данные

### 1️⃣ data_loader.py:411-413 — PyPDF извлечение БЕЗ очистки

```python
# ❌ ПРОБЛЕМА: text используется БЕЗ нормализации
loader = PyPDFLoader(str(pdf_path))
documents = loader.load()  # Contains raw text with artifacts
total_text = "".join(doc.page_content for doc in documents)
```

**Что находится в `page_content`:**
- Множественные пробелы (2-10 подряд)
- Переносы строк внутри слов: `при\nнять`
- Табуляции: `\t`
- No-break space: `\u00A0`
- Артефакты PDF: странные символы, разделители

---

### 2️⃣ data_loader.py:718 — Метаданные сохраняют грязный текст

```python
# ❌ ПРОБЛЕМА: page_content передается в метаданные как есть
result[doc_id] = {
    'text': doc.page_content,  # ← БЕЗ очистки!
    'metadata': {...}
}
```

**Следствие:** Эмбеддинги будут созданы из нечищенного текста!

---

### 3️⃣ vector_store.py:257 — Эмбеддинги из грязного текста

```python
# ❌ КРИТИЧНО: documents содержат нечищенный page_content
self.vector_store = FAISS.from_documents(documents, self.embeddings)
```

**Цепочка:**
```
PDF → PyPDFLoader → page_content (грязный) 
  → split_documents() → TextSplitter на грязном тексте
  → FAISS.from_documents() → embeddings на грязных чанках
  → Низкое качество поиска!
```

---

### 4️⃣ data_loader.py:178 — OCR также без нормализации

```python
# ❌ ПРОБЛЕМА: OCR output используется как есть
text = result.document.export_to_text()  # Может содержать артефакты
cache_path.write_text(text, encoding="utf-8")
```

---

## Текущие параметры (config.py)

```python
# Строка 172-174
chunk_size: int = 500
chunk_overlap: int = 80
chunk_separators: List[str] = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
```

**Проблема с сепаратором `"\n"`:**
- Если в тексте есть переносы **внутри слов** → TextSplitter разбивает их неправильно
- Множественные пробелы → растрата символов на whitespace вместо контента

---

## Функции которых НЕ существует

| Функция | Где нужна | Зачем |
|---------|-----------|-------|
| `normalize_text()` | data_loader.py | Очистка whitespace, переносов, артефактов |
| `clean_pdf_text()` | load_pdf_with_metadata() | Нормализация после PyPDFLoader |
| `clean_ocr_text()` | load_pdf_with_ocr() | Нормализация после OCR |

---

## Потенциальные потери данных

### 1. Разреженность в эмбеддингах
```
Грязный текст:
"Это   важное  слово"  (5 пробелов)
vs
"Это важное слово"  (2 пробела)

При embedding → разные векторы!
Векторные пространства не совпадают → поиск не находит релевантные документы
```

### 2. Разбитые слова при чанкировании
```
Текст: "При\nнять решение"
TextSplitter видит: "При" + перенос + "нять решение"
Чанк 1: "При"
Чанк 2: "нять решение"

Поиск не найдет "принять" → потеря смысла
```

### 3. Потеря контекста при неправильном разбиении
```
500 символов с 10 пробелами подряд вместо 1:
- Реальный контент: 420 символов
- Потраченные на пробелы: 80 символов

TextSplitter видит 500 символов → разбивает на границе контента
Результат: рваные чанки, потеря информации
```

---

## ✅ Рекомендованное решение

### Шаг 1: Добавить нормализацию

```python
import re

def normalize_text(text: str) -> str:
    """Очистка текста из PDF для улучшения эмбеддингов."""
    text = re.sub(r' {2,}', ' ', text)  # Множественные пробелы → один
    text = text.replace('\t', ' ')       # Табуляции → пробел
    text = re.sub(r'\n{3,}', '\n\n', text)  # Множественные переносы → два
    text = re.sub(r'([а-яa-z])\n([а-яa-z])', r'\1\2', text)  # Переносы внутри слов
    return text.strip()
```

### Шаг 2: Применить в load_pdf_with_metadata (после строки 412)

```python
for doc in documents:
    doc.page_content = normalize_text(doc.page_content)
    logger.info(f"PDF нормализован: {len(doc.page_content)} символов")
```

### Шаг 3: Применить в load_pdf_with_ocr (после строки 178)

```python
text = normalize_text(result.document.export_to_text())
```

---

## Проверка эффекта

**Команда для тестирования:**
```bash
# Включить DEBUG логирование
export LOG_LEVEL=DEBUG
.venv/bin/python app.py --mode ui

# Проверить создать temp/ директорию с текстовыми файлами
# Сравнить original text vs cleaned text
```

**Ожидаемый результат:**
- Размер текста уменьшится на 15-30% (нет лишних пробелов)
- Чанки будут более компактными и смысловыми
- Эмбеддинги улучшатся → лучший поиск
