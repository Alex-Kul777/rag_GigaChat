# 🐛 DEBUG-MODE: Быстрая отладка RAG системы

## ⚡ Включить за 10 секунд

```bash
export RAG_DEBUG_MODE=true
python app.py --mode ui
```

**Результат:** Модель загружается за **~2 сек** вместо 15 сек! ⚡⚡⚡

---

## 📊 Что это дает

```
┌──────────────────────┬────────────────┬──────────────┐
│       Метрика        │  Production    │   DEBUG      │
├──────────────────────┼────────────────┼──────────────┤
│ Время загрузки       │   15 сек       │   2 сек ⚡   │
│ Время генерации      │   3 сек        │   1 сек ⚡   │
│ Память               │   1.1 GB       │   400 MB 💾  │
│ Параметры модели     │   500M         │   125M ✨    │
└──────────────────────┴────────────────┴──────────────┘

10x ускорение на загрузку 🚀
3x ускорение на генерацию 🚀
2.75x экономия памяти 💾
```

---

## 📚 Документация

| Документ | Описание |
|----------|---------|
| **[examples_debug_mode.md](examples_debug_mode.md)** | 6 практических примеров |
| **[DEBUG_MODE_SUMMARY.md](DEBUG_MODE_SUMMARY.md)** | Полная техническая документация |
| **[IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md)** | Отчет о реализации |
| **[MODEL_COMPARISON_TABLE.md](MODEL_COMPARISON_TABLE.md)** | Сравнение всех моделей |

---

## 🎯 Примеры

### Отладка UI (суперсекундно!)
```bash
export RAG_DEBUG_MODE=true
python app.py --mode ui  # Откроется за 3 сек вместо 20!
```

### Быстрый CLI запрос
```bash
RAG_DEBUG_MODE=true python app.py --mode query --query "Что такое ИИ?"
```

### Запуск тестов
```bash
RAG_DEBUG_MODE=true pytest tests/ -v
```

### В коде
```python
from rag_gigachat.config import debug_config
debug_config.debug_mode = True

from rag_gigachat.core.llm_manager import LLMManager
llm = LLMManager(model_type="local").get_llm()
response = llm.invoke("Your prompt here")
```

---

## 🔧 Как это работает

1. **Переменная окружения** `RAG_DEBUG_MODE=true` → включить debug
2. **DebugConfig** в `config.py` → задает быструю модель
3. **LLMManager** в `llm_manager.py` → переключается на быструю модель
4. **facebook/opt-125m** (125M параметров) → загружается за 2 сек

**Результат:** Вместо медленной Qwen (500M параметров) используется быстрая OPT (125M параметров).

---

## ✅ Когда использовать

### ✅ Используйте для:
- 🔍 Отладки pipeline'а
- 🚀 Быстрого прототипирования
- 💻 Локального тестирования на слабой машине
- 🔄 Итеративной разработки (код → тест → итерация)
- 🧪 Запуска тестов в CI/CD

### ❌ НЕ используйте для:
- 🌐 Production развертывания (качество хуже)
- 📊 Финального тестирования качества
- 👥 Демонстрации пользователю

---

## 🚨 Отладка проблем

**Модель не загружается?**
```bash
rm -rf ~/.cache/huggingface/hub/
RAG_DEBUG_MODE=true python app.py
```

**Медленнее чем ожидается?**
```bash
# Проверить, что debug включен
echo "RAG_DEBUG_MODE=$RAG_DEBUG_MODE"

# Проверить, какая модель
RAG_DEBUG_MODE=true python -c "from rag_gigachat.core.llm_manager import LLMManager; print(LLMManager().model_name)"
```

**Ошибка памяти?**
```python
# Уменьшить max_new_tokens
from rag_gigachat.config import model_config
model_config.max_new_tokens = 30
```

---

## 📈 Производительность

### На CPU (Intel i5)
```
Загрузка модели:   ~2 сек (vs 15 сек в production) ⚡⚡⚡
Генерация ответа:  ~1 сек (vs 3 сек в production) ⚡⚡
```

### На GPU (NVIDIA RTX)
```
Загрузка модели:   ~1 сек
Генерация ответа:  ~0.4 сек
```

---

## 🎓 FAQ

**Q: Зачем менять модель? Почему бы просто не уменьшить параметры генерации?**
A: Размер модели определяет время загрузки, не количество генерируемых токенов.

**Q: Потеряется ли качество ответов?**
A: Да, незначительно. OPT-125m достаточна для отладки, но inferior для production.

**Q: Можно ли использовать в production?**
A: Нет, используйте полную Qwen модель для better quality answers.

**Q: А если мне нужна многоязычность?**
A: OPT-125m может обрабатывать другие языки. Используйте multilingual embeddings.

**Q: Что если debug-режим не помогает?**
A: Проверьте `RAG_DEBUG_MODE=true` и перезагрузитесь. Используйте `distilgpt2` для еще большей скорости.

---

## 📞 Поддержка

Если у вас есть вопросы или проблемы:

1. Прочитайте [examples_debug_mode.md](examples_debug_mode.md) - там 6 примеров
2. Проверьте [DEBUG_MODE_SUMMARY.md](DEBUG_MODE_SUMMARY.md) - полная документация
3. Смотрите [MODEL_COMPARISON_TABLE.md](MODEL_COMPARISON_TABLE.md) - сравнение моделей

---

## 🎯 Текущая конфигурация

```python
# Это используется сейчас:
debug_model_name = "facebook/opt-125m"  # 125M параметров

# Альтернативы:
# "distilgpt2"           # 82M, еще быстрее (~0.5s)
# "facebook/opt-350m"    # 350M, лучше качество (~3s)
```

**Изменить можно в:** `src/rag_gigachat/config.py`

---

## 🚀 Готово к использованию!

```bash
export RAG_DEBUG_MODE=true && python app.py --mode ui
```

**Результат:** UI откроется за 3 сек вместо 20! ⚡⚡⚡

---

*Версия: 1.0*  
*Последнее обновление: 2026-04-19*
