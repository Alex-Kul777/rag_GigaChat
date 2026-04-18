#!/usr/bin/env python3
"""
DEBUG скрипт для проверки RAG пайплайна на одном запросе
"""
import sys
import logging
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent))

from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.config import data_config, model_config, logging_config
from rag_gigachat.models import RetrievalType

# 🔴 ВКЛЮЧАЕМ DEBUG ЛОГИРОВАНИЕ
logging.basicConfig(
    level=logging.DEBUG,  # ← DEBUG уровень для максимальной видимости
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Убеждаемся что logger RAGPipeline на DEBUG
logger = logging.getLogger('rag_gigachat.core.rag_pipeline')
logger.setLevel(logging.DEBUG)

print("=" * 70)
print("🚀 DEBUG QUERY TEST - RAG Pipeline Verification")
print("=" * 70)

# Параметры
QUERY = "Что такое точка фокуса для самолёта и как её расчитать?"
K_RETRIEVE = 5
DATA_DIR = Path(__file__).parent / "data/domain_7_UAV/books"

print(f"\n📝 Query: {QUERY}")
print(f"🔍 K (documents to retrieve): {K_RETRIEVE}")
print(f"📁 Data directory: {DATA_DIR}")
print(f"💻 Device: {model_config.device}")
print(f"🧠 Embedding model: {model_config.embedding_model_name}")
print(f"🤖 LLM model: {model_config.llm_model_name}")
print("\n" + "=" * 70)

try:
    # 1. Инициализируем пайплайн
    print("\n[1/3] 🔧 Инициализирующ RAGPipeline...")
    pipeline = RAGPipeline(
        retrieval_type=RetrievalType.DENSE,
        embedding_type="huggingface",
        llm_type="local"
    )
    print("✅ RAGPipeline инициализирован")

    # 2. Загружаем документы
    print(f"\n[2/3] 📚 Загружаю PDF документы из {DATA_DIR}...")
    # Используем прямой метод с метаданными
    # force_reload=True чтобы пересоздать кэш (мертвый кэш был найден)
    pipeline.load_from_pdf_directory_with_metadata(DATA_DIR, recursive=False, force_reload=True)
    print(f"✅ Документы загружены. vector_store_initialized={pipeline.vector_store_initialized}")

    # 3. Обрабатываем запрос
    print(f"\n[3/3] ⚙️  Обработка запроса через RAG пайплайн...")
    print("-" * 70)
    result = pipeline.process_query(QUERY, k=K_RETRIEVE)
    print("-" * 70)

    # Выводим результаты
    print(f"\n{'='*70}")
    print("📊 РЕЗУЛЬТАТЫ ЗАПРОСА")
    print(f"{'='*70}")

    print(f"\n🤖 ОТВЕТ:\n{result.answer}\n")

    print(f"📚 НАЙДЕННЫЕ ДОКУМЕНТЫ ({len(result.retrieval_results.retrieved_docs)}):")
    for i, doc in enumerate(result.retrieval_results.retrieved_docs, 1):
        print(f"\n  [{i}] Источник: {doc['doc_id']}")
        print(f"      Score: {doc.get('score', 'N/A'):.3f}")
        print(f"      Preview: {doc['text'][:150]}...")

    print(f"\n⏱️  Время обработки: {result.generation_time:.2f} сек")
    print(f"🔢 Токенов в ответе: {result.tokens_generated}")
    print(f"\n{'='*70}")
    print("✅ Тест завершен успешно!")

except Exception as e:
    print(f"\n❌ ОШИБКА: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
