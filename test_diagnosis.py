#!/usr/bin/env python3
"""Диагностический скрипт для проверки сохранения имён файлов"""
import sys
import logging
from pathlib import Path

# Настройка логирования на DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_gigachat.config import data_config
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.data.data_loader import CorpusLoader

print("\n" + "="*80)
print("🔍 ДИАГНОСТИЧЕСКИЙ СКРИПТ - Проверка сохранения имён файлов")
print("="*80 + "\n")

# Создаём pipeline
print("📦 Инициализация RAG Pipeline...")
pipeline = RAGPipeline()

# Загружаем документы
pdf_dir = Path(data_config.documents_dirs.get("debug", "data/debug"))
print(f"📁 Загружаем документы из: {pdf_dir}")
print(f"   Существует директория? {pdf_dir.exists()}")

if pdf_dir.exists():
    files = list(pdf_dir.glob("*.pdf"))
    print(f"   Найдено PDF файлов: {len(files)}")
    for f in files[:5]:
        print(f"      - {f.name}")

print("\n🚀 Загрузка в RAG Pipeline...")
try:
    pipeline.load_from_pdf_directory_with_metadata(pdf_dir)
    print("✅ Документы загружены успешно")
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Выполняем поиск
print("\n" + "="*80)
print("🔎 ВЫПОЛНЕНИЕ ПОИСКА")
print("="*80)

query = "что такое RAG"
print(f"📝 Запрос: '{query}'")
print()

try:
    result = pipeline.process_query(query, k=3)
    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТЫ ПОИСКА")
    print("="*80)
    if result.retrieval_results:
        print(f"✅ Найдено документов: {len(result.retrieval_results.retrieved_docs)}")
        for i, doc in enumerate(result.retrieval_results.retrieved_docs, 1):
            print(f"\n📄 Документ {i}:")
            print(f"   - doc_id: {doc.get('doc_id', 'NOT FOUND')}")
            print(f"   - source_file: {doc.get('source_file', 'NOT FOUND')}")
            print(f"   - page: {doc.get('page', 'NOT FOUND')}")
            print(f"   - score: {doc.get('score', 'NOT FOUND')}")
            print(f"   - text preview: {doc.get('text', '')[:100]}...")
    else:
        print("❌ Документы не найдены")
except Exception as e:
    print(f"❌ Ошибка поиска: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
