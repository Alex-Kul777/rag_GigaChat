#!/usr/bin/env python
"""Диагностика качества текста в PDF файлах"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Any

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from rag_gigachat.utils.text_utils import analyze_text_quality


def diagnose_pdf_set(pdf_dir: Path, max_pdfs: int = 5) -> Dict[str, Any]:
    """Диагностировать набор PDF файлов"""
    from rag_gigachat.data.data_loader import DocumentLoader

    loader = DocumentLoader()

    stats = {
        'timestamp': str(Path.cwd()),
        'directory': str(pdf_dir),
        'pdfs_analyzed': 0,
        'total_documents': 0,
        'total_chars': 0,
        'avg_doc_length': 0,
        'aggregated_issues': {
            'multiple_spaces': 0,
            'multiple_newlines': 0,
            'tabs': 0,
            'no_break_spaces': 0,
            'broken_words': 0,
        },
        'total_waste_chars': 0,
        'documents': []
    }

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"❌ PDF файлы не найдены в {pdf_dir}")
        return stats

    print(f"\n📁 Найдено PDF файлов: {len(pdf_files)}")
    print(f"📊 Анализирую первые {min(max_pdfs, len(pdf_files))} файлов...\n")

    for pdf_file in pdf_files[:max_pdfs]:
        try:
            logger.info(f"🔍 Обработка: {pdf_file.name}")
            docs = loader.load_pdf_with_metadata(pdf_file)

            for i, doc in enumerate(docs):
                text = doc.page_content
                analysis = analyze_text_quality(text)

                stats['total_documents'] += 1
                stats['total_chars'] += analysis['char_count']
                stats['total_waste_chars'] += analysis['total_issues']

                # Агрегируем проблемы
                for key in stats['aggregated_issues']:
                    stats['aggregated_issues'][key] += analysis['issues'].get(key, 0)

                stats['documents'].append({
                    'pdf': pdf_file.name,
                    'page': i + 1,
                    'char_count': analysis['char_count'],
                    'word_count': analysis['word_count'],
                    'issues': analysis['issues'],
                    'waste_percent': analysis['waste_percent'],
                })

                stats['pdfs_analyzed'] += 1

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {pdf_file.name}: {e}")

    # Итоги
    if stats['total_documents'] > 0:
        stats['avg_doc_length'] = stats['total_chars'] // stats['total_documents']
        stats['avg_waste_percent'] = round((stats['total_waste_chars'] / stats['total_chars']) * 100, 2)

    return stats


def print_diagnostics(stats: Dict[str, Any]):
    """Красивый вывод результатов диагностики"""
    print("\n" + "="*80)
    print("🔍 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ КАЧЕСТВА ТЕКСТА")
    print("="*80)

    print(f"\n📊 СТАТИСТИКА:")
    print(f"  Директория: {stats['directory']}")
    print(f"  PDF файлов обработано: {stats['pdfs_analyzed']}")
    print(f"  Всего документов: {stats['total_documents']}")
    print(f"  Всего символов: {stats['total_chars']:,}")
    print(f"  Средний размер документа: {stats['avg_doc_length']:,} символов")

    print(f"\n❌ АГРЕГИРОВАННЫЕ ПРОБЛЕМЫ:")
    for issue, count in stats['aggregated_issues'].items():
        if count > 0:
            print(f"  • {issue}: {count}")

    print(f"\n📉 ПОТЕРИ ДАННЫХ:")
    print(f"  Всего символов на артефакты: {stats['total_waste_chars']:,}")
    print(f"  Средний процент потерь: {stats.get('avg_waste_percent', 0):.2f}%")

    if stats['documents']:
        print(f"\n📋 ПРИМЕРЫ ПРОБЛЕМ (первые 5):")
        for doc in stats['documents'][:5]:
            print(f"\n  {doc['pdf']} (страница {doc['page']}):")
            print(f"    Размер: {doc['char_count']:,} символов")
            print(f"    Потери: {doc['waste_percent']:.2f}%")
            for issue, count in doc['issues'].items():
                if count > 0:
                    print(f"      • {issue}: {count}")

    print("\n" + "="*80)


def save_diagnostics(stats: Dict[str, Any], output_file: Path):
    """Сохранить результаты в JSON"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Результаты сохранены в {output_file}")


if __name__ == "__main__":
    # Диагностируем папку data/corpus
    pdf_dir = Path("data/corpus")

    if not pdf_dir.exists():
        logger.warning(f"❌ Директория {pdf_dir} не существует. Пропускаю диагностику.")
        sys.exit(0)

    # Запускаем диагностику
    stats = diagnose_pdf_set(pdf_dir, max_pdfs=5)

    # Выводим результаты
    print_diagnostics(stats)

    # Сохраняем в JSON
    output_file = Path("DIAGNOSTICS_RESULTS.json")
    save_diagnostics(stats, output_file)

    logger.info(f"✅ Диагностика завершена!")
