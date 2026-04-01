"""
create_wikieval_dataset.py - Финальная версия для создания PDF и testset.json
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Union
import logging
from datasets import load_dataset

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    import textwrap
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("❌ ReportLab не установлен! Установите: pip install reportlab")
    exit(1)

logger = logging.getLogger(__name__)

class WikiEvalDatasetCreator:
    """Создание PDF документов и testset.json"""
    
    def __init__(self, output_dir: Path = Path("data/domain_3_WikiEval_simple")):
        self.output_dir = Path(output_dir)
        self.pdf_dir = self.output_dir / "books"
        self.testset_path = self.output_dir / "testset.json"
        
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PDF директория: {self.pdf_dir}")
    
    def sanitize_filename(self, text: str) -> str:
        """Очистка имени файла"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            text = text.replace(char, '_')
        text = text.replace(' ', '_')
        text = text.replace('-', '_')
        text = text.replace('–', '_')
        text = text.replace('(', '_')
        text = text.replace(')', '_')
        text = text.replace(',', '_')
        if len(text) > 100:
            text = text[:100]
        return text
    
    def convert_to_text(self, content: Union[str, List[str]]) -> str:
        """Конвертация в строку"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return ' '.join(str(item) for item in content)
        return str(content) if content else ""
    
    def create_pdf(self, text_v1: str, text_v2: str, output_path: Path, title: str = None) -> bool:
        """Создание PDF с двумя страницами"""
        try:
            doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                                   rightMargin=72, leftMargin=72,
                                   topMargin=72, bottomMargin=72)
            
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                        fontSize=14, fontName='Helvetica-Bold',
                                        spaceAfter=20, alignment=0)
            subtitle_style = ParagraphStyle('CustomSubtitle', parent=styles['Heading2'],
                                          fontSize=12, fontName='Helvetica-Bold',
                                          spaceAfter=15, alignment=0)
            normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'],
                                        fontSize=10, fontName='Helvetica',
                                        leading=14, spaceAfter=10)
            
            story = []
            
            # Страница 1
            if title:
                story.append(Paragraph(f"<b>{title}</b>", title_style))
                story.append(Spacer(1, 10))
            story.append(Paragraph("<b>CONTEXT VERSION 1</b>", subtitle_style))
            story.append(Spacer(1, 10))
            
            for para in text_v1.split('\n\n'):
                if para.strip():
                    wrapped = '\n'.join(textwrap.wrap(para, width=80))
                    story.append(Paragraph(wrapped.replace('\n', '<br/>'), normal_style))
                    story.append(Spacer(1, 6))
            
            story.append(PageBreak())
            
            # Страница 2
            if title:
                story.append(Paragraph(f"<b>{title}</b>", title_style))
                story.append(Spacer(1, 10))
            story.append(Paragraph("<b>CONTEXT VERSION 2</b>", subtitle_style))
            story.append(Spacer(1, 10))
            
            for para in text_v2.split('\n\n'):
                if para.strip():
                    wrapped = '\n'.join(textwrap.wrap(para, width=80))
                    story.append(Paragraph(wrapped.replace('\n', '<br/>'), normal_style))
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            return True
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return False
    
    def create_dataset(self, max_samples: int = None):
        """Создание датасета"""
        logger.info("Загрузка explodinggradients/WikiEval...")
        dataset = load_dataset("explodinggradients/WikiEval")['train']
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        logger.info(f"Загружено {len(dataset)} примеров")
        
        # Формат словаря для testset.json
        test_samples = {}
        stats = {'total': len(dataset), 'created': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Создание PDF документов ({len(dataset)} примеров)")
        logger.info(f"{'='*60}\n")
        
        for idx, item in enumerate(dataset):
            try:
                # Извлекаем данные
                qid = f"q{idx:03d}"
                question = self.convert_to_text(item.get('question', ''))
                context_v1 = self.convert_to_text(item.get('context_v1', ''))
                context_v2 = self.convert_to_text(item.get('context_v2', ''))
                source = self.convert_to_text(item.get('source', f'doc_{idx}'))
                answer = self.convert_to_text(item.get('answer', ''))
                
                # Проверяем наличие вопроса
                if not question:
                    logger.warning(f"[{idx+1}] Пропуск: пустой вопрос")
                    stats['skipped'] += 1
                    continue
                
                # Проверяем наличие контекстов
                if not context_v1 and not context_v2:
                    logger.warning(f"[{idx+1}] Пропуск: пустые контексты")
                    stats['skipped'] += 1
                    continue
                
                # Заполняем пустые контексты
                if not context_v1:
                    context_v1 = "No context available for version 1"
                if not context_v2:
                    context_v2 = "No context available for version 2"
                
                # Создаем PDF
                pdf_filename = f"{self.sanitize_filename(source)}.pdf"
                pdf_path = self.pdf_dir / pdf_filename
                
                success = self.create_pdf(context_v1, context_v2, pdf_path, source)
                
                if success:
                    stats['created'] += 1
                    
                    # Добавляем в словарь test_samples (формат для data_loader)
                    test_samples[qid] = {
                        "query": question,
                        "relevant_docs": [pdf_filename],
                        "reference_answer": answer if answer else ""
                    }
                    
                    logger.info(f"[{idx+1}/{len(dataset)}] ✓ PDF: {pdf_filename}")
                    logger.info(f"    Вопрос: {question[:80]}...")
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                logger.error(f"[{idx+1}] Ошибка: {e}")
                stats['failed'] += 1
        
        # Сохраняем testset.json
        with open(self.testset_path, 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Тестовый набор сохранен: {self.testset_path}")
        logger.info(f"{'='*60}")
        logger.info(f"  Количество примеров: {len(test_samples)}")
        logger.info(f"  Формат: словарь с ключами q000, q001, ...")
        
        # Показываем пример
        if test_samples:
            first_key = list(test_samples.keys())[0]
            logger.info(f"\n📝 Пример формата:")
            logger.info(json.dumps({first_key: test_samples[first_key]}, indent=2, ensure_ascii=False)[:500])
        
        return stats, test_samples


def main(args_list=None):
    import argparse
    parser = argparse.ArgumentParser(description="Создание PDF и testset.json из WikiEval")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Максимальное количество примеров")
    parser.add_argument("--output_dir", type=str, default="data/domain_3_WikiEval_1row",
                       help="Директория для сохранения")
    
    # Если передан список аргументов, используем его, иначе берем из sys.argv
    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()    
    
    creator = WikiEvalDatasetCreator(output_dir=Path(args.output_dir))
    stats, test_samples = creator.create_dataset(max_samples=args.max_samples)
    
    print(f"\n{'='*60}")
    print(f"СОЗДАНИЕ ДАТАСЕТА - СВОДКА")
    print(f"{'='*60}")
    print(f"\n📊 Статистика:")
    print(f"  Всего примеров: {stats['total']}")
    print(f"  Создано PDF: {stats['created']}")
    print(f"  Ошибок: {stats['failed']}")
    print(f"  Пропущено: {stats['skipped']}")
    print(f"  Тестовых запросов: {len(test_samples)}")
    
    if test_samples:
        print(f"\n✅ Готово! Запустите эксперимент:")
        print(f"   python experiment.py \\")
        print(f"     --pdf_dir {args.output_dir}/books \\")
        print(f"     --testset {args.output_dir}/testset.json \\")
        print(f"     --name wikieval_experiment")
    else:
        print(f"\n❌ Не удалось создать тестовые примеры!")


if __name__ == "__main__":
    #main(["--max_samples", "1", "--output_dir", "data/domain_3_WikiEval_1row"])
    main(["--max_samples", "2", "--output_dir", "data/domain_4_WikiEval_2row"])
    
    # Или если хотите запускать без аргументов:
    # main()
#    & d:/study/2026-03_RAG/.venv/Scripts/python.exe d:/study/2026-03_RAG/create_wikieval_dataset.py --max_samples 1 --output_dir data/domain_3_WikiEval_1row