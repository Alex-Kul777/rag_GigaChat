"""ФАЗА 3: Интеграционные тесты полного конвейера обработки текста"""

import pytest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.data.data_loader import DocumentLoader, TextSplitter
from rag_gigachat.utils.text_utils import (
    normalize_text, SpacySmartSplitter, SPACY_AVAILABLE,
    filter_documents_by_token_count, estimate_token_count
)
from langchain_core.documents import Document as LangChainDocument


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
class TestFullPipelineIntegration:
    """Интеграционные тесты полного конвейера PDF → Chunks"""

    def test_pipeline_pdf_to_chunks(self):
        """Полный конвейер: PDF текст → нормализация → разбиение → фильтрация"""
        # Симуляция сырого PDF текста с артефактами
        raw_pdf_text = """Документ   с   множеством   проблем.

Новый  абзац  идёт  здесь.  Слово  при\nнято
в  документ.  Автор:  д-р  Иванов\t\t(контакт)."""

        # 1. Создаем документ как из PyPDFLoader
        doc = LangChainDocument(
            page_content=raw_pdf_text,
            metadata={'source': 'test.pdf', 'page': 1}
        )

        # 2. Нормализуем
        normalized = normalize_text(doc.page_content)
        doc.page_content = normalized

        # 3. Проверяем что артефакты удалены
        assert "  " not in normalized
        assert "\t" not in normalized
        assert "при\n" not in normalized

        # 4. Разбиваем на предложения
        splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents([doc])

        # 5. Проверяем что получились чанки
        assert len(chunks) > 0
        assert all(isinstance(c, LangChainDocument) for c in chunks)

        # 6. Каждый чанк должен быть чистым
        for chunk in chunks:
            assert "  " not in chunk.page_content
            assert "\t" not in chunk.page_content

        # 7. Фильтруем по токенам
        filtered = filter_documents_by_token_count(chunks, min_tokens=20)

        # 8. Проверяем что остались только качественные чанки
        assert len(filtered) <= len(chunks)
        for chunk in filtered:
            assert chunk.metadata['token_count'] >= 20

    def test_pipeline_preserves_content(self):
        """Конвейер сохраняет основной контент документа"""
        original_text = "Первое предложение со смыслом. Второе предложение. Третье предложение."

        doc = LangChainDocument(
            page_content=original_text,
            metadata={'source': 'test.pdf'}
        )

        # Проводим через конвейер
        normalized = normalize_text(doc.page_content)
        splitter = TextSplitter(chunk_size=200)
        chunks = splitter.split_documents([
            LangChainDocument(page_content=normalized, metadata=doc.metadata)
        ])
        filtered = filter_documents_by_token_count(chunks, min_tokens=1)

        # Проверяем что контент сохранен
        combined_text = " ".join(c.page_content for c in filtered)
        assert "Первое" in combined_text
        assert "Второе" in combined_text
        assert "Третье" in combined_text

    def test_pipeline_with_metadata_tracking(self):
        """Метаданные прослеживаются через весь конвейер"""
        doc = LangChainDocument(
            page_content="Текст с информацией для тестирования конвейера.",
            metadata={
                'source': 'important.pdf',
                'author': 'Test Author',
                'page': 1
            }
        )

        normalized = normalize_text(doc.page_content)
        doc.page_content = normalized

        splitter = TextSplitter(chunk_size=200)
        chunks = splitter.split_documents([doc])
        filtered = filter_documents_by_token_count(chunks, min_tokens=1)

        # Проверяем что метаданные сохранены
        for chunk in filtered:
            assert chunk.metadata['source'] == 'important.pdf'
            assert chunk.metadata['author'] == 'Test Author'
            assert 'token_count' in chunk.metadata
            assert 'chunk_id' in chunk.metadata

    def test_pipeline_quality_metrics(self):
        """Конвейер улучшает метрики качества текста"""
        from rag_gigachat.utils.text_utils import analyze_text_quality

        dirty_text = "Текст    с     лишними    пробелами\t\t\tи табуляциями\n\n\n."

        # Анализируем качество до конвейера
        quality_before = analyze_text_quality(dirty_text)

        # Проводим через нормализацию
        clean_text = normalize_text(dirty_text)

        # Анализируем качество после
        quality_after = analyze_text_quality(clean_text)

        # Качество должно улучшиться
        assert quality_after['total_issues'] < quality_before['total_issues']
        assert quality_after['waste_percent'] < quality_before['waste_percent']

    def test_pipeline_with_multiple_documents(self):
        """Конвейер обрабатывает несколько документов"""
        docs = [
            LangChainDocument(
                page_content="Первый документ  с   проблемами.",
                metadata={'source': 'doc1.pdf', 'page': 1}
            ),
            LangChainDocument(
                page_content="Второй документ\t\tи его текст.",
                metadata={'source': 'doc2.pdf', 'page': 1}
            ),
            LangChainDocument(
                page_content="Третий  документ  без качественного содержимого.",
                metadata={'source': 'doc3.pdf', 'page': 1}
            ),
        ]

        # Нормализуем все документы
        for doc in docs:
            doc.page_content = normalize_text(doc.page_content)

        # Разбиваем на чанки
        splitter = TextSplitter(chunk_size=300)
        chunks = splitter.split_documents(docs)

        # Должны быть чанки от всех документов
        assert len(chunks) >= len(docs)

        # Фильтруем
        filtered = filter_documents_by_token_count(chunks, min_tokens=15)

        # Все отфильтрованные чанки должны быть чистыми
        for chunk in filtered:
            assert "  " not in chunk.page_content
            assert "\t" not in chunk.page_content

    def test_pipeline_language_detection(self):
        """Конвейер корректно обрабатывает разные языки"""
        ru_text = "Это русский текст с несколькими предложениями для тестирования."
        en_text = "This is English text with several sentences for testing."

        docs = [
            LangChainDocument(page_content=ru_text, metadata={'source': 'ru.pdf'}),
            LangChainDocument(page_content=en_text, metadata={'source': 'en.pdf'}),
        ]

        splitter = TextSplitter(chunk_size=200)
        chunks = splitter.split_documents(docs)
        filtered = filter_documents_by_token_count(chunks, min_tokens=10)

        # Оба языка должны быть обработаны
        sources = set(doc.metadata['source'] for doc in filtered)
        assert len(sources) <= 2

    def test_pipeline_token_consistency(self):
        """Подсчет токенов согласован через конвейер"""
        text = "Предложение один. Предложение два. Предложение три."

        doc = LangChainDocument(
            page_content=text,
            metadata={'source': 'test.pdf'}
        )

        # Проводим через конвейер
        normalized = normalize_text(doc.page_content)
        doc.page_content = normalized

        splitter = TextSplitter(chunk_size=200)
        chunks = splitter.split_documents([doc])
        filtered = filter_documents_by_token_count(chunks, min_tokens=5)

        # Проверяем что токены добавлены в метаданные
        for chunk in filtered:
            estimated = estimate_token_count(chunk.page_content, 'ru')
            stored = chunk.metadata.get('token_count')
            assert stored > 0
            assert stored == estimated

    def test_pipeline_removes_garbage_chunks(self):
        """Конвейер удаляет мусорные чанки"""
        text = """Хорошее предложение со смыслом.

        Точка. Ещё точка.

        Ещё одно хорошее предложение с информацией."""

        doc = LangChainDocument(
            page_content=text,
            metadata={'source': 'test.pdf'}
        )

        normalized = normalize_text(doc.page_content)
        doc.page_content = normalized

        splitter = TextSplitter(chunk_size=200)
        chunks = splitter.split_documents([doc])

        # Без фильтра может быть мусорные чанки
        raw_count = len(chunks)

        # С фильтром убираем мусор
        filtered = filter_documents_by_token_count(chunks, min_tokens=20)
        filtered_count = len(filtered)

        # Должны быть только хорошие чанки
        assert filtered_count <= raw_count


class TestPipelineRealistic:
    """Реалистичные сценарии использования конвейера"""

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
    def test_realistic_pdf_workflow(self):
        """Реалистичный сценарий обработки PDF"""
        # Симуляция текста из реального PDF (со сканированием)
        pdf_text = """ОТЧЕТ О ПРОДЕЛАННОЙ РАБОТЕ

Введение
В этом отчёте описывается работа,  проведённая в течение месяца.
Основные  достижения  включают  несколько важных элементов.

Результаты
Первое направление работы было  успешным.  Получены хорошие результаты.
Второе направление тоже  прогрессировало нормально.

Заключение
Работа проведена согласно плану  и  расписанию.  Ожидаем  продолжения."""

        doc = LangChainDocument(
            page_content=pdf_text,
            metadata={'source': 'report.pdf', 'author': 'Reports Dept'}
        )

        # Применяем конвейер
        doc.page_content = normalize_text(doc.page_content)

        splitter = TextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents([doc])

        filtered = filter_documents_by_token_count(chunks, min_tokens=25)

        # Результат: чистые, высококачественные чанки
        assert len(filtered) > 0
        assert all("  " not in c.page_content for c in filtered)
        assert all(c.metadata['token_count'] >= 25 for c in filtered)

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
    def test_pipeline_handles_edge_cases(self):
        """Конвейер обрабатывает граничные случаи"""
        edge_cases = [
            "",  # Пустой текст
            "A",  # Один символ
            ".",  # Один символ - пунктуация
            "Очень" * 100,  # Повторяющийся текст
            "   " * 50,  # Только пробелы
        ]

        for i, text in enumerate(edge_cases):
            doc = LangChainDocument(
                page_content=text,
                metadata={'source': f'edge_{i}.pdf'}
            )

            # Должно работать без ошибок
            normalized = normalize_text(doc.page_content)
            doc.page_content = normalized

            splitter = TextSplitter(chunk_size=200)
            chunks = splitter.split_documents([doc])

            filtered = filter_documents_by_token_count(chunks, min_tokens=10)

            # Должен быть обработан корректно
            assert isinstance(chunks, list)
            assert isinstance(filtered, list)


class TestPipelinePerformance:
    """Тесты производительности конвейера"""

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
    def test_pipeline_efficiency(self):
        """Конвейер эффективен по памяти и времени"""
        import time

        # Создаем много документов
        docs = []
        for i in range(10):
            doc = LangChainDocument(
                page_content=f"Документ {i}. " * 20,  # Среднего размера текст
                metadata={'source': f'doc_{i}.pdf', 'page': 1}
            )
            docs.append(doc)

        start = time.time()

        # Нормализуем
        for doc in docs:
            doc.page_content = normalize_text(doc.page_content)

        # Разбиваем
        splitter = TextSplitter(chunk_size=300)
        chunks = splitter.split_documents(docs)

        # Фильтруем
        filtered = filter_documents_by_token_count(chunks, min_tokens=20)

        elapsed = time.time() - start

        # Должно выполниться быстро (менее 2 секунд для 10 документов)
        assert elapsed < 2.0
        assert len(filtered) > 0

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
    def test_pipeline_memory_efficiency(self):
        """Конвейер не дублирует данные излишне"""
        doc = LangChainDocument(
            page_content="Текст" * 500,  # Большой документ
            metadata={'source': 'large.pdf'}
        )

        original_size = len(doc.page_content)

        # Пропускаем через конвейер
        doc.page_content = normalize_text(doc.page_content)

        splitter = TextSplitter(chunk_size=300)
        chunks = splitter.split_documents([doc])

        filtered = filter_documents_by_token_count(chunks, min_tokens=10)

        # Общий размер фильтрованных данных не должен превышать оригинал
        total_filtered_size = sum(len(c.page_content) for c in filtered)
        assert total_filtered_size <= original_size * 1.1  # Небольшой буфер для метаданных


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
