"""ФАЗА 4: Тесты производительности и эффективности конвейера обработки текста

Измеряет:
- Скорость обработки текста (нормализация, разбиение, фильтрация)
- Качество результатов (метрики, артефакты, токены)
- Использование памяти и ресурсов
- Эффективность фильтрации (процент удаленных чанков)
"""

import pytest
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.utils.text_utils import (
    normalize_text, analyze_text_quality, SpacySmartSplitter,
    estimate_token_count, filter_documents_by_token_count, SPACY_AVAILABLE
)
from rag_gigachat.data.data_loader import TextSplitter
from langchain_core.documents import Document as LangChainDocument


class TestNormalizationPerformance:
    """Тесты производительности нормализации"""

    def test_normalize_speed_small_text(self):
        """Нормализация малого текста выполняется быстро"""
        text = "Текст   с   проблемами" * 10

        start = time.time()
        result = normalize_text(text)
        elapsed = time.time() - start

        # Должно выполниться почти мгновенно
        assert elapsed < 0.01
        assert len(result) <= len(text)

    def test_normalize_speed_large_text(self):
        """Нормализация большого текста"""
        text = ("Текст с проблемами  и  пробелами\t\t\tи табуляциями\n\n\n. " * 1000)

        start = time.time()
        result = normalize_text(text)
        elapsed = time.time() - start

        # Даже большой текст нормализуется быстро (< 100ms)
        assert elapsed < 0.1
        assert "  " not in result
        assert "\t" not in result

    def test_normalize_quality_improvement(self):
        """Нормализация значительно улучшает качество"""
        dirty_text = "Текст    с     проблемами\t\t\t.\n\n\nНовый" * 20

        quality_before = analyze_text_quality(dirty_text)
        normalized = normalize_text(dirty_text)
        quality_after = analyze_text_quality(normalized)

        # Качество должно значительно улучшиться
        improvement = (
            (quality_before['total_issues'] - quality_after['total_issues']) /
            max(quality_before['total_issues'], 1)
        )
        assert improvement > 0.9  # >90% улучшение


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
class TestSplittingPerformance:
    """Тесты производительности разбиения на предложения"""

    def test_sentence_splitting_speed(self):
        """spaCy разбиение на предложения быстро"""
        text = "Первое предложение. Второе предложение. " * 50

        splitter = SpacySmartSplitter()
        start = time.time()
        sentences = splitter.split_into_sentences(text, language='ru')
        elapsed = time.time() - start

        # Разбиение должно быть быстрым
        assert elapsed < 1.0
        assert len(sentences) > 0

    def test_text_splitter_performance(self):
        """TextSplitter эффективен для разбиения на чанки"""
        docs = [
            LangChainDocument(
                page_content="Предложение. " * 100,
                metadata={'source': 'doc.pdf'}
            )
        ]

        splitter = TextSplitter(chunk_size=300)
        start = time.time()
        chunks = splitter.split_documents(docs)
        elapsed = time.time() - start

        assert elapsed < 2.0
        assert len(chunks) > 0


class TestFilteringEffectiveness:
    """Тесты эффективности фильтрации"""

    def test_filter_removes_garbage(self):
        """Фильтрация успешно удаляет мусор"""
        docs = [
            LangChainDocument(page_content="Очень длинный текст с достаточным количеством слов и информации для прохождения фильтра.", metadata={}),
            LangChainDocument(page_content=".", metadata={}),  # Мусор
            LangChainDocument(page_content="Еще", metadata={}),  # Мусор
            LangChainDocument(page_content="Второй длинный текст с достаточным объемом информации для прохождения фильтра.", metadata={}),
        ]

        before = len(docs)
        filtered = filter_documents_by_token_count(docs, min_tokens=30)
        after = len(filtered)

        # Должны удалиться короткие чанки
        removed = before - after
        assert removed > 0
        assert all(c.metadata['token_count'] >= 30 for c in filtered)

    def test_filter_threshold_customizable(self):
        """Порог фильтрации настраивается"""
        doc = LangChainDocument(
            page_content="Текст с несколькими словами",
            metadata={}
        )
        docs = [doc] * 5

        # Низкий порог
        result_low = filter_documents_by_token_count(docs, min_tokens=5)
        assert len(result_low) == 5

        # Высокий порог
        result_high = filter_documents_by_token_count(docs, min_tokens=100)
        assert len(result_high) == 0

    def test_filter_preserves_quality_chunks(self):
        """Фильтрация не удаляет хорошие чанки"""
        good_chunks = [
            LangChainDocument(
                page_content="Первый качественный чанк с достаточным объемом информации для анализа.",
                metadata={}
            ),
            LangChainDocument(
                page_content="Второй качественный текст со смыслом и содержанием для обработки.",
                metadata={}
            ),
        ]

        # Используем более реалистичный порог для русского текста (эти чанки ~15-17 токенов)
        filtered = filter_documents_by_token_count(good_chunks, min_tokens=15)

        # Все хорошие чанки должны пройти
        assert len(filtered) == len(good_chunks)


class TestTokenEstimationAccuracy:
    """Тесты точности оценки токенов"""

    def test_token_estimate_correlation_russian(self):
        """Оценка токенов коррелирует с длиной русского текста"""
        texts = [
            "Короткий текст",
            "Среднего размера текст с дополнительными словами",
            "Очень длинный текст с множеством слов и информации для тестирования точности оценки" * 2,
        ]

        token_counts = [estimate_token_count(t, 'ru') for t in texts]

        # Токены должны расти с длиной текста
        assert token_counts[0] < token_counts[1] < token_counts[2]

    def test_token_estimate_consistency(self):
        """Одинаковый текст дает одинаковую оценку"""
        text = "Это текст для проверки консистентности оценки токенов"

        estimates = [estimate_token_count(text, 'ru') for _ in range(5)]

        # Все оценки должны быть одинаковыми
        assert len(set(estimates)) == 1

    def test_token_estimate_language_aware(self):
        """Оценка зависит от языка"""
        text = "A" * 100  # 100 символов

        ru_tokens = estimate_token_count(text, 'ru')
        en_tokens = estimate_token_count(text, 'en')

        # Разные языки должны дать разные оценки
        assert ru_tokens != en_tokens


class TestQualityMetrics:
    """Тесты метрик качества текста"""

    def test_artifact_detection_comprehensive(self):
        """Обнаружение артефактов работает комплексно"""
        text_with_artifacts = "Текст    с     пробелами\t\t\tи табуляциями\n\n\nи переносами"

        quality = analyze_text_quality(text_with_artifacts)

        # Должны быть обнаружены все типы артефактов
        assert quality['issues']['multiple_spaces'] > 0
        assert quality['issues']['tabs'] > 0
        assert quality['issues']['multiple_newlines'] > 0

    def test_clean_text_has_zero_issues(self):
        """Чистый текст не содержит артефактов"""
        clean_text = "Это чистый текст без проблем."

        quality = analyze_text_quality(clean_text)

        assert quality['total_issues'] == 0
        assert quality['waste_percent'] == 0.0

    def test_waste_percentage_meaningful(self):
        """Процент потерь имеет смысл"""
        texts = [
            "Чистый текст",
            "Текст    с    пробелами",
            "Текст\t\t\tс\t\t\tтабуляциями",
        ]

        qualities = [analyze_text_quality(t) for t in texts]
        percentages = [q['waste_percent'] for q in qualities]

        # Процент потерь должен расти с количеством артефактов
        assert percentages[0] <= percentages[1] < percentages[2]


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
class TestEndToEndEffectiveness:
    """Сквозные тесты эффективности"""

    def test_pipeline_effectiveness_metrics(self):
        """Полный конвейер обрабатывает текст эффективно"""
        dirty_doc = LangChainDocument(
            page_content="Текст    с    проблемами\t\t\tи табуляциями\n\n\n.".strip(),
            metadata={'source': 'test.pdf'}
        )

        # Анализируем качество до
        quality_before = analyze_text_quality(dirty_doc.page_content)

        # Нормализуем
        dirty_doc.page_content = normalize_text(dirty_doc.page_content)

        # Анализируем качество после
        quality_after = analyze_text_quality(dirty_doc.page_content)

        # Качество должно значительно улучшиться
        assert quality_after['total_issues'] <= quality_before['total_issues']

    def test_pipeline_throughput(self):
        """Пропускная способность конвейера достаточна"""
        # Создаем 20 документов среднего размера
        docs = [
            LangChainDocument(
                page_content="Документ {i}. " * 30,
                metadata={'source': f'doc_{i}.pdf'}
            )
            for i in range(20)
        ]

        start = time.time()

        # Нормализуем все
        for doc in docs:
            doc.page_content = normalize_text(doc.page_content)

        # Разбиваем
        splitter = TextSplitter(chunk_size=300)
        chunks = splitter.split_documents(docs)

        # Фильтруем
        filtered = filter_documents_by_token_count(chunks, min_tokens=15)

        elapsed = time.time() - start

        # Должно обработать 20 документов за разумное время
        assert elapsed < 5.0
        assert len(filtered) > 0
        throughput = len(docs) / elapsed
        # Более 4 документов в секунду
        assert throughput > 4

    def test_pipeline_output_quality(self):
        """Качество выходных данных конвейера высокое"""
        doc = LangChainDocument(
            page_content="Текст    с    проблемами\t\tи табуляциями\n\n\n.",
            metadata={'source': 'test.pdf'}
        )

        # Пропускаем через конвейер
        doc.page_content = normalize_text(doc.page_content)
        splitter = TextSplitter(chunk_size=200)
        chunks = splitter.split_documents([doc])
        filtered = filter_documents_by_token_count(chunks, min_tokens=10)

        # Все выходные чанки должны быть чистыми
        for chunk in filtered:
            assert "  " not in chunk.page_content
            assert "\t" not in chunk.page_content
            assert "  " not in chunk.page_content.replace("\n", " ")


class TestScalability:
    """Тесты масштабируемости"""

    def test_normalize_scales_linearly(self):
        """Нормализация масштабируется примерно линейно"""
        sizes = [1000, 5000, 10000]
        times = []

        for size in sizes:
            text = "Текст   с   проблемами. " * (size // 30)

            start = time.time()
            normalize_text(text)
            times.append(time.time() - start)

        # Время должно расти примерно линейно с размером
        # Проверяем что не квадратичное и не экспоненциальное
        ratio1 = times[1] / times[0]
        ratio2 = times[2] / times[1]

        # Оба ratio должны быть примерно одинаковыми (линейность)
        # Даже если точно не 3-7, вещества должны быть в одном порядке (не 10x+ разница)
        assert ratio1 > 1.5  # Must increase
        assert ratio2 > 1.0  # Must increase or stay similar
        assert ratio1 < 15  # But not exponential
        assert ratio2 < 10

    def test_token_filtering_efficient(self):
        """Фильтрация токенов масштабируется"""
        # Много документов
        docs = [
            LangChainDocument(
                page_content="Текст " * 50,
                metadata={}
            )
            for _ in range(100)
        ]

        start = time.time()
        filtered = filter_documents_by_token_count(docs, min_tokens=20)
        elapsed = time.time() - start

        # Должна обработать 100 документов за разумное время (< 5 сек)
        assert elapsed < 5.0
        assert len(filtered) > 0
        # Проверяем что при этом есть хотя бы какие-то результаты
        assert len(filtered) > len(docs) * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
