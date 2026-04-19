"""RAG Pipeline Integration - Step 1: Token Filtering in PDF Loading

Tests that token filtering is applied correctly in load_from_pdf_directory_with_metadata()
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.utils.text_utils import SPACY_AVAILABLE
from langchain_core.documents import Document


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy не установлена")
class TestRAGPipelineTokenFiltering:
    """Тесты интеграции фильтрации токенов в RAG пайплайне"""

    def test_rag_pipeline_imports_token_filtering(self):
        """RAGPipeline импортирует функции фильтрации"""
        from rag_gigachat.core.rag_pipeline import filter_documents_by_token_count
        assert callable(filter_documents_by_token_count)

    def test_rag_pipeline_initialization(self):
        """RAGPipeline инициализируется с новой функциональностью"""
        pipeline = RAGPipeline(chunk_size=500, chunk_overlap=50)
        assert pipeline is not None
        assert pipeline.chunk_size == 500

    def test_token_filtering_in_load_documents(self):
        """Фильтрация токенов применяется при загрузке документов"""
        pipeline = RAGPipeline(chunk_size=300, chunk_overlap=50)

        # Мокируем corpus_loader для возврата контролируемых документов
        mock_documents = {
            'doc_1': {
                'text': 'Длинный документ с достаточным количеством слов и информации для прохождения фильтра.',
                'metadata': {'source': 'test.pdf', 'page': 1}
            },
            'doc_2': {
                'text': 'Очень короткий',  # Будет отфильтрован
                'metadata': {'source': 'test.pdf', 'page': 1}
            },
            'doc_3': {
                'text': 'Второй документ с достаточным объемом текста для прохождения фильтра качества.',
                'metadata': {'source': 'test.pdf', 'page': 2}
            },
        }

        # Мокируем VectorStoreManager
        with patch.object(pipeline.vector_store_manager, 'create_from_texts_with_cache') as mock_vectorize:
            with patch.object(pipeline.corpus_loader, 'load_from_pdf_directory_with_metadata', return_value=mock_documents):
                # Вызываем метод загрузки
                pipeline.load_from_pdf_directory_with_metadata(Path('test/path'), force_reload=True)

                # Проверяем что vectorize был вызван
                assert mock_vectorize.called
                # Получаем переданные аргументы
                call_args = mock_vectorize.call_args
                passed_documents = call_args[0][0]  # Первый позиционный аргумент

                # После фильтрации должно остаться только 2 документа (длинные)
                assert len(passed_documents) == 2
                assert 'doc_1' in passed_documents
                assert 'doc_3' in passed_documents
                assert 'doc_2' not in passed_documents  # Короткий документ отфильтрован

    def test_token_filtering_logs_statistics(self, caplog):
        """Фильтрация логирует статистику удаления"""
        pipeline = RAGPipeline(chunk_size=300)

        mock_documents = {
            'good_1': {
                'text': 'Достаточно длинный текст с информацией.',
                'metadata': {}
            },
            'bad_1': {
                'text': 'Короткий',
                'metadata': {}
            },
        }

        with patch.object(pipeline.vector_store_manager, 'create_from_texts_with_cache'):
            with patch.object(pipeline.corpus_loader, 'load_from_pdf_directory_with_metadata', return_value=mock_documents):
                pipeline.load_from_pdf_directory_with_metadata(Path('test/path'), force_reload=True)

                # Проверяем что был залогирован статистика фильтрации
                assert any('Фильтрация по токенам' in record.message for record in caplog.records)

    def test_token_filtering_with_real_documents(self):
        """Фильтрация работает с реальными документами"""
        pipeline = RAGPipeline(chunk_size=400)

        # Реальные документы с разными размерами
        mock_documents = {
            'realistic_1': {
                'text': 'Первое предложение со смыслом. Второе предложение. Третье предложение с деталями.',
                'metadata': {'source': 'doc1.pdf', 'page': 1}
            },
            'realistic_2': {
                'text': 'Еще одно достаточно длинное предложение с информацией для тестирования фильтра качества.',
                'metadata': {'source': 'doc1.pdf', 'page': 2}
            },
            'garbage_1': {
                'text': '.',
                'metadata': {}
            },
            'garbage_2': {
                'text': 'Word',
                'metadata': {}
            },
        }

        with patch.object(pipeline.vector_store_manager, 'create_from_texts_with_cache') as mock_vec:
            with patch.object(pipeline.corpus_loader, 'load_from_pdf_directory_with_metadata', return_value=mock_documents):
                pipeline.load_from_pdf_directory_with_metadata(Path('test'), force_reload=True)

                # Проверяем что переданы отфильтрованные документы
                assert mock_vec.called
                passed_docs = mock_vec.call_args[0][0]
                assert len(passed_docs) < len(mock_documents)  # Некоторые удалены

    def test_pipeline_handles_empty_after_filtering(self):
        """Пайплайн корректно обрабатывает случай когда все документы отфильтрованы"""
        pipeline = RAGPipeline(chunk_size=300)

        # Только короткие документы
        mock_documents = {
            'short_1': {'text': 'A', 'metadata': {}},
            'short_2': {'text': 'B', 'metadata': {}},
        }

        with patch.object(pipeline.vector_store_manager, 'create_from_texts_with_cache') as mock_vec:
            with patch.object(pipeline.corpus_loader, 'load_from_pdf_directory_with_metadata', return_value=mock_documents):
                pipeline.load_from_pdf_directory_with_metadata(Path('test'), force_reload=True)

                # Vectorize не должен быть вызван если нет документов после фильтра
                # Или был вызван с пустым словарем
                if mock_vec.called:
                    passed_docs = mock_vec.call_args[0][0]
                    assert len(passed_docs) == 0 or isinstance(passed_docs, dict) and not passed_docs

    def test_token_filtering_preserves_metadata(self):
        """Фильтрация сохраняет метаданные документов"""
        pipeline = RAGPipeline(chunk_size=300)

        mock_documents = {
            'doc_with_meta': {
                'text': 'Достаточно длинный документ с полезной информацией.',
                'metadata': {
                    'source': 'important.pdf',
                    'author': 'Test Author',
                    'page': 5
                }
            },
        }

        with patch.object(pipeline.vector_store_manager, 'create_from_texts_with_cache') as mock_vec:
            with patch.object(pipeline.corpus_loader, 'load_from_pdf_directory_with_metadata', return_value=mock_documents):
                pipeline.load_from_pdf_directory_with_metadata(Path('test'), force_reload=True)

                assert mock_vec.called
                metadata = mock_vec.call_args[1]['metadata_dict']
                assert metadata is not None
                if 'doc_with_meta' in metadata:
                    assert metadata['doc_with_meta']['source'] == 'important.pdf'
                    assert metadata['doc_with_meta']['author'] == 'Test Author'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
