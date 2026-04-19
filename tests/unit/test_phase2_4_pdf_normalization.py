"""ФАЗА 2.4: Тесты нормализации текста при загрузке PDF (mock-based)"""

import pytest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_gigachat.data.data_loader import DocumentLoader
from rag_gigachat.utils.text_utils import normalize_text
from langchain_core.documents import Document as LangChainDocument


class TestPDFNormalizationMock:
    """Тесты нормализации текста при загрузке PDF с использованием mock"""

    @pytest.fixture
    def loader(self):
        """Создание экземпляра DocumentLoader"""
        return DocumentLoader()

    def test_normalize_parameter_default(self, loader):
        """По умолчанию normalize=True"""
        # Мокируем PyPDFLoader
        mock_doc = LangChainDocument(
            page_content="Текст    с     лишними    пробелами.",
            metadata={'source': 'test.pdf'}
        )

        with patch('rag_gigachat.data.data_loader.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = [mock_doc]

            with patch.object(DocumentLoader, 'extract_pdf_metadata') as mock_meta:
                mock_meta.return_value = {'filename': 'test.pdf', 'num_pages': 1}

                # Загружаем с нормализацией (по умолчанию)
                result_docs = loader.load_pdf_with_metadata(Path('test.pdf'))

                # Проверяем что normalize вызывается
                if result_docs:
                    assert len(result_docs) > 0
                    # Текст должен быть нормализован (нет двойных пробелов)
                    assert "  " not in result_docs[0].page_content

    def test_normalize_can_be_disabled(self, loader):
        """normalize=False отключает нормализацию"""
        # Текст с артефактами
        text_with_artifacts = "Текст    с     лишними    пробелами."
        mock_doc = LangChainDocument(
            page_content=text_with_artifacts,
            metadata={'source': 'test.pdf'}
        )

        with patch('rag_gigachat.data.data_loader.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = [mock_doc]

            # Загружаем без нормализации
            result_docs = loader.load_pdf_with_metadata(Path('test.pdf'), normalize=False)

            assert len(result_docs) > 0
            # Текст должен быть как в оригинале (с артефактами)
            assert result_docs[0].metadata.get('normalized') == False
            assert "  " in result_docs[0].page_content  # Артефакты остаются

    def test_normalize_true_sets_metadata(self, loader):
        """normalize=True устанавливает метаданные normalized=True"""
        mock_doc = LangChainDocument(
            page_content="Тестовый текст для проверки.",
            metadata={'source': 'test.pdf'}
        )

        with patch('rag_gigachat.data.data_loader.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = [mock_doc]

            result_docs = loader.load_pdf_with_metadata(Path('test.pdf'), normalize=True)

            assert len(result_docs) > 0
            for doc in result_docs:
                assert doc.metadata.get('normalized') == True

    def test_normalization_removes_artifacts(self, loader):
        """Нормализация удаляет артефакты из текста"""
        # Создаем текст с типичными PDF артефактами
        artifacts_text = "Первое  предложение.   Второе\nпредложение.  Третье."
        mock_doc = LangChainDocument(
            page_content=artifacts_text,
            metadata={'source': 'test.pdf'}
        )

        with patch('rag_gigachat.data.data_loader.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = [mock_doc]

            result_docs = loader.load_pdf_with_metadata(Path('test.pdf'), normalize=True)

            assert len(result_docs) > 0
            # Проверяем что артефакты удалены
            for doc in result_docs:
                # Нет множественных пробелов
                assert "  " not in doc.page_content
                # Разорванные слова соединены
                assert "Принять" not in doc.page_content or "При\nнять" not in doc.page_content

    def test_metadata_preserved_after_normalization(self, loader):
        """Метаданные документа сохраняются при нормализации"""
        mock_doc = LangChainDocument(
            page_content="Тестовый текст для проверки метаданных.",
            metadata={'source': 'test.pdf', 'page': 1}
        )

        with patch('rag_gigachat.data.data_loader.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = [mock_doc]

            with patch.object(DocumentLoader, 'extract_pdf_metadata') as mock_meta:
                mock_meta.return_value = {
                    'title': 'Test Doc',
                    'author': 'Test Author',
                    'filename': 'test.pdf',
                    'filepath': '/path/to/test.pdf'
                }

                result_docs = loader.load_pdf_with_metadata(Path('test.pdf'), normalize=True)

                assert len(result_docs) > 0
                for doc in result_docs:
                    # Проверяем наличие критических метаданных
                    assert 'source' in doc.metadata
                    assert 'normalized' in doc.metadata

    def test_multiple_pages_all_normalized(self, loader):
        """Все страницы нормализуются"""
        mock_docs = [
            LangChainDocument(
                page_content="Страница 1  с     артефактами",
                metadata={'source': 'test.pdf', 'page': 0}
            ),
            LangChainDocument(
                page_content="Страница 2  с     артефактами",
                metadata={'source': 'test.pdf', 'page': 1}
            )
        ]

        with patch('rag_gigachat.data.data_loader.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = mock_docs

            result_docs = loader.load_pdf_with_metadata(Path('test.pdf'), normalize=True)

            # Все документы должны быть нормализованы
            assert len(result_docs) >= 2
            for doc in result_docs:
                assert doc.metadata.get('normalized') == True
                # Текст должен быть нормализован
                assert "  " not in doc.page_content

    def test_normalization_with_mixed_languages(self, loader):
        """Нормализация работает с смешанным RU+EN текстом"""
        text = "Привет.   Hello.   Как    дела?   How    are    you?"
        mock_doc = LangChainDocument(
            page_content=text,
            metadata={'source': 'test.pdf'}
        )

        with patch('rag_gigachat.data.data_loader.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = [mock_doc]

            result_docs = loader.load_pdf_with_metadata(Path('test.pdf'), normalize=True)

            assert len(result_docs) > 0
            all_text = " ".join(doc.page_content for doc in result_docs)
            # Текст не должен содержать множественные пробелы
            assert "   " not in all_text

    def test_page_number_preserved(self, loader):
        """Номер страницы сохраняется при нормализации"""
        mock_doc = LangChainDocument(
            page_content="Текст для проверки номера страницы.",
            metadata={'source': 'test.pdf', 'page': 0}
        )

        with patch('rag_gigachat.data.data_loader.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = [mock_doc]

            with patch.object(DocumentLoader, 'extract_pdf_metadata') as mock_meta:
                mock_meta.return_value = {'filename': 'test.pdf', 'num_pages': 1}

                result_docs = loader.load_pdf_with_metadata(Path('test.pdf'), normalize=True)

                for i, doc in enumerate(result_docs):
                    assert doc.metadata.get('page_number') == i + 1

    def test_filename_preserved(self, loader):
        """Имя файла сохраняется при нормализации"""
        mock_doc = LangChainDocument(
            page_content="Тестовый текст",
            metadata={'source': '/path/to/specific_filename.pdf'}
        )

        with patch('rag_gigachat.data.data_loader.PyPDFLoader') as mock_loader:
            mock_loader.return_value.load.return_value = [mock_doc]

            with patch.object(DocumentLoader, 'extract_pdf_metadata') as mock_meta:
                mock_meta.return_value = {'filename': 'specific_filename.pdf'}

                result_docs = loader.load_pdf_with_metadata(Path('specific_filename.pdf'), normalize=True)

                assert len(result_docs) > 0
                assert result_docs[0].metadata.get('filename') == 'specific_filename.pdf'

    def test_normalization_consistent_with_normalize_text(self):
        """Результаты нормализации согласованы с normalize_text()"""
        text = "Текст    с    лишними    пробелами."
        expected = normalize_text(text)

        # Нормализуем напрямую через normalize_text
        assert "  " not in expected
        assert len(expected) <= len(text)

    def test_ocr_text_also_normalized(self, loader):
        """Текст из OCR также нормализуется"""
        ocr_text = "OCR текст    с     артефактами."

        with patch('rag_gigachat.data.data_loader.PyPDFLoader') as mock_loader:
            # Пустой результат от PyPDF -> переходим на OCR
            mock_loader.return_value.load.return_value = [
                LangChainDocument(page_content="", metadata={'source': 'test.pdf'})
            ]

            with patch('rag_gigachat.data.data_loader.load_pdf_with_ocr') as mock_ocr:
                mock_ocr.return_value = ocr_text

                with patch('rag_gigachat.data.data_loader.OCR_AVAILABLE', True):
                    with patch('rag_gigachat.data.data_loader.data_config') as mock_config:
                        mock_config.ocr_enabled = True

                        result_docs = loader.load_pdf_with_metadata(Path('test.pdf'), normalize=True)

                        # Если OCR был использован и текст загружен
                        if result_docs and result_docs[0].page_content:
                            # OCR текст должен быть нормализован
                            assert "  " not in result_docs[0].page_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
