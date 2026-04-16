"""
tests/test_ocr.py — тесты OCR-функциональности в data_loader.py

Все тесты работают без реального Docling и без PDF-файлов:
используются tmp_path, mock и patch.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
# sys.path.insert больше не нужен - пакет в sys.path автоматически

import rag_gigachat.data.data_loader as data_loader


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------

def _make_fake_pdf(tmp_path: Path, size_bytes: int = 1024, name: str = "test.pdf") -> Path:
    """Создаёт фиктивный PDF-файл заданного размера."""
    pdf = tmp_path / name
    pdf.write_bytes(b"%PDF-1.4 " + b"x" * size_bytes)
    return pdf


# ---------------------------------------------------------------------------
# load_pdf_with_ocr
# ---------------------------------------------------------------------------

class TestLoadPdfWithOcr:

    def test_returns_empty_when_ocr_not_available(self, tmp_path):
        """Если Docling не установлен — возвращает '' без исключения."""
        pdf = _make_fake_pdf(tmp_path)
        with patch.object(data_loader, "OCR_AVAILABLE", False):
            result = data_loader.load_pdf_with_ocr(pdf)
        assert result == ""

    def test_returns_empty_when_ocr_disabled_in_config(self, tmp_path):
        """Если ocr_enabled=False в конфиге — OCR не запускается."""
        pdf = _make_fake_pdf(tmp_path)
        with patch.object(data_loader, "OCR_AVAILABLE", True), \
             patch.object(data_loader.data_config, "ocr_enabled", False):
            result = data_loader.load_pdf_with_ocr(pdf)
        assert result == ""

    def test_returns_empty_when_file_too_large(self, tmp_path):
        """Файл больше лимита → '' без запуска OCR."""
        pdf = _make_fake_pdf(tmp_path, size_bytes=1024)
        with patch.object(data_loader, "OCR_AVAILABLE", True), \
             patch.object(data_loader.data_config, "ocr_enabled", True), \
             patch.object(data_loader.data_config, "ocr_max_file_size_mb", 0):
            result = data_loader.load_pdf_with_ocr(pdf)
        assert result == ""

    def test_returns_cached_text_without_calling_docling(self, tmp_path):
        """При наличии кэша — возвращает кэш, Docling не вызывается."""
        pdf = _make_fake_pdf(tmp_path)
        cache_dir = tmp_path / "ocr"
        cache_dir.mkdir()
        file_hash = data_loader._pdf_file_hash(pdf)
        cache_file = cache_dir / f"{file_hash}.txt"
        cache_file.write_text("кэшированный текст", encoding="utf-8")

        mock_converter = MagicMock()

        with patch.object(data_loader, "OCR_AVAILABLE", True), \
             patch.object(data_loader.data_config, "ocr_enabled", True), \
             patch.object(data_loader.data_config, "ocr_max_file_size_mb", 50), \
             patch.object(data_loader.data_config, "cache_dir", tmp_path), \
             patch("rag_gigachat.data.data_loader._get_ocr_converter", return_value=mock_converter):
            result = data_loader.load_pdf_with_ocr(pdf)

        assert result == "кэшированный текст"
        mock_converter.convert.assert_not_called()

    def test_saves_ocr_result_to_cache(self, tmp_path):
        """После OCR — результат сохраняется в кэш-файл."""
        pdf = _make_fake_pdf(tmp_path)

        mock_doc = MagicMock()
        mock_doc.export_to_text.return_value = "распознанный текст"
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with patch.object(data_loader, "OCR_AVAILABLE", True), \
             patch.object(data_loader.data_config, "ocr_enabled", True), \
             patch.object(data_loader.data_config, "ocr_max_file_size_mb", 50), \
             patch.object(data_loader.data_config, "cache_dir", tmp_path), \
             patch("rag_gigachat.data.data_loader._get_ocr_converter", return_value=mock_converter):
            result = data_loader.load_pdf_with_ocr(pdf)

        assert result == "распознанный текст"
        file_hash = data_loader._pdf_file_hash(pdf)
        cache_file = tmp_path / "ocr" / f"{file_hash}.txt"
        assert cache_file.exists()
        assert cache_file.read_text(encoding="utf-8") == "распознанный текст"

    def test_returns_empty_on_docling_exception(self, tmp_path):
        """Ошибка Docling → '' без пробрасывания исключения."""
        pdf = _make_fake_pdf(tmp_path)
        mock_converter = MagicMock()
        mock_converter.convert.side_effect = RuntimeError("Docling сломался")

        with patch.object(data_loader, "OCR_AVAILABLE", True), \
             patch.object(data_loader.data_config, "ocr_enabled", True), \
             patch.object(data_loader.data_config, "ocr_max_file_size_mb", 50), \
             patch.object(data_loader.data_config, "cache_dir", tmp_path), \
             patch("rag_gigachat.data.data_loader._get_ocr_converter", return_value=mock_converter):
            result = data_loader.load_pdf_with_ocr(pdf)

        assert result == ""


# ---------------------------------------------------------------------------
# OCR fallback в load_pdf_with_metadata
# ---------------------------------------------------------------------------

class TestLoadPdfWithMetadataOcrFallback:

    def _make_loader(self):
        """Создаёт DocumentLoader с патченными зависимостями."""
        from rag_gigachat.data.data_loader import DocumentLoader
        with patch("rag_gigachat.data.data_loader.PyPDFLoader"), \
             patch("rag_gigachat.data.data_loader.OCR_AVAILABLE", True):
            loader = DocumentLoader.__new__(DocumentLoader)
            loader.cache = MagicMock()
            loader.cache.get.return_value = None
            return loader

    def _make_document_loader(self):
        """Создаёт экземпляр DocumentLoader с заглушками."""
        from rag_gigachat.data.data_loader import DocumentLoader
        loader = DocumentLoader.__new__(DocumentLoader)
        loader.cache = MagicMock()
        loader.cache.get.return_value = None
        return loader

    def _metadata(self, pdf):
        return {
            "title": "", "author": "", "subject": "", "keywords": "",
            "filename": pdf.name, "filepath": str(pdf), "num_pages": 1
        }

    def test_ocr_not_called_when_text_present(self, tmp_path):
        """PyPDFLoader вернул текст → OCR не вызывается."""
        pdf = _make_fake_pdf(tmp_path)

        fake_doc = MagicMock()
        fake_doc.page_content = "Нормальный текст страницы"
        fake_doc.metadata = {}

        with patch("rag_gigachat.data.data_loader.PyPDFLoader") as mock_loader_cls, \
             patch("rag_gigachat.data.data_loader.load_pdf_with_ocr") as mock_ocr, \
             patch("rag_gigachat.data.data_loader.OCR_AVAILABLE", True), \
             patch("rag_gigachat.data.data_loader.logging_config") as mock_log_cfg:
            mock_log_cfg.log_level = "INFO"   # отключаем DEBUG-сохранение файлов
            mock_loader_cls.return_value.load.return_value = [fake_doc]
            loader = self._make_document_loader()
            loader.extract_pdf_metadata = MagicMock(return_value=self._metadata(pdf))
            docs = loader.load_pdf_with_metadata(pdf)

        mock_ocr.assert_not_called()
        assert len(docs) == 1

    def test_ocr_called_when_text_empty(self, tmp_path):
        """PyPDFLoader вернул пустой текст → OCR вызывается."""
        pdf = _make_fake_pdf(tmp_path)

        fake_doc = MagicMock()
        fake_doc.page_content = "   "  # пустой
        fake_doc.metadata = {}

        with patch("rag_gigachat.data.data_loader.PyPDFLoader") as mock_loader_cls, \
             patch("rag_gigachat.data.data_loader.load_pdf_with_ocr", return_value="OCR текст") as mock_ocr, \
             patch("rag_gigachat.data.data_loader.OCR_AVAILABLE", True), \
             patch("rag_gigachat.data.data_loader.logging_config") as mock_log_cfg:
            mock_log_cfg.log_level = "INFO"
            mock_loader_cls.return_value.load.return_value = [fake_doc]
            loader = self._make_document_loader()
            loader.extract_pdf_metadata = MagicMock(return_value=self._metadata(pdf))
            docs = loader.load_pdf_with_metadata(pdf)

        mock_ocr.assert_called_once_with(pdf)


# ---------------------------------------------------------------------------
# _pdf_file_hash
# ---------------------------------------------------------------------------

class TestPdfFileHash:

    def test_same_file_same_hash(self, tmp_path):
        pdf = _make_fake_pdf(tmp_path)
        assert data_loader._pdf_file_hash(pdf) == data_loader._pdf_file_hash(pdf)

    def test_different_files_different_hash(self, tmp_path):
        pdf1 = _make_fake_pdf(tmp_path, size_bytes=100, name="a.pdf")
        pdf2 = _make_fake_pdf(tmp_path, size_bytes=200, name="b.pdf")
        assert data_loader._pdf_file_hash(pdf1) != data_loader._pdf_file_hash(pdf2)

    def test_hash_is_16_chars(self, tmp_path):
        pdf = _make_fake_pdf(tmp_path)
        assert len(data_loader._pdf_file_hash(pdf)) == 16
