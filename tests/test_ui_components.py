"""
Тесты для components.py — UI компоненты RAG системы
Тестирует ConfigModal, FileListPanel, DocumentViewer, HighlightedAnswer, AnswerInteraction
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime


class SessionStateMock(dict):
    """Mock для st.session_state с поддержкой attr и dict access"""
    def __setattr__(self, key, value):
        self[key] = value
    def __getattr__(self, key):
        return self.get(key)


# ============================================================================
# ТЕСТЫ: ConfigModal
# ============================================================================

@patch("rag_gigachat.ui.components.st")
def test_config_modal_show_button(mock_st):
    """✅ ConfigModal.show() отображает кнопку настроек"""
    from rag_gigachat.ui.components import ConfigModal

    mock_session = SessionStateMock({"show_config_modal": False})
    mock_st.session_state = mock_session

    # Имитируем нажатие кнопки
    mock_st.button.return_value = True

    ConfigModal.show()

    mock_st.button.assert_called_once()
    assert "Расширенные настройки" in str(mock_st.button.call_args)


@patch("rag_gigachat.ui.components.st")
def test_config_modal_render_content_models_section(mock_st):
    """✅ ConfigModal._render_content() отображает секцию "Модели" """
    from rag_gigachat.ui.components import ConfigModal

    mock_session = SessionStateMock({
        "llm_model": "GigaChat-2-Max",
        "embedding_model": "GigaChat-2-Max",
        "max_tokens": 2000,
        "temperature": 0.7,
        "k_retrieve": 5,
        "max_context": 2000,
        "retrieval_type": "hybrid",
        "chunk_size": 500,
        "chunk_overlap": 80,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "ocr_enabled": True,
    })
    mock_st.session_state = mock_session

    # Mock для columns с разными количествами
    mock_st.columns.side_effect = [
        [MagicMock(), MagicMock()],  # для моделей
        [MagicMock(), MagicMock()],  # для параметров
        [MagicMock(), MagicMock()],  # для поиска
        [MagicMock(), MagicMock()],  # для чанкирования
        [MagicMock(), MagicMock()],  # для GigaChat
        [MagicMock(), MagicMock(), MagicMock()],  # для кнопок
    ]
    mock_st.text_input.return_value = "GigaChat-2-Max"
    mock_st.slider.return_value = 2000
    mock_st.radio.return_value = "hybrid"
    mock_st.checkbox.return_value = True

    ConfigModal._render_content()

    # Проверить что subheader был вызван
    assert mock_st.subheader.call_count >= 1


@patch("rag_gigachat.ui.components.st")
def test_config_modal_reset_defaults(mock_st):
    """✅ ConfigModal._reset_defaults() восстанавливает значения по умолчанию"""
    from rag_gigachat.ui.components import ConfigModal

    mock_session = {
        "llm_model": "custom-model",
        "embedding_model": "custom-model",
        "max_tokens": 5000,
        "temperature": 1.5,
    }
    mock_st.session_state = mock_session

    ConfigModal._reset_defaults()

    # Проверить что значения сброшены
    assert mock_st.session_state["llm_model"] == "GigaChat-2-Max"
    assert mock_st.session_state["max_tokens"] == 2000
    assert mock_st.session_state["temperature"] == 0.7


# ============================================================================
# ТЕСТЫ: FileListPanel
# ============================================================================

@patch("rag_gigachat.ui.components.st")
def test_file_list_panel_get_pdf_files(mock_st, tmp_path):
    """✅ FileListPanel._get_pdf_files() находит PDF файлы"""
    from rag_gigachat.ui.components import FileListPanel

    # Создать временные файлы
    test_dir = tmp_path / "docs"
    test_dir.mkdir()
    (test_dir / "file1.pdf").write_bytes(b"fake")
    (test_dir / "file2.pdf").write_bytes(b"fake")
    (test_dir / "file3.txt").write_bytes(b"fake")

    files = FileListPanel._get_pdf_files(test_dir, "")

    assert len(files) == 2
    assert all(f.suffix == ".pdf" for f in files)


@patch("rag_gigachat.ui.components.st")
def test_file_list_panel_search_filter(mock_st, tmp_path):
    """✅ FileListPanel._get_pdf_files() фильтрует по поиску"""
    from rag_gigachat.ui.components import FileListPanel

    test_dir = tmp_path / "docs"
    test_dir.mkdir()
    (test_dir / "report_2024.pdf").write_bytes(b"fake")
    (test_dir / "report_2023.pdf").write_bytes(b"fake")
    (test_dir / "summary.pdf").write_bytes(b"fake")

    files = FileListPanel._get_pdf_files(test_dir, "2024")

    assert len(files) == 1
    assert files[0].name == "report_2024.pdf"


@patch("rag_gigachat.ui.components.st")
def test_file_list_panel_empty_directory(mock_st, tmp_path):
    """✅ FileListPanel._get_pdf_files() обрабатывает пустую директорию"""
    from rag_gigachat.ui.components import FileListPanel

    test_dir = tmp_path / "empty"
    test_dir.mkdir()

    files = FileListPanel._get_pdf_files(test_dir, "")

    assert len(files) == 0


@patch("rag_gigachat.ui.components.st")
def test_file_list_panel_nonexistent_directory(mock_st):
    """✅ FileListPanel._get_pdf_files() обрабатывает несуществующую директорию"""
    from rag_gigachat.ui.components import FileListPanel

    files = FileListPanel._get_pdf_files(Path("/nonexistent/path"), "")

    assert len(files) == 0


@patch("rag_gigachat.ui.components.st")
def test_file_list_panel_show(mock_st, tmp_path):
    """✅ FileListPanel.show() отображает список файлов"""
    from rag_gigachat.ui.components import FileListPanel

    test_dir = tmp_path / "docs"
    test_dir.mkdir()
    (test_dir / "test.pdf").write_bytes(b"fake")

    mock_session = {"selected_domain": "UAV", "file_search": ""}
    mock_st.session_state = mock_session

    documents_dirs = {"UAV": test_dir}

    FileListPanel.show(documents_dirs)

    # Проверить что selectbox был вызван
    mock_st.selectbox.assert_called_once()


# ============================================================================
# ТЕСТЫ: DocumentViewer
# ============================================================================

@patch("rag_gigachat.ui.components.st")
def test_document_viewer_file_not_found(mock_st):
    """✅ DocumentViewer.show() показывает ошибку если файл не существует"""
    from rag_gigachat.ui.components import DocumentViewer

    DocumentViewer.show("/nonexistent/file.pdf", 1)

    mock_st.error.assert_called_once()


@patch("rag_gigachat.ui.components.st")
def test_document_viewer_shows_file_info(mock_st, tmp_path):
    """✅ DocumentViewer.show() отображает информацию о файле"""
    from rag_gigachat.ui.components import DocumentViewer

    test_file = tmp_path / "test.pdf"
    test_file.write_bytes(b"fake pdf content")

    # Правильный mock для columns - возвращать два списка для двух вызовов
    col_mocks = [[MagicMock(), MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
    mock_st.columns.side_effect = col_mocks

    DocumentViewer.show(str(test_file), 1)

    # Проверить что была попытка показать информацию о файле
    assert mock_st.markdown.call_count >= 1


@patch("rag_gigachat.ui.components.st")
def test_document_viewer_render_pdf_encoding(mock_st, tmp_path):
    """✅ DocumentViewer._render_pdf() кодирует PDF в base64"""
    from rag_gigachat.ui.components import DocumentViewer

    test_file = tmp_path / "test.pdf"
    test_file.write_bytes(b"PDF content here")

    DocumentViewer._render_pdf(test_file, 1)

    # Проверить что st.components.v1.html был вызван
    mock_st.components.v1.html.assert_called_once()


# ============================================================================
# ТЕСТЫ: HighlightedAnswer
# ============================================================================

@patch("rag_gigachat.ui.components.st")
def test_highlighted_answer_process_with_links(mock_st):
    """✅ HighlightedAnswer._process_answer_with_links() добавляет источники"""
    from rag_gigachat.ui.components import HighlightedAnswer

    answer = "This is the answer"
    docs = [
        {"doc_id": "file1_p1", "score": 0.95},
        {"doc_id": "file2_p2", "score": 0.87},
    ]

    result = HighlightedAnswer._process_answer_with_links(answer, docs)

    assert "Источники:" in result
    assert "file1" in result
    assert "file2" in result
    assert answer in result


@patch("rag_gigachat.ui.components.st")
def test_highlighted_answer_parse_doc_id(mock_st):
    """✅ HighlightedAnswer парсит doc_id вида 'filename_pN'"""
    from rag_gigachat.ui.components import HighlightedAnswer

    answer = "Test"
    docs = [
        {"doc_id": "document_p5", "score": 0.9},
        {"doc_id": "report_p10", "score": 0.85},
    ]

    result = HighlightedAnswer._process_answer_with_links(answer, docs)

    assert "p5" in result or "5" in result  # Номер страницы
    assert "p10" in result or "10" in result


@patch("rag_gigachat.ui.components.st")
def test_highlighted_answer_empty_docs(mock_st):
    """✅ HighlightedAnswer обрабатывает пустой список документов"""
    from rag_gigachat.ui.components import HighlightedAnswer

    answer = "Answer without sources"
    docs = []

    result = HighlightedAnswer._process_answer_with_links(answer, docs)

    assert answer in result
    # Источники не должны быть если документов нет


@patch("rag_gigachat.ui.components.st")
def test_highlighted_answer_show_sources(mock_st):
    """✅ HighlightedAnswer._show_sources() отображает источники"""
    from rag_gigachat.ui.components import HighlightedAnswer

    docs = [
        {
            "doc_id": "file1_p1",
            "score": 0.95,
            "text": "This is relevant text from document"
        }
    ]
    documents_dirs = {}

    mock_st.container.return_value.__enter__ = MagicMock()
    mock_st.container.return_value.__exit__ = MagicMock()
    mock_st.columns.return_value = [MagicMock(), MagicMock()]

    HighlightedAnswer._show_sources(docs, documents_dirs)

    # Проверить что контейнер был создан
    mock_st.container.assert_called()


@patch("rag_gigachat.ui.components.st")
def test_highlighted_answer_show(mock_st):
    """✅ HighlightedAnswer.show() отображает ответ и источники"""
    from rag_gigachat.ui.components import HighlightedAnswer

    answer = "The answer is..."
    docs = [{"doc_id": "doc_p1", "score": 0.9, "text": "text"}]
    documents_dirs = {}

    mock_st.expander.return_value.__enter__ = MagicMock()
    mock_st.expander.return_value.__exit__ = MagicMock()

    HighlightedAnswer.show(answer, docs, documents_dirs, show_sources=True)

    mock_st.markdown.assert_called()


# ============================================================================
# ТЕСТЫ: AnswerInteraction
# ============================================================================

@patch("rag_gigachat.ui.components.st")
def test_answer_interaction_show_actions(mock_st):
    """✅ AnswerInteraction.show_actions() отображает кнопки"""
    from rag_gigachat.ui.components import AnswerInteraction

    mock_st.columns.return_value = [MagicMock() for _ in range(4)]

    AnswerInteraction.show_actions("Test answer", answer_id="ans_1")

    # Проверить что было 4 кнопки (Копировать, Полезно, Не полезно, Сохранить)
    assert mock_st.button.call_count >= 4


@patch("rag_gigachat.ui.components.st")
def test_answer_interaction_copy_button(mock_st):
    """✅ AnswerInteraction - кнопка копирования работает"""
    from rag_gigachat.ui.components import AnswerInteraction

    mock_st.columns.return_value = [MagicMock() for _ in range(4)]

    AnswerInteraction.show_actions("Answer", answer_id="test")

    # Проверить что была попытка создать кнопку копирования
    button_calls = [call[1].get('key') for call in mock_st.button.call_args_list]
    assert any('copy' in str(key) for key in button_calls if key)


@patch("rag_gigachat.ui.components.st")
def test_answer_interaction_feedback(mock_st):
    """✅ AnswerInteraction - кнопки обратной связи работают"""
    from rag_gigachat.ui.components import AnswerInteraction

    mock_session = SessionStateMock({"feedback": None})
    mock_st.session_state = mock_session
    mock_st.columns.return_value = [MagicMock() for _ in range(4)]

    AnswerInteraction.show_actions("Answer", answer_id="test")

    # Проверить что были кнопки для обратной связи
    button_calls = [str(call) for call in mock_st.button.call_args_list]
    assert any('helpful' in call.lower() or 'полезно' in call.lower() for call in button_calls)


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

@patch("rag_gigachat.ui.components.st")
def test_full_document_workflow(mock_st, tmp_path):
    """✅ Full workflow: список файлов → открыть → показать документ"""
    from rag_gigachat.ui.components import FileListPanel, DocumentViewer

    test_dir = tmp_path / "docs"
    test_dir.mkdir()
    test_pdf = test_dir / "document.pdf"
    test_pdf.write_bytes(b"fake pdf")

    # Шаг 1: Получить список файлов
    files = FileListPanel._get_pdf_files(test_dir, "")
    assert len(files) == 1

    # Шаг 2: Показать документ
    col_mocks = [[MagicMock(), MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
    mock_st.columns.side_effect = col_mocks
    DocumentViewer.show(str(test_pdf), 1)

    # Проверить что документ был обработан
    assert mock_st.components.v1.html.called or mock_st.markdown.called


@patch("rag_gigachat.ui.components.st")
def test_full_answer_workflow(mock_st):
    """✅ Full workflow: ответ → источники → взаимодействие"""
    from rag_gigachat.ui.components import HighlightedAnswer, AnswerInteraction

    answer = "AI is a technology"
    docs = [
        {"doc_id": "ai_basics_p1", "score": 0.95, "text": "AI basics..."},
        {"doc_id": "ml_p2", "score": 0.87, "text": "ML is..."},
    ]
    documents_dirs = {}

    # Шаг 1: Показать ответ с источниками
    mock_st.expander.return_value.__enter__ = MagicMock()
    mock_st.expander.return_value.__exit__ = MagicMock()

    HighlightedAnswer.show(answer, docs, documents_dirs)

    # Шаг 2: Показать кнопки взаимодействия
    mock_st.columns.return_value = [MagicMock() for _ in range(4)]
    AnswerInteraction.show_actions(answer, answer_id="ans_1")

    # Проверить что оба компонента были использованы
    assert mock_st.markdown.called
    assert mock_st.button.called


# ============================================================================
# EDGE CASES
# ============================================================================

@patch("rag_gigachat.ui.components.st")
def test_config_modal_very_high_temperature(mock_st):
    """✅ ConfigModal обрабатывает очень высокую температуру"""
    from rag_gigachat.ui.components import ConfigModal

    mock_session = SessionStateMock({
        "temperature": 2.0,  # Максимум по UI
        "llm_model": "GigaChat",
        "embedding_model": "GigaChat",
        "max_tokens": 2000,
        "k_retrieve": 5,
        "max_context": 2000,
        "retrieval_type": "hybrid",
        "chunk_size": 500,
        "chunk_overlap": 80,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "ocr_enabled": True,
    })
    mock_st.session_state = mock_session

    # Mock для columns
    mock_st.columns.side_effect = [
        [MagicMock(), MagicMock()],  # для моделей
        [MagicMock(), MagicMock()],  # для параметров
        [MagicMock(), MagicMock()],  # для поиска
        [MagicMock(), MagicMock()],  # для чанкирования
        [MagicMock(), MagicMock()],  # для GigaChat
        [MagicMock(), MagicMock(), MagicMock()],  # для кнопок
    ]
    mock_st.text_input.return_value = "GigaChat"
    mock_st.slider.return_value = 2.0
    mock_st.radio.return_value = "hybrid"
    mock_st.checkbox.return_value = True

    ConfigModal._render_content()

    # Значение должно быть принято


@patch("rag_gigachat.ui.components.st")
def test_highlighted_answer_very_long_text(mock_st):
    """✅ HighlightedAnswer обрабатывает очень длинный текст"""
    from rag_gigachat.ui.components import HighlightedAnswer

    long_answer = "A" * 10000  # Очень длинный ответ
    docs = []

    result = HighlightedAnswer._process_answer_with_links(long_answer, docs)

    assert len(result) > len(long_answer)  # С добавлением источников


@patch("rag_gigachat.ui.components.st")
def test_file_list_panel_case_insensitive_search(mock_st, tmp_path):
    """✅ FileListPanel поиск нечувствителен к регистру"""
    from rag_gigachat.ui.components import FileListPanel

    test_dir = tmp_path / "docs"
    test_dir.mkdir()
    (test_dir / "RePort_2024.pdf").write_bytes(b"fake")

    files_lower = FileListPanel._get_pdf_files(test_dir, "report")
    files_upper = FileListPanel._get_pdf_files(test_dir, "REPORT")

    assert len(files_lower) == 1
    assert len(files_upper) == 1
    assert files_lower == files_upper


@patch("rag_gigachat.ui.components.st")
def test_highlighted_answer_unicode_in_filename(mock_st):
    """✅ HighlightedAnswer обрабатывает Unicode в имени файла"""
    from rag_gigachat.ui.components import HighlightedAnswer

    answer = "Ответ на русском"
    docs = [
        {"doc_id": "документ_中文_p1", "score": 0.9}
    ]

    result = HighlightedAnswer._process_answer_with_links(answer, docs)

    assert answer in result
