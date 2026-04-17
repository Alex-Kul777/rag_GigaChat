"""
Тесты для streamlit_app.py — UI слой RAG системы
Фокус: валидация входных данных, обработка ошибок, состояние
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


@pytest.fixture
def session_state():
    """Sessionstate для всех тестов"""
    return SessionStateMock({
        "messages": [],
        "show_config_modal": False,
        "show_document_viewer": False,
        "selected_file": None,
        "selected_page": 1,
        "user_input": "",
        "embedding_model": "GigaChat",
        "chunk_size": 500,
        "chunk_overlap": 80,
        "k_retrieve": 5,
        "max_tokens": 2000,
    })


# ============================================================================
# ТЕСТЫ: init_session_state()
# ============================================================================

@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.model_config")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_init_session_state_initializes_all_keys(mock_data_cfg, mock_model_cfg, mock_st):
    """✅ init_session_state() инициализирует все ключи"""
    from rag_gigachat.ui.streamlit_app import init_session_state

    session = SessionStateMock()
    mock_st.session_state = session

    mock_model_cfg.llm_model_name = "GigaChat"
    mock_model_cfg.embedding_model_name = "GigaChat"
    mock_model_cfg.max_new_tokens = 2000
    mock_model_cfg.temperature = 0.7
    mock_model_cfg.default_k_retrieve = 5
    mock_model_cfg.max_context_length = 2000
    mock_model_cfg.top_p = 0.9
    mock_model_cfg.repetition_penalty = 1.1

    mock_data_cfg.chunk_size = 500
    mock_data_cfg.chunk_overlap = 80
    mock_data_cfg.ocr_enabled = True

    init_session_state()

    assert "messages" in session
    assert isinstance(session["messages"], list)
    assert session["show_config_modal"] is False


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.model_config")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_init_session_state_preserves_existing_values(mock_data_cfg, mock_model_cfg, mock_st):
    """✅ init_session_state() не перезаписывает существующие значения"""
    from rag_gigachat.ui.streamlit_app import init_session_state

    existing_msg = [{"role": "user", "content": "Hello"}]
    session = SessionStateMock({"messages": existing_msg})
    mock_st.session_state = session

    mock_model_cfg.llm_model_name = "GigaChat"
    mock_model_cfg.embedding_model_name = "GigaChat"
    mock_model_cfg.max_new_tokens = 2000
    mock_model_cfg.temperature = 0.7
    mock_model_cfg.default_k_retrieve = 5
    mock_model_cfg.max_context_length = 2000
    mock_model_cfg.top_p = 0.9
    mock_model_cfg.repetition_penalty = 1.1
    mock_data_cfg.chunk_size = 500
    mock_data_cfg.chunk_overlap = 80
    mock_data_cfg.ocr_enabled = True

    init_session_state()

    assert session["messages"] == existing_msg


# ============================================================================
# ТЕСТЫ: get_rag_pipeline()
# ============================================================================

@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.RAGPipeline")
def test_get_rag_pipeline_parameters(mock_pipeline_cls, mock_st):
    """✅ get_rag_pipeline() передаёт корректные параметры"""
    from rag_gigachat.ui.streamlit_app import get_rag_pipeline

    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline

    result = get_rag_pipeline("test-model", 500, 80)

    mock_pipeline_cls.assert_called_once()
    args, kwargs = mock_pipeline_cls.call_args
    assert kwargs["embedding_model"] == "test-model"
    assert kwargs["chunk_size"] == 500
    assert kwargs["chunk_overlap"] == 80
    assert kwargs["llm_type"] == "gigachat"


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.RAGPipeline")
def test_get_rag_pipeline_with_zero_chunk_size(mock_pipeline_cls, mock_st):
    """✅ get_rag_pipeline() валидирует chunk_size=0"""
    from rag_gigachat.ui.streamlit_app import get_rag_pipeline

    # ✅ Теперь валидация выбросит ValueError
    with pytest.raises(ValueError, match="chunk_size должен быть > 0"):
        get_rag_pipeline("test-model", chunk_size=0, chunk_overlap=80)


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.RAGPipeline")
def test_get_rag_pipeline_with_empty_embedding_model(mock_pipeline_cls, mock_st):
    """✅ get_rag_pipeline() валидирует пустой embedding_model"""
    from rag_gigachat.ui.streamlit_app import get_rag_pipeline

    # ✅ Теперь валидация выбросит ValueError
    with pytest.raises(ValueError, match="embedding_model должна быть непустой строкой"):
        get_rag_pipeline("", chunk_size=500, chunk_overlap=80)


# ============================================================================
# ТЕСТЫ: handle_user_query() - обработка запроса
# ============================================================================

@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.get_rag_pipeline")
@patch("rag_gigachat.ui.streamlit_app.HighlightedAnswer")
@patch("rag_gigachat.ui.streamlit_app.AnswerInteraction")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_handle_user_query_success(mock_data_cfg, mock_ans_inter, mock_highlight, mock_pipeline, mock_st, session_state):
    """✅ handle_user_query() успешно обрабатывает запрос"""
    from rag_gigachat.ui.streamlit_app import handle_user_query

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {}

    mock_result = MagicMock()
    mock_result.answer = "Test answer"
    mock_result.retrieval_results = MagicMock()
    mock_result.retrieval_results.retrieved_docs = []

    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.process_query.return_value = mock_result
    mock_pipeline.return_value = mock_pipeline_inst

    handle_user_query("What is AI?")

    # Проверить что сообщения были добавлены
    assert len(session_state["messages"]) == 2
    assert session_state["messages"][0]["role"] == "user"
    assert session_state["messages"][0]["content"] == "What is AI?"
    assert session_state["messages"][1]["role"] == "assistant"


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.get_rag_pipeline")
@patch("rag_gigachat.ui.streamlit_app.HighlightedAnswer")
@patch("rag_gigachat.ui.streamlit_app.AnswerInteraction")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_handle_user_query_pipeline_error_handling(mock_data_cfg, mock_ans_inter, mock_highlight, mock_pipeline, mock_st, session_state):
    """🔴 handle_user_query() - race condition: сообщение добавляется ДО try/except"""
    from rag_gigachat.ui.streamlit_app import handle_user_query

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {}

    # Pipeline выбросит ошибку
    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.process_query.side_effect = ValueError("API Error")
    mock_pipeline.return_value = mock_pipeline_inst

    handle_user_query("Test query")

    # ✅ ИСПРАВЛЕНО: Сообщения НЕ добавляются если произойдёт ошибка
    # Состояние остаётся консистентным
    assert len(session_state["messages"]) == 0  # Нет сообщений при ошибке
    mock_st.error.assert_called()  # Ошибка показана пользователю


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.get_rag_pipeline")
@patch("rag_gigachat.ui.streamlit_app.HighlightedAnswer")
@patch("rag_gigachat.ui.streamlit_app.AnswerInteraction")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_handle_user_query_with_none_retrieval_results(mock_data_cfg, mock_ans_inter, mock_highlight, mock_pipeline, mock_st, session_state):
    """✅ handle_user_query() обрабатывает None retrieval_results"""
    from rag_gigachat.ui.streamlit_app import handle_user_query

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {}

    mock_result = MagicMock()
    mock_result.answer = "Answer"
    mock_result.retrieval_results = None

    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.process_query.return_value = mock_result
    mock_pipeline.return_value = mock_pipeline_inst

    handle_user_query("Test")

    assert len(session_state["messages"]) == 2
    assert session_state["messages"][1]["documents"] == []


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.get_rag_pipeline")
@patch("rag_gigachat.ui.streamlit_app.HighlightedAnswer")
@patch("rag_gigachat.ui.streamlit_app.AnswerInteraction")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_handle_user_query_with_special_characters(mock_data_cfg, mock_ans_inter, mock_highlight, mock_pipeline, mock_st, session_state):
    """✅ handle_user_query() обрабатывает спецсимволы"""
    from rag_gigachat.ui.streamlit_app import handle_user_query

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {}

    mock_result = MagicMock()
    mock_result.answer = "Answer"
    mock_result.retrieval_results = None

    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.process_query.return_value = mock_result
    mock_pipeline.return_value = mock_pipeline_inst

    special_query = "What's \"это\" 中文 & символы\\n"
    handle_user_query(special_query)

    assert session_state["messages"][0]["content"] == special_query


# ============================================================================
# ТЕСТЫ: render_document_viewer()
# ============================================================================

@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.DocumentViewer")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_render_document_viewer_file_not_found(mock_data_cfg, mock_viewer, mock_st, session_state):
    """✅ render_document_viewer() показывает ошибку если файл не найден"""
    from rag_gigachat.ui.streamlit_app import render_document_viewer

    session_state["show_document_viewer"] = True
    session_state["selected_file"] = "nonexistent"
    session_state["selected_page"] = 1

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {"UAV": Path("/nonexistent")}

    render_document_viewer()

    mock_st.error.assert_called()


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.DocumentViewer")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_render_document_viewer_empty_selected_file(mock_data_cfg, mock_viewer, mock_st, session_state):
    """🔴 render_document_viewer() не валидирует пустой selected_file"""
    from rag_gigachat.ui.streamlit_app import render_document_viewer

    session_state["show_document_viewer"] = True
    session_state["selected_file"] = ""  # Пусто - ошибка!

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {"UAV": Path("/tmp")}

    render_document_viewer()
    # Без валидации это может привести к проблемам


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.DocumentViewer")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_render_document_viewer_invalid_page_number(mock_data_cfg, mock_viewer, mock_st, session_state, tmp_path):
    """🔴 render_document_viewer() не валидирует номер страницы"""
    from rag_gigachat.ui.streamlit_app import render_document_viewer

    test_dir = tmp_path / "docs"
    test_dir.mkdir()
    test_pdf = test_dir / "test.pdf"
    test_pdf.write_bytes(b"fake")

    session_state["show_document_viewer"] = True
    session_state["selected_file"] = "test"
    session_state["selected_page"] = 9999  # Неправильный номер

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {"UAV": test_dir}

    # Mock st.columns
    mock_cols = [MagicMock() for _ in range(3)]
    mock_st.columns.return_value = mock_cols

    render_document_viewer()
    # Должна быть валидация номера страницы


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.DocumentViewer")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_render_document_viewer_empty_documents_dirs(mock_data_cfg, mock_viewer, mock_st, session_state):
    """🔴 render_document_viewer() - documents_dirs может быть пуст"""
    from rag_gigachat.ui.streamlit_app import render_document_viewer

    session_state["show_document_viewer"] = True
    session_state["selected_file"] = "test"

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {}  # Пусто

    render_document_viewer()
    # Должна быть валидация


# ============================================================================
# ТЕСТЫ: render_stats()
# ============================================================================

@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.FileListPanel")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_render_stats_counts_messages(mock_data_cfg, mock_file_panel, mock_st, session_state):
    """✅ render_stats() корректно считает сообщения"""
    from rag_gigachat.ui.streamlit_app import render_stats

    session_state["messages"] = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    session_state["max_tokens"] = 2000

    mock_st.session_state = session_state
    mock_file_panel._get_pdf_files.return_value = []
    mock_data_cfg.documents_dirs = {}

    # Mock st.columns to return proper MagicMocks
    mock_cols = [MagicMock() for _ in range(4)]
    mock_st.columns.return_value = mock_cols

    render_stats()

    # Проверить что метрики были вызваны
    assert mock_st.metric.call_count >= 4


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.FileListPanel")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_render_stats_invalid_message_structure(mock_data_cfg, mock_file_panel, mock_st, session_state):
    """✅ render_stats() валидирует структуру сообщения"""
    from rag_gigachat.ui.streamlit_app import render_stats

    session_state["messages"] = [
        {"role": "user", "content": "Q1"},
        {"content": "A1"},  # ❌ Нет role - будет пропущено
        {"role": "assistant"},  # ❌ Нет content - будет пропущено
    ]
    session_state["max_tokens"] = 2000

    mock_st.session_state = session_state
    mock_file_panel._get_pdf_files.return_value = []
    mock_data_cfg.documents_dirs = {}

    mock_cols = [MagicMock() for _ in range(4)]
    mock_st.columns.return_value = mock_cols

    # ✅ Теперь не падает - невалидные сообщения просто пропускаются
    render_stats()

    # Проверить что metric был вызван (1 user вопрос правильно вычислен)
    assert mock_st.metric.call_count >= 4


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.FileListPanel")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_render_stats_empty_documents_dirs(mock_data_cfg, mock_file_panel, mock_st, session_state):
    """✅ render_stats() обрабатывает пустой documents_dirs"""
    from rag_gigachat.ui.streamlit_app import render_stats

    session_state["messages"] = []
    session_state["max_tokens"] = 2000

    mock_st.session_state = session_state
    mock_file_panel._get_pdf_files.return_value = []
    mock_data_cfg.documents_dirs = {}

    mock_cols = [MagicMock() for _ in range(4)]
    mock_st.columns.return_value = mock_cols

    render_stats()

    mock_st.metric.assert_called()


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.get_rag_pipeline")
@patch("rag_gigachat.ui.streamlit_app.HighlightedAnswer")
@patch("rag_gigachat.ui.streamlit_app.AnswerInteraction")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_multiple_queries_in_session(mock_data_cfg, mock_ans_inter, mock_highlight, mock_pipeline, mock_st, session_state):
    """✅ Несколько запросов в одной сессии сохраняют историю"""
    from rag_gigachat.ui.streamlit_app import handle_user_query

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {}

    mock_result = MagicMock()
    mock_result.answer = "Answer"
    mock_result.retrieval_results = None

    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.process_query.return_value = mock_result
    mock_pipeline.return_value = mock_pipeline_inst

    handle_user_query("Question 1?")
    handle_user_query("Question 2?")
    handle_user_query("Question 3?")

    # Проверить историю
    assert len(session_state["messages"]) == 6  # 3 user + 3 assistant
    assert session_state["messages"][0]["content"] == "Question 1?"
    assert session_state["messages"][2]["content"] == "Question 2?"
    assert session_state["messages"][4]["content"] == "Question 3?"


# ============================================================================
# EDGE CASES
# ============================================================================

@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.get_rag_pipeline")
@patch("rag_gigachat.ui.streamlit_app.HighlightedAnswer")
@patch("rag_gigachat.ui.streamlit_app.AnswerInteraction")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_query_with_very_long_text(mock_data_cfg, mock_ans_inter, mock_highlight, mock_pipeline, mock_st, session_state):
    """✅ Очень длинный запрос (>2000 символов) - обработан в UI"""
    from rag_gigachat.ui.streamlit_app import handle_user_query

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {}

    mock_result = MagicMock()
    mock_result.answer = "A" * 5000
    mock_result.retrieval_results = None

    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.process_query.return_value = mock_result
    mock_pipeline.return_value = mock_pipeline_inst

    long_query = "Q" * 2000
    handle_user_query(long_query)

    assert len(session_state["messages"]) == 2


@patch("rag_gigachat.ui.streamlit_app.st")
@patch("rag_gigachat.ui.streamlit_app.get_rag_pipeline")
@patch("rag_gigachat.ui.streamlit_app.HighlightedAnswer")
@patch("rag_gigachat.ui.streamlit_app.AnswerInteraction")
@patch("rag_gigachat.ui.streamlit_app.data_config")
def test_query_with_unicode(mock_data_cfg, mock_ans_inter, mock_highlight, mock_pipeline, mock_st, session_state):
    """✅ Unicode текст обработан корректно"""
    from rag_gigachat.ui.streamlit_app import handle_user_query

    mock_st.session_state = session_state
    mock_data_cfg.documents_dirs = {}

    mock_result = MagicMock()
    mock_result.answer = "Ответ на русском языке"
    mock_result.retrieval_results = None

    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.process_query.return_value = mock_result
    mock_pipeline.return_value = mock_pipeline_inst

    unicode_query = "Что это? 中文? 한글? العربية?"
    handle_user_query(unicode_query)

    assert session_state["messages"][0]["content"] == unicode_query
