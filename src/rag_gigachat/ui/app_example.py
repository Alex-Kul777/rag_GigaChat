"""
app_example.py - Пример интеграции компонентов в Streamlit приложение
Показывает, как использовать ConfigModal, FileListPanel, DocumentViewer, HighlightedAnswer
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Optional

# Импортируем компоненты
from rag_gigachat.ui.components import (
    ConfigModal,
    FileListPanel,
    DocumentViewer,
    HighlightedAnswer,
    AnswerInteraction
)
from rag_gigachat.config import model_config, data_config, gigachat_config
from rag_gigachat.core.rag_pipeline import RAGPipeline


def init_session_state():
    """Инициализировать session_state при первом запуске"""
    defaults = {
        "show_config_modal": False,
        "show_document_viewer": False,
        "selected_file": None,
        "selected_page": 1,
        "selected_files": [],
        "force_reload_index": False,
        # Модели
        "llm_model": model_config.llm_model_name,
        "embedding_model": model_config.embedding_model_name,
        "max_tokens": model_config.max_new_tokens,
        "temperature": model_config.temperature,
        # Поиск
        "k_retrieve": model_config.default_k_retrieve,
        "max_context": model_config.max_context_length,
        "retrieval_type": "hybrid",
        # Чанкирование
        "chunk_size": data_config.chunk_size,
        "chunk_overlap": data_config.chunk_overlap,
        # GigaChat
        "top_p": model_config.top_p,
        "repeat_penalty": model_config.repetition_penalty,
        "ocr_enabled": data_config.ocr_enabled,
        # Сессия чата
        "messages": [],
        "chat_history": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_main_interface():
    """Рендерить главное окно чата"""

    # Заголовок
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #1E88E5;">🤖 RAG Chat</h1>
        <p style="color: #666; font-size: 1.1rem;">Интеллектуальный поиск в документах с GigaChat</p>
    </div>
    """, unsafe_allow_html=True)

    # Основное содержимое в две колоны (в зависимости от выбора режима)
    col_main, col_right = st.columns([1, 0], gap="large")

    with col_main:
        # Область чата
        st.subheader("💬 Диалог")

        # История сообщений
        messages_container = st.container(height=400, border=True)

        with messages_container:
            for msg in st.session_state.get("messages", []):
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant").write(msg["content"])

        # Поле ввода для вопроса
        st.markdown("---")
        col_input, col_send = st.columns([5, 1], gap="small")

        with col_input:
            user_input = st.text_area(
                "Ваш вопрос",
                placeholder="Спросите о чём-нибудь из документов...",
                height=80,
                key="user_input"
            )

        with col_send:
            st.write("")  # Выравнивание по высоте
            st.write("")
            if st.button("🚀 Отправить", use_container_width=True):
                if user_input.strip():
                    # Обработать запрос
                    handle_user_query(user_input)


def handle_user_query(query: str):
    """
    Обработать запрос пользователя

    Args:
        query: Текст запроса
    """

    # Добавить сообщение пользователя
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Здесь должна быть интеграция с RAGPipeline
    try:
        # Инициализировать pipeline с параметрами из session_state
        pipeline = RAGPipeline(
            llm_model_name=st.session_state.llm_model,
            embedding_model_name=st.session_state.embedding_model,
            **{"default_k_retrieve": st.session_state.k_retrieve}
        )

        # Получить ответ с источниками
        answer, retrieved_docs = pipeline.query(
            query,
            top_k=st.session_state.k_retrieve,
            retrieval_type=st.session_state.retrieval_type
        )

        # Добавить ответ в историю
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "documents": retrieved_docs
        })

        # Показать ответ с источниками
        st.markdown("---")
        HighlightedAnswer.show(
            answer=answer,
            retrieved_docs=retrieved_docs,
            documents_dirs=data_config.documents_dirs,
            show_sources=True
        )

        # Интерактивные кнопки
        st.markdown("---")
        AnswerInteraction.show_actions(answer, answer_id=f"answer_{len(st.session_state.messages)}")

    except Exception as e:
        st.error(f"❌ Ошибка при обработке запроса: {e}")


def render_sidebar():
    """Рендерить боковую панель"""

    with st.sidebar:
        # Логотип/название
        st.markdown("""
        ### 📚 RAG GigaChat
        Система поиска по документам
        """)

        st.divider()

        # Кнопка расширенных настроек
        ConfigModal.show()

        st.divider()

        # Панель файлов
        FileListPanel.show(data_config.documents_dirs)


def render_document_viewer():
    """Показать просмотр документа в модальном окне"""

    if st.session_state.get("show_document_viewer") and st.session_state.get("selected_file"):
        # Найти полный путь к файлу
        file_name = st.session_state.selected_file
        page = st.session_state.get("selected_page", 1)

        # Поиск файла во всех доменах
        file_path = None
        for domain_path in data_config.documents_dirs.values():
            candidate = domain_path / f"{file_name}.pdf"
            if candidate.exists():
                file_path = str(candidate)
                break

            # Или прямой путь
            candidate = Path(file_name)
            if candidate.exists():
                file_path = str(candidate)
                break

        if file_path:
            # Использовать dialog для модального окна (Streamlit >= 1.30)
            if st.session_state.get("show_document_viewer"):
                with st.dialog("Просмотр документа", width="large"):
                    DocumentViewer.show(file_path, page)

                    # Кнопка закрытия
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        if st.button("✕ Закрыть", use_container_width=True):
                            st.session_state.show_document_viewer = False
                            st.rerun()


def render_stats():
    """Показать статистику в нижней части экрана"""

    st.markdown("---")
    st.markdown("### 📊 Статистика сессии")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Вопросов", len([m for m in st.session_state.messages if m["role"] == "user"]))

    with col2:
        st.metric("Ответов", len([m for m in st.session_state.messages if m["role"] == "assistant"]))

    with col3:
        st.metric("Документов загружено", len(FileListPanel._get_pdf_files(
            data_config.documents_dirs.get("UAV", Path(".")), ""
        )))

    with col4:
        tokens_per_query = st.session_state.max_tokens
        st.metric("Макс токен/запрос", tokens_per_query)


def main():
    """Главная функция приложения"""

    # Конфигурация страницы
    st.set_page_config(
        page_title="RAG Chat with PDF",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Инициализация состояния
    init_session_state()

    # Кастомные стили
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #1E88E5;
            margin-bottom: 2rem;
        }
        .stats-container {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
        }
        .source-container {
            border-left: 4px solid #1E88E5;
            padding-left: 10px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Боковая панель
    render_sidebar()

    # Основной интерфейс
    render_main_interface()

    # Просмотр документа
    render_document_viewer()

    # Статистика
    render_stats()


if __name__ == "__main__":
    main()
