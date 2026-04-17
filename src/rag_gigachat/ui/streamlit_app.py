"""
Streamlit UI для RAG GigaChat системы.
Основное приложение с полной интеграцией компонентов и RAGPipeline.
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


def get_rag_pipeline(embedding_model: str, chunk_size: int, chunk_overlap: int) -> RAGPipeline:
    """Получить или создать RAGPipeline (хранится в session_state)"""
    # ✅ Валидация параметров
    if not embedding_model or not isinstance(embedding_model, str):
        raise ValueError(f"❌ embedding_model должна быть непустой строкой, получено: {embedding_model}")

    if chunk_size <= 0:
        raise ValueError(f"❌ chunk_size должен быть > 0, получено: {chunk_size}")

    if chunk_overlap < 0:
        raise ValueError(f"❌ chunk_overlap не может быть отрицательным, получено: {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(f"❌ chunk_overlap ({chunk_overlap}) должен быть < chunk_size ({chunk_size})")

    # Проверить, есть ли pipeline в session_state
    pipeline_key = f"pipeline_{embedding_model}_{chunk_size}_{chunk_overlap}"
    if pipeline_key not in st.session_state:
        st.session_state[pipeline_key] = RAGPipeline(
            embedding_model=embedding_model,
            llm_type="gigachat",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    return st.session_state[pipeline_key]


def load_documents_to_pipeline(pipeline: RAGPipeline, domain_path: Path):
    """Загрузить документы из директории в FAISS индекс"""
    try:
        with st.spinner("📚 Загрузка документов в индекс..."):
            print(f"🔍 DEBUG: Начало загрузки из {domain_path}")
            print(f"🔍 DEBUG: Директория существует: {domain_path.exists()}")

            # Проверить наличие PDF файлов
            pdf_files = list(domain_path.rglob("*.pdf"))
            print(f"🔍 DEBUG: Найдено PDF файлов: {len(pdf_files)}")

            if not pdf_files:
                st.warning(f"⚠️ PDF файлы не найдены в {domain_path}")
                return False

            pipeline.load_from_pdf_directory(
                directory=domain_path,
                recursive=True,
                force_reload=True
            )

            print(f"🔍 DEBUG: vector_store_initialized = {pipeline.vector_store_initialized}")
            print(f"🔍 DEBUG: vector_store_manager.is_initialized = {pipeline.vector_store_manager.is_initialized}")

        st.success("✅ Документы успешно загружены в индекс!")
        return True
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"❌ DEBUG: Ошибка: {error_msg}")
        traceback.print_exc()
        st.error(f"❌ Ошибка загрузки документов: {error_msg}")
        return False


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

    # Инструкции при первом запуске
    if not st.session_state.messages:
        with st.info("💡 **Как начать:**"):
            st.markdown("""
            1. **Выберите документы** → Справа в боковой панели нажмите на файлы
            2. **Задайте вопрос** → Введите вопрос о содержимом документов
            3. **Настройки** (опционально) → ⚙️ Расширенные настройки для параметров поиска
            """)

    # Основное содержимое
    with st.container():
        # Область чата
        st.subheader("💬 Диалог")

        # История сообщений
        messages_container = st.container(height=400, border=True)

        with messages_container:
            for msg in st.session_state.get("messages", []):
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])

        # Поле ввода для вопроса
        st.markdown("---")
        col_input, col_send = st.columns([5, 1], gap="small")

        with col_input:
            user_input = st.text_area(
                "Ваш вопрос",
                placeholder="Спросите о чём-нибудь из документов...",
                height=80,
                key="user_input",
                max_chars=2000
            )
            st.caption(f"📝 {len(user_input)}/2000")

        def clear_input():
            """Callback для очистки поля ввода"""
            handle_user_query(st.session_state.user_input)
            st.session_state.user_input = ""

        with col_send:
            st.write("")  # Выравнивание по высоте
            st.write("")
            st.button(
                "🚀 Отправить",
                use_container_width=True,
                on_click=clear_input,
                disabled=not user_input.strip()
            )


def handle_user_query(query: str):
    """
    Обработать запрос пользователя

    Args:
        query: Текст запроса
    """
    if not query or not query.strip():
        st.error("❌ Пожалуйста, введите вопрос")
        return

    # Интеграция с RAGPipeline
    try:
        # Получить pipeline
        pipeline = get_rag_pipeline(
            embedding_model=st.session_state.embedding_model,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap
        )

        # Проверить, загружены ли документы
        initialized = pipeline.vector_store_initialized
        mgr_initialized = pipeline.vector_store_manager.is_initialized
        print(f"🔍 DEBUG query: vector_store_initialized={initialized}, manager.is_initialized={mgr_initialized}")

        if not initialized or not mgr_initialized:
            st.error(
                "❌ FAISS индекс не инициализирован.\n\n"
                "**Решение:**\n"
                "1. Откройте боковую панель (📁 Документы)\n"
                "2. Нажмите кнопку '🔄 Обновить индекс'\n"
                "3. Дождитесь сообщения '✅ Документы успешно загружены'\n"
                "4. Задайте вопрос снова"
            )
            return
        with st.spinner("🔄 Обработка запроса..."):
            # Получить ответ с источниками
            result = pipeline.process_query(
                query,
                k=st.session_state.k_retrieve
            )

        # ✅ Добавить сообщения ТОЛЬКО после успешного получения ответа
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        # Добавить ответ в историю
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "documents": result.retrieval_results.retrieved_docs if result.retrieval_results else []
        })

        # Показать ответ с источниками
        st.markdown("---")
        HighlightedAnswer.show(
            answer=result.answer,
            retrieved_docs=result.retrieval_results.retrieved_docs if result.retrieval_results else [],
            documents_dirs=data_config.documents_dirs,
            show_sources=True
        )

        # Интерактивные кнопки
        st.markdown("---")
        AnswerInteraction.show_actions(result.answer, answer_id=f"answer_{len(st.session_state.messages)}")

    except ValueError as e:
        # ✅ Ошибки валидации - показать пользователю
        st.error(f"⚠️ Ошибка параметров: {str(e)}")
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
        file_name = st.session_state.selected_file  # Имя без расширения

        # ✅ Валидация selected_file
        if not isinstance(file_name, str) or not file_name.strip():
            st.error("❌ Ошибка: название файла некорректно")
            return

        page = st.session_state.get("selected_page", 1)

        # ✅ Валидация номера страницы
        if not isinstance(page, int) or page < 1:
            st.error(f"❌ Ошибка: номер страницы должен быть ≥ 1, получено: {page}")
            return

        # Поиск файла во всех доменах
        file_path = None
        for domain_path in data_config.documents_dirs.values():
            candidate = domain_path / f"{file_name}.pdf"
            if candidate.exists():
                file_path = candidate
                break

        if file_path and file_path.exists():
            st.subheader("📄 Просмотр документа")
            DocumentViewer.show(str(file_path), page)

            # Кнопка закрытия
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("✕ Закрыть", use_container_width=True):
                    st.session_state.show_document_viewer = False
                    st.rerun()
        else:
            st.error(f"❌ Файл '{file_name}.pdf' не найден в документах")


def render_stats():
    """Показать статистику в нижней части экрана"""

    st.markdown("---")
    st.markdown("### 📊 Статистика сессии")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # ✅ Безопасный подсчёт с валидацией структуры
        user_count = len([
            m for m in st.session_state.messages
            if isinstance(m, dict) and m.get("role") == "user"
        ])
        st.metric("Вопросов", user_count)

    with col2:
        # ✅ Безопасный подсчёт с валидацией структуры
        assistant_count = len([
            m for m in st.session_state.messages
            if isinstance(m, dict) and m.get("role") == "assistant"
        ])
        st.metric("Ответов", assistant_count)

    with col3:
        # Количество PDF во всех доменах
        total_docs = 0
        for domain_path in data_config.documents_dirs.values():
            total_docs += len(FileListPanel._get_pdf_files(domain_path, ""))
        st.metric("Документов загружено", total_docs)

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

    # Авто-загрузка документов если индекс пуст (ПЕРЕД DEBUG чтобы видеть процесс)
    if not st.session_state.get("_docs_auto_loaded", False):
        st.write("🔍 АВТОЗАГРУЗКА: Проверка индекса...")
        try:
            pipeline = get_rag_pipeline(
                embedding_model=st.session_state.embedding_model,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            st.write(f"АВТОЗАГРУЗКА: Pipeline.vector_store_initialized: {pipeline.vector_store_initialized}")
            st.write(f"АВТОЗАГРУЗКА: Manager.is_initialized: {pipeline.vector_store_manager.is_initialized}")

            if not pipeline.vector_store_manager.is_initialized:
                st.info("📚 АВТОЗАГРУЗКА: Загрузка документов...")
                # Загружаем из первой доступной domain директории
                first_domain = list(data_config.documents_dirs.values())[0]
                st.write(f"АВТОЗАГРУЗКА: Загрузка из {first_domain}")
                st.write(f"АВТОЗАГРУЗКА: Dir exists: {first_domain.exists()}")

                try:
                    pipeline.load_from_pdf_directory(
                        directory=first_domain,
                        recursive=True,
                        force_reload=True
                    )
                    st.write(f"АВТОЗАГРУЗКА: load_from_pdf_directory завершена")
                except Exception as e:
                    st.error(f"АВТОЗАГРУЗКА: Exception in load_from_pdf_directory: {e}")
                    import traceback
                    st.write(traceback.format_exc())

                st.write(f"АВТОЗАГРУЗКА: vector_store_initialized={pipeline.vector_store_initialized}")
                st.write(f"АВТОЗАГРУЗКА: Manager.is_initialized={pipeline.vector_store_manager.is_initialized}")
                st.write(f"АВТОЗАГРУЗКА: Manager.vector_store={pipeline.vector_store_manager.vector_store is not None}")
                st.session_state._docs_auto_loaded = True
                st.success("✅ АВТОЗАГРУЗКА: Документы загружены!")
            else:
                st.write("✅ АВТОЗАГРУЗКА: Индекс уже инициализирован")
                st.session_state._docs_auto_loaded = True
        except Exception as e:
            import traceback
            st.error(f"❌ АВТОЗАГРУЗКА ОШИБКА: {e}")
            st.write(traceback.format_exc())
            st.session_state._docs_auto_loaded = True

    # Показать финальный статус индекса
    with st.expander("🔧 DEBUG: Финальный статус", expanded=False):
        try:
            pipeline = get_rag_pipeline(
                embedding_model=st.session_state.embedding_model,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            st.write(f"✅ Pipeline создана")
            st.write(f"Pipeline.vector_store_initialized: {pipeline.vector_store_initialized}")
            st.write(f"Manager.is_initialized: {pipeline.vector_store_manager.is_initialized}")
            st.write(f"_docs_auto_loaded flag: {st.session_state.get('_docs_auto_loaded', 'Not set')}")
        except Exception as e:
            st.error(f"Ошибка при проверке: {e}")

    # Обработка загрузки документов
    if st.session_state.get("force_reload_index", False):
        st.info("⏳ Загрузка документов начата...")
        try:
            domain_path = data_config.documents_dirs.get(st.session_state.get("selected_domain", list(data_config.documents_dirs.keys())[0]))
            st.write(f"📂 Директория: {domain_path}")

            if domain_path and domain_path.exists():
                pipeline = get_rag_pipeline(
                    embedding_model=st.session_state.embedding_model,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap
                )
                st.write(f"🔧 Pipeline создана")

                if load_documents_to_pipeline(pipeline, domain_path):
                    st.session_state.force_reload_index = False
                    st.write(f"✅ Флаг force_reload_index = {st.session_state.force_reload_index}")
                else:
                    st.write(f"❌ Загрузка вернула False")
            else:
                st.error(f"❌ Директория не существует: {domain_path}")
                st.session_state.force_reload_index = False
        except Exception as e:
            import traceback
            st.error(f"❌ Ошибка при загрузке документов: {e}")
            st.write(traceback.format_exc())
            st.session_state.force_reload_index = False

    # Авто-загрузка документов если индекс пуст
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
