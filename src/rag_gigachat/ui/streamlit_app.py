"""
Streamlit UI для RAG GigaChat системы.
Основное приложение с полной интеграцией компонентов и RAGPipeline.
"""

import logging
import os
# Подавляем шумные предупреждения от transformers
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'  # Режим оффлайн для HF Hub
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Оффлайн режим для transformers
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'  # Кэш локальных моделей

import streamlit as st
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Импортируем компоненты
from rag_gigachat.ui.components import (
    ConfigModal,
    FileListPanel,
    DocumentViewer,
    HighlightedAnswer,
    AnswerInteraction
)
from rag_gigachat.config import model_config, data_config, gigachat_config, debug_config
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
            embedding_type="huggingface",
            llm_type="local",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    return st.session_state[pipeline_key]


def load_documents_to_pipeline(pipeline: RAGPipeline, domain_path: Path):
    """Загрузить документы из директории в FAISS индекс

    📌 Использует session_state кэш для предотвращения множественных загрузок
    """
    # ✅ КЭШИРОВАНИЕ: Проверяем, уже ли загружен индекс для этой директории
    cache_key = f"index_loaded_{domain_path}"
    if cache_key in st.session_state and st.session_state[cache_key]:
        logger.debug(f"✅ Индекс уже в памяти: {domain_path} (используем кэш session_state)")
        return True

    try:
        logger.info(f"🔄 ВЫЗОВ load_documents_to_pipeline (domain={domain_path.name})")
        with st.spinner("📚 Загрузка документов в индекс..."):
            logger.info(f"📍 Начало загрузки документов из: {domain_path}")
            logger.info(f"📍 Директория существует: {domain_path.exists()}")

            # Проверить наличие PDF файлов
            pdf_files = list(domain_path.rglob("*.pdf"))
            logger.info(f"📊 Найдено PDF файлов: {len(pdf_files)}")
            if pdf_files:
                logger.debug(f"📋 Список PDF файлов: {[f.name for f in pdf_files[:5]]}{'...' if len(pdf_files) > 5 else ''}")

            if not pdf_files:
                logger.warning(f"⚠️ PDF файлы не найдены в {domain_path}")
                st.warning(f"⚠️ PDF файлы не найдены в {domain_path}")
                return False

            logger.debug(f"🚀 Загружаем {len(pdf_files)} PDF файлов в pipeline...")
            pipeline.load_from_pdf_directory(
                directory=domain_path,
                recursive=True,
                force_reload=False
            )

            logger.info(f"✅ PDF загружены в индекс")
            logger.debug(f"📊 Статус индекса: vector_store_initialized={pipeline.vector_store_initialized}, "
                        f"manager.is_initialized={pipeline.vector_store_manager.is_initialized}")

            # ✅ Сохраняем в session_state, чтобы не загружать заново
            st.session_state[cache_key] = True
            logger.debug(f"💾 Кэш session_state сохранен: {cache_key}")

        st.success("✅ Документы успешно загружены в индекс!")
        logger.info(f"✅ Загрузка завершена успешно")
        return True
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ Ошибка загрузки документов: {error_msg}", exc_info=True)
        logger.debug(f"Traceback: {traceback.format_exc()}")
        st.error(f"❌ Ошибка загрузки документов: {error_msg}")
        return False


def init_session_state():
    """Инициализировать session_state при первом запуске"""
    # 🐛 DEBUG MODE: Выбрать правильное имя модели для отображения
    import os
    env_debug_mode = os.getenv("RAG_DEBUG_MODE", "false").lower() == "true"
    llm_model_display = debug_config.debug_model_name if env_debug_mode else model_config.llm_model_name

    defaults = {
        "show_config_modal": False,
        "show_document_viewer": False,
        "selected_file": None,
        "selected_page": 1,
        "selected_files": [],
        "force_reload_index": False,
        # Модели
        "llm_model": llm_model_display,
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
        # Debug режим
        "debug_query_executed": False,
        "auto_test_question_sent": False,  # 🧪 Флаг для автоматического вопроса
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
        logger.debug(f"Query check: vector_store_initialized={initialized}, manager.is_initialized={mgr_initialized}")

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

        # Контейнер для статуса обработки
        status_container = st.container()
        progress_bar = None
        status_text = None

        def update_progress(stage: str, message: str, progress: float = None):
            """Callback для обновления прогресса в UI"""
            nonlocal progress_bar, status_text
            with status_container:
                if progress_bar is None:
                    status_text = st.empty()
                    progress_bar = st.empty()

                status_text.markdown(f"**Ход выполнения:** {message}")
                if progress is not None:
                    progress_bar.progress(min(int(progress * 100), 99), text=f"{stage.upper()}: {message}")

        # Получить ответ с источниками
        result = pipeline.process_query(
            query,
            k=st.session_state.k_retrieve,
            progress_callback=update_progress,
            llm_model_name=st.session_state.llm_model
        )

        # Обновить финальный прогресс
        with status_container:
            status_text.markdown(f"**Ход выполнения:** ✅ Обработка завершена")
            progress_bar.progress(100, text="ЗАВЕРШЕНО")

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

        # Очистить статус и показать результат
        status_container.empty()

        # Показать время обработки
        st.markdown(f"⏱️ **Время обработки:** {result.generation_time:.1f} сек | 📚 Документов найдено: {len(result.retrieval_results.retrieved_docs) if result.retrieval_results else 0}")

        # Показать ответ с источниками
        st.markdown("---")
        st.subheader("💬 Ответ")
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

    logger.debug(f"🚀 RERUN STREAMLIT #1: Инициализация main()")

    # 🐛 ДИАГНОСТИКА DEBUG-РЕЖИМА: Проверяем и логируем статус
    import os
    env_debug = os.getenv("RAG_DEBUG_MODE", "false").lower() == "true"

    if env_debug or debug_config.debug_mode:
        logger.info(f"✅ DEBUG-РЕЖИМ АКТИВЕН: Используется {debug_config.debug_model_name} (125M параметров)")
        st.info(f"✅ **DEBUG-режим активен**: Используется быстрая модель {debug_config.debug_model_name} (125M параметров, ~3 сек на загрузку)")
    else:
        logger.info(f"📦 PRODUCTION-РЕЖИМ: Используется {model_config.llm_model_name} (500M параметров, высокое качество)")
        logger.info(f"💡 Для включения debug-режима выполните: export RAG_DEBUG_MODE=true")

    # 🧪 ТЕСТОВЫЙ ВОПРОС: Проверяем есть ли вопрос для автоматического отправления
    test_question = debug_config.test_question
    if test_question:
        logger.info(f"🧪 АВТОМАТИЧЕСКИЙ ВОПРОС: '{test_question}'")
        st.info(f"🧪 **Автоматический тест**: Отправляется вопрос '{test_question}'")

    # Инициализация состояния
    init_session_state()

    # Проверка режима debug и тестового вопроса
    is_debug_mode = os.environ.get("RAG_DEBUG", "").lower() == "true"
    debug_query = "Что такое RAG?"

    # 🧪 ТЕСТОВЫЙ ВОПРОС: Может быть установлен через env var или флаг
    test_question = debug_config.test_question or os.getenv("RAG_TEST_QUESTION", "")
    if test_question:
        logger.info(f"🧪 Автоматический вопрос установлен: '{test_question}'")
        # Используем test_question вместо debug_query, если он установлен
        question_to_auto_send = test_question
    else:
        question_to_auto_send = None

    # Авто-загрузка документов если индекс пуст
    try:
        logger.debug(f"🚀 RERUN STREAMLIT #2: get_rag_pipeline()")
        pipeline = get_rag_pipeline(
            embedding_model=st.session_state.embedding_model,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap
        )

        # ✅ КЭШИРОВАНИЕ: Проверяем, нужна ли загрузка
        # Используем session_state флаг для предотвращения множественных загрузок
        auto_load_key = "auto_load_executed"
        if auto_load_key not in st.session_state:
            st.session_state[auto_load_key] = False

        # Загружаем ТОЛЬКО если: 1) индекс пуст И 2) еще не загружали в этой сессии
        if not pipeline.vector_store_manager.is_initialized and not st.session_state[auto_load_key]:
            logger.info(f"📚 Первый запуск: загружаем документы в индекс (выполняется один раз за сессию)")
            st.info("📚 Загрузка документов в индекс (выполняется один раз за сессию)...")

            first_domain = list(data_config.documents_dirs.values())[0]
            logger.debug(f"📁 Первый домен для загрузки: {first_domain.name}")

            if first_domain.exists():
                try:
                    logger.info(f"🔄 Начинаю загрузку PDF из {first_domain}")
                    pipeline.load_from_pdf_directory(
                        directory=first_domain,
                        recursive=True,
                        force_reload=False
                    )
                    st.session_state[auto_load_key] = True  # ✅ Отмечаем, что загрузили
                    logger.info(f"✅ Документы загружены в индекс успешно")
                    st.success("✅ Документы загружены в индекс!")
                except Exception as e:
                    logger.error(f"❌ Ошибка загрузки PDF: {e}", exc_info=True)
                    st.error(f"❌ Ошибка загрузки документов: {e}")
            else:
                logger.error(f"❌ Директория не найдена: {first_domain}")
                st.error(f"❌ Директория не найдена: {first_domain}")
        else:
            logger.debug(f"✅ Индекс уже инициализирован или уже загружали в этой сессии (auto_load={st.session_state[auto_load_key]})")

    except Exception as e:
        logger.error(f"❌ Ошибка инициализации main(): {e}", exc_info=True)
        st.error(f"❌ Ошибка инициализации: {e}")

    # Обработка явной перезагрузки индекса (кнопка в боковой панели)
    if st.session_state.get("force_reload_index", False):
        st.info("⏳ Перезагрузка индекса...")
        try:
            domain_path = data_config.documents_dirs.get(
                st.session_state.get("selected_domain", list(data_config.documents_dirs.keys())[0])
            )

            if domain_path and domain_path.exists():
                pipeline = get_rag_pipeline(
                    embedding_model=st.session_state.embedding_model,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap
                )
                if load_documents_to_pipeline(pipeline, domain_path):
                    st.session_state.force_reload_index = False
                    st.success("✅ Индекс обновлен!")
            else:
                st.error(f"❌ Директория не найдена: {domain_path}")
                st.session_state.force_reload_index = False
        except Exception as e:
            logger.error(f"Ошибка перезагрузки: {e}", exc_info=True)
            st.error(f"❌ Ошибка: {e}")
            st.session_state.force_reload_index = False

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

    # 🧪 АВТОМАТИЧЕСКАЯ ОТПРАВКА ВОПРОСА: После загрузки документов
    # Используем session_state флаг чтобы отправить только один раз за сессию
    if question_to_auto_send and not st.session_state.get("auto_test_question_sent", False):
        # Проверяем, что индекс инициализирован (документы загружены)
        try:
            pipeline = get_rag_pipeline(
                embedding_model=st.session_state.embedding_model,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            if pipeline.vector_store_manager.is_initialized:
                st.session_state.auto_test_question_sent = True
                logger.info(f"🧪 ОТПРАВКА АВТОМАТИЧЕСКОГО ВОПРОСА: '{question_to_auto_send}'")
                # Задержка для стабилизации интерфейса
                with st.spinner(f"🧪 Автоматическая отправка вопроса: '{question_to_auto_send}'"):
                    import time
                    time.sleep(1)
                    try:
                        handle_user_query(question_to_auto_send)
                        logger.info(f"✅ Автоматический вопрос обработан успешно")
                    except Exception as e:
                        logger.error(f"❌ Ошибка при автоматической отправке: {e}", exc_info=True)
                        st.error(f"❌ Ошибка при отправке вопроса: {e}")
        except Exception as e:
            logger.error(f"❌ Ошибка при инициализации для автоответа: {e}", exc_info=True)

    # DEBUG MODE: Автоматически выполнить тестовый запрос (старая логика для совместимости)
    elif is_debug_mode and not st.session_state.debug_query_executed:
        st.session_state.debug_query_executed = True
        # Задержка для загрузки документов
        with st.spinner("🔄 DEBUG: Автоматический запрос..."):
            import time
            time.sleep(2)
            try:
                handle_user_query(debug_query)
            except Exception as e:
                st.error(f"❌ DEBUG запрос ошибка: {e}")
                logger.error(f"Debug query error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
