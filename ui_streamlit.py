"""
ui_streamlit.py - Переиспользуемые UI компоненты для Streamlit
Интегрирован с RAGPipeline и централизованной конфигурацией

Что такое схема утка?
как рассчитать силы действующие на самолёт схемы утка?
Что такое точка фокуса и как её рассчиать?
Какая формула лобового сопротивления крыла?
НА какие виды делятся беспилотные аппараты в зависимости от веса или массы?
How much classifications by construction of UAV?
"""

import streamlit as st
import os
import warnings
from pathlib import Path

# Подавляем предупреждения
warnings.filterwarnings("ignore", category=UserWarning)

# Импортируем из наших модулей
from config import model_config, data_config, gigachat_config
from rag_core import RAGPipeline
from models import RetrievalType

#pdf_dir = data_config.documents_dirs["debug"]
pdf_dir = data_config.documents_dirs["UAV"]

# Page config
st.set_page_config(
    page_title="Chat with PDF", 
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .stats-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📄 Общайся с pdf документами</p>', unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

if "selected_pdf_dir" not in st.session_state:
    # По умолчанию используем UAV, если не выбран другой
    st.session_state.selected_pdf_dir = data_config.documents_dirs["UAV"]

# Sidebar
with st.sidebar:
    st.header("⚙️ Настройки")
    
    st.markdown("### 📂 Выбор документов")

    # Получаем список доступных доменов из конфига
    available_domains = list(data_config.documents_dirs.keys())
    
    # Добавляем опцию "custom" если нужно
    domain_options = available_domains + ["custom"]
    
    # Создаем словарь для отображения понятных названий
    domain_labels = {
        "UAV": "🛸 Беспилотные летательные аппараты (БПЛА)",
        "debug": "🐛 Debug документы",
        "ai": "🤖 Искусственный интеллект",
        "custom": "📁 Своя директория"
    }
    
    # Функция форматирования с проверкой наличия ключа
    def format_domain_option(option):
        if option in domain_labels:
            return domain_labels[option]
        return f"📂 {option.capitalize()}"
    
    # Выбор типа документов с UAV по умолчанию
    doc_type = st.selectbox(
        "Источник документов",
        options=domain_options,
        format_func=format_domain_option,
        index=domain_options.index("UAV"),  # 👈 UAV выбран по умолчанию
        help="Выберите набор документов для работы"
    )

    # Путь к директории в зависимости от выбора
    if doc_type == "custom":
        custom_path = st.text_input(
            "Путь к директории с PDF",
            value=str(Path("./data/custom")),
            help="Укажите полный или относительный путь к папке с PDF файлами"
        )
        selected_pdf_dir = Path(custom_path)
    else:
        selected_pdf_dir = data_config.documents_dirs[doc_type]

    # Показываем информацию о выбранной директории
    if selected_pdf_dir.exists():
        pdf_files = list(selected_pdf_dir.glob("*.pdf"))
        st.success(f"✅ Найдено {len(pdf_files)} PDF файлов")
        if pdf_files:
            with st.expander("📄 Показать файлы"):
                for f in pdf_files[:5]:
                    st.text(f.name)
                if len(pdf_files) > 5:
                    st.text(f"... и {len(pdf_files)-5} других")
    else:
        st.error(f"❌ Директория не найдена: {selected_pdf_dir}")

    # Кнопка применения новой директории
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Применить и перезагрузить", type="primary", use_container_width=True):
            st.session_state.selected_pdf_dir = selected_pdf_dir
            st.session_state.rag_initialized = False
            st.session_state.pipeline = None
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("🗑️ Очистить чат", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=model_config.temperature,
        step=0.05,
        help="Higher values make output more creative, lower values more deterministic"
    )
    
    k_retrieve = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=model_config.default_k_retrieve,
        help="How many relevant chunks to use for answering"
    )
    
    # Обновляем конфиг
    model_config.temperature = temperature
    model_config.default_k_retrieve = k_retrieve
    
    st.markdown("### 🔧 Управление")
    col_reset, col_clear = st.columns(2)
    with col_reset:
        if st.button("🔄 Сбросить систему", type="secondary", use_container_width=True):
            st.session_state.rag_initialized = False
            st.session_state.pipeline = None
            st.session_state.messages = []
            if "selected_pdf_dir" in st.session_state:
                del st.session_state.selected_pdf_dir
            st.rerun()

    with col_clear:
        if st.button("🗑️ Очистить историю", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


    if not st.session_state.rag_initialized:
        with st.spinner("🔄 Инициализация RAG системы... Это может занять минуту."):
            try:
                # Проверка API ключа (оставляем как есть)
                if not gigachat_config.api_key:
                    st.error("❌ GigaChat API ключ не найден в .env файле")
                    st.stop()

                # Создаем RAGPipeline (оставляем как есть)
                st.session_state.pipeline = RAGPipeline(
                    chunk_size=data_config.chunk_size,
                    chunk_overlap=data_config.chunk_overlap,
                    embedding_model=model_config.embedding_model_name,
                    embedding_type="gigachat",
                    llm_type="gigachat"
                )

                # ИСПОЛЬЗУЕМ ВЫБРАННУЮ ДИРЕКТОРИЮ
                pdf_dir_to_load = st.session_state.selected_pdf_dir

                if pdf_dir_to_load.exists():
                    st.info(f"📁 Загрузка документов из: {pdf_dir_to_load}")
                    st.session_state.pipeline.load_from_pdf_directory_with_metadata(
                        pdf_dir_to_load, 
                        recursive=True, 
                        force_reload=data_config.force_reload
                    )
                    st.success(f"✅ Загружены документы из {pdf_dir_to_load}")
                else:
                    st.warning(f"⚠️ Директория не найдена: {pdf_dir_to_load}")
                    st.info("Загружаю пример документов для тестирования...")
                    st.session_state.pipeline.load_from_sample_corpus()
                    st.success("✅ Загружены примеры документов")

                st.session_state.rag_initialized = True
                st.success("✅ RAG система готова к работе!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Ошибка инициализации RAGPipeline: {str(e)}")
                import traceback
                traceback.print_exc()
                st.stop()





    
    st.markdown("---")
    st.markdown("### 📚 Как пользоваться:")
    st.markdown("""
    1. Дождитесь инициализации системы
    2. Задавайте вопросы о документах
    3. Система отвечает на русском языке
    """)
    
    st.markdown("---")
    st.markdown("### 📂 Текущие документы")

    current_dir = st.session_state.get("selected_pdf_dir", data_config.documents_dirs["UAV"])
    if current_dir.exists():
        files = list(current_dir.glob("*.pdf"))
        st.markdown(f"**Директория:** `{current_dir}`")
        st.markdown(f"**Документов:** {len(files)}")
        if files:
            with st.expander("📄 Список документов"):
                for f in files[:5]:
                    st.text(f.name)
                if len(files) > 5:
                    st.text(f"... и {len(files)-5} других")
    else:
        st.warning(f"Директория не найдена: {current_dir}")

    st.markdown("---")
    st.markdown("### 🛠️ Built with:")
    st.markdown("""
    • **LangChain** - RAG Framework
    • **GigaChat** - LLM & Embeddings
    • **FAISS** - Vector Search
    • **Streamlit** - UI Framework
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Разработка")
    st.markdown("**Куликов Алексей**")

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Инициализация RAG системы
    if not st.session_state.rag_initialized:
        with st.spinner("🔄 Инициализация RAG системы... Это может занять минуту."):
            try:
                # Проверка наличия API ключа GigaChat
                if not gigachat_config.api_key:
                    st.error("❌ GigaChat API ключ не найден в .env файле")
                    st.info("""
                    **Как настроить:**
                    1. Создайте файл `.env` в корне проекта
                    2. Добавьте строку: `GIGACHAT_API_KEY=ваш_ключ`
                    3. Перезапустите приложение
                    """)
                    st.stop()
                
                # Создаем RAGPipeline
                st.session_state.pipeline = RAGPipeline(
                    chunk_size=data_config.chunk_size,
                    chunk_overlap=data_config.chunk_overlap,
                    embedding_model=model_config.embedding_model_name,
                    embedding_type="gigachat",
                    llm_type="gigachat"
                )
                
                # Загрузка документов
                #pdf_dir = Path("data/domain_2_Debug/books")
                if pdf_dir.exists():
                    st.info(f"📁 Загрузка документов из: {pdf_dir}")
                    st.session_state.pipeline.load_from_pdf_directory_with_metadata(
                        pdf_dir, 
                        recursive=True, 
                        force_reload=data_config.force_reload
                    )
                    st.success(f"✅ Загружены документы из {pdf_dir}")
                else:
                    st.warning(f"⚠️ Директория не найдена: {pdf_dir}")
                    st.info("Загружаю пример документов для тестирования...")
                    st.session_state.pipeline.load_from_sample_corpus()
                    st.success("✅ Загружены примеры документов")
                
                st.session_state.rag_initialized = True
                st.success("✅ RAG система готова к работе!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Ошибка инициализации RAGPipeline: {str(e)}")
                import traceback
                traceback.print_exc()
                st.stop()


# Chat interface
st.markdown("---")
st.markdown("### 💬 Поговори о загруженном файле")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Clear chat button
col_clear, col_space = st.columns([1, 5])
with col_clear:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# User input
if question := st.chat_input("Спроси о документе..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Используем RAGPipeline для обработки запроса
                if st.session_state.pipeline and st.session_state.rag_initialized:
                    # Обновляем параметры генерации
                    model_config.temperature = temperature
                    model_config.max_new_tokens = 500
                    
                    # Обрабатываем запрос
                    result = st.session_state.pipeline.process_query(question, k=k_retrieve)
                    answer = result.answer
                    
                    # Показываем источники
                    if result.retrieval_results.retrieved_docs:
                        with st.expander("📚 Источники"):
                            for i, doc in enumerate(result.retrieval_results.retrieved_docs[:3], 1):
                                source = doc.get('doc_id', 'unknown')
                                score = doc.get('score', 0)
                                text = doc.get('text', '')
                                st.markdown(f"**Источник {i}:** `{source}` (score: {score:.3f})")
                                st.text(text[:200] + "..." if len(text) > 200 else text)
                                st.markdown("---")
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("RAG система не инициализирована")
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>💡 Powered by RAG (Retrieval-Augmented Generation) with GigaChat</p>",
    unsafe_allow_html=True
)