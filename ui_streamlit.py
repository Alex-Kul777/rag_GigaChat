"""
ui_streamlit.py - Переиспользуемые UI компоненты для Streamlit
Интегрирован с RAGPipeline и централизованной конфигурацией
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
pdf_dir = data_config.documents_dirs["ai"]

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

# Sidebar
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Параметры RAG
    st.markdown("### ⚙️ Параметры RAG")
    
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
    
    # Кнопка переинициализации
    st.markdown("---")
    if st.button("🔄 Переинициализировать RAG", type="primary", use_container_width=True):
        st.session_state.rag_initialized = False
        st.session_state.pipeline = None
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Как пользоваться:")
    st.markdown("""
    1. Дождитесь инициализации системы
    2. Задавайте вопросы о документах
    3. Система отвечает на русском языке
    """)
    
    st.markdown("---")
    st.markdown("### 📂 Источники документов")

    if pdf_dir.exists():
        files = list(pdf_dir.glob("*.pdf"))
        st.markdown(f"**Директория:** `{pdf_dir}`")
        st.markdown(f"**Документов:** {len(files)}")
        if files:
            with st.expander("📄 Список документов"):
                for f in files[:5]:
                    st.text(f.name)
                if len(files) > 5:
                    st.text(f"... и {len(files)-5} других")
    else:
        st.warning(f"Директория не найдена: {pdf_dir}")
    
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