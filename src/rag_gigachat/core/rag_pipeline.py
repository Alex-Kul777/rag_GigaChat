"""
rag_core.py - Facade: RAG пайплайн (LangChain + LangGraph).

Классы VectorStoreManager и LLMManager вынесены в отдельные модули:
  vector_store.py — VectorStoreManager
  llm_manager.py  — LLMManager
Re-экспорт сохранён для обратной совместимости.
"""
import logging
import re
import time
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Optional, Any, TypedDict
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

try:
    from langchain_gigachat.chat_models import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False

from rag_gigachat.config import (
    model_config, data_config, vectorstore_config,
    experiment_config, logging_config, gigachat_config
)

from rag_gigachat.models import RetrievalResult, GenerationResult, RetrievalType
from rag_gigachat.data.data_loader import CorpusLoader, DocumentLoader, TextSplitter
from rag_gigachat.token_counter import TokenCounter
from rag_gigachat.utils.text_utils import filter_documents_by_token_count
from rag_gigachat.core.vector_store import VectorStoreManager       # noqa: F401 (re-export)
from rag_gigachat.core.llm_manager import LLMManager                # noqa: F401 (re-export)
from rag_gigachat.core.retriever import BaseRetriever, DenseRetriever, make_retriever  # noqa: F401
from rag_gigachat.utils.event_log import emit

# Conditional import for process mining (BKL-002)
try:
    from rag_gigachat.utils.event_log import emit
    EMIT_AVAILABLE = True
except ImportError:
    EMIT_AVAILABLE = False
    def emit(*args, **kwargs):
        from contextlib import contextmanager
        @contextmanager
        def noop(*a, **kw):
            yield
        return noop()

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, logging_config.log_level),
    format=logging_config.log_format,
    datefmt=logging_config.log_date_format
)
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, logging_config.log_level))


def format_answer(text: str) -> str:
    """Форматирование ответа LLM для компактного отображения.

    - Множественные пробелы → один пробел
    - Каждое предложение (после точки) → новая строка
    - Без пустых строк
    - Левое выравнивание
    """
    if not text:
        return ""

    # Удаляем множественные пробелы/переносы
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Точка + пробелы + (заглавная буква или цифра) → новая строка
    text = re.sub(r'\.\s+(?=[А-ЯЁ0-9])', '.\n', text)

    # Убираем пустые строки
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)


class RAGState(TypedDict):
    """Состояние для LangGraph"""
    question: str
    context: List[Document]
    answer: str
    metadata: Dict[str, Any]

class RAGPipeline:
    """
    Основной RAG пайплайн с использованием FAISS и LangGraph
    Поддерживает разные модели (HuggingFace, GigaChat, OpenAI)
    """
    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 embedding_model: str = None,
                 embedding_type: str = "huggingface",
                 llm_type: str = "gigachat",
                 retrieval_type: RetrievalType = RetrievalType.DENSE,
                 vector_store_manager: Optional["VectorStoreManager"] = None,
                 llm_manager: Optional["LLMManager"] = None,
                 token_counter: Optional[TokenCounter] = None,
                 retriever: Optional["BaseRetriever"] = None):
        """
        Инициализация RAG пайплайна

        Args:
            chunk_size: Размер чанка для разделения документов
            chunk_overlap: Перекрытие между чанками
            embedding_model: Модель эмбеддингов
            embedding_type: Тип эмбеддингов ("huggingface", "gigachat")
            llm_type: Тип LLM ("local", "gigachat", "openai")
            retrieval_type: Тип поиска (DENSE/SPARSE/HYBRID)
            vector_store_manager: Готовый экземпляр VectorStoreManager (DI)
            llm_manager: Готовый экземпляр LLMManager (DI)
            token_counter: Готовый экземпляр TokenCounter (DI)
            retriever: Готовая стратегия поиска (DI, переопределяет retrieval_type)
        """
        chunk_size = chunk_size or data_config.chunk_size
        chunk_overlap = chunk_overlap or data_config.chunk_overlap

        self.vector_store_manager = vector_store_manager or VectorStoreManager(
            embedding_model=embedding_model or model_config.embedding_model_name,
            embedding_type=embedding_type,
            persist_dir=vectorstore_config.persist_dir
        )
        # Auto-detect если GigaChat недоступен, fallback на local
        _llm_type = llm_type
        if llm_type == "gigachat" and not gigachat_config.api_key:
            logger.warning("⚠️ GigaChat API ключ не найден, переключаюсь на локальную модель")
            _llm_type = "local"

        self.llm_manager = llm_manager or LLMManager(
            model_name=model_config.llm_model_name if _llm_type == "local" else None,
            model_type=_llm_type
        )

        # Используем загрузчик из data_loader
        self.corpus_loader = CorpusLoader(data_dir=data_config.corpus_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.vector_store_initialized = False
        self.graph = None
        self.prompt = None
        self.documents_metadata = {}

        self.llm = None
        self.token_counter = token_counter or TokenCounter()

        # Strategy pattern: инициализируем ретривер
        self.retriever = retriever or make_retriever(
            retrieval_type, self.vector_store_manager
        )

        self.gigachat_client = None  # Для хранения клиента GigaChat

        logger.info(
            f"RAGPipeline инициализирован. chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}, llm_type={llm_type}, "
            f"embedding_type={embedding_type}"
        )

    
    def set_gigachat_client(self, client):
        """
        Установка GigaChat клиента для отслеживания баланса
        """
        self.gigachat_client = client
        logger.info("GigaChat клиент установлен для отслеживания баланса")


    def get_balance_info(self, client) -> Optional[Dict[str, Any]]:
        """
        Получение информации о балансе из GigaChat клиента

        Args:
            client: GigaChat клиент

        Returns:
            Словарь с информацией о балансе или None
        """
        if client is None:
            logger.warning("Client is None, cannot get balance")
            return None

        try:
            balance_obj = None

            # Пробуем разные методы получения баланса
            if hasattr(client, 'get_balance'):
                try:
                    balance_obj = client.get_balance()
                    logger.debug("Balance obtained via get_balance()")
                except Exception as e:
                    logger.debug(f"get_balance() failed: {e}")

            if balance_obj is None and hasattr(client, 'balance'):
                try:
                    balance_obj = client.balance
                    logger.debug("Balance obtained via balance property")
                except Exception as e:
                    logger.debug(f"balance property failed: {e}")

            if balance_obj is None and hasattr(client, 'get_account_balance'):
                try:
                    balance_obj = client.get_account_balance()
                    logger.debug("Balance obtained via get_account_balance()")
                except Exception as e:
                    logger.debug(f"get_account_balance() failed: {e}")

            if balance_obj is None:
                logger.warning("Could not get balance: all methods failed")
                return None

            # Преобразуем в словарь
            if hasattr(balance_obj, 'model_dump'):
                balance_dict = balance_obj.model_dump()
            elif hasattr(balance_obj, 'dict'):
                balance_dict = balance_obj.dict()
            elif hasattr(balance_obj, '__dict__'):
                balance_dict = vars(balance_obj)
            elif isinstance(balance_obj, dict):
                balance_dict = balance_obj
            else:
                # Если ничего не подошло, создаем словарь с базовой информацией
                balance_dict = {'balance': str(balance_obj), 'raw_value': balance_obj}

            # Добавляем временную метку
            balance_dict['timestamp'] = datetime.now().isoformat()

            logger.info(f"Balance retrieved successfully. Fields: {list(balance_dict.keys())}")
            return balance_dict

        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return None
        
    def load_documents_from_dict(
            self, documents_dict: Dict[str, str], force_reload: bool = False) -> None:
        """
        Загрузка документов из словаря с кэшированием FAISS индекса
        
        Args:
            documents_dict: Словарь {doc_id: text}
            force_reload: Принудительная перезагрузка
        """
        logger.info(f"Загрузка {len(documents_dict)} документов из словаря")

        # Разделяем на чанки через TextSplitter
        splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunked_docs: Dict[str, str] = {}
        for doc_id, text in documents_dict.items():
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunked_docs[f"{doc_id}_chunk_{i}"] = chunk

        # Создаем FAISS индекс с кэшированием
        from_cache = self.vector_store_manager.create_from_texts_with_cache(
            chunked_docs, 
            force_reload=force_reload
        )
        
        self.vector_store_initialized = True
        
        if from_cache:
            logger.info(f"📦 Загружено {len(chunked_docs)} чанков из кэша FAISS")
        else:
            logger.info(f"✅ Создано {len(chunked_docs)} чанков")
    
    def load_from_pdf_directory(self,
                                directory: Path,
                                recursive: bool = True,
                                chunk_size: int = None,
                                chunk_overlap: int = None,
                                force_reload: bool = False) -> None:
        """
        Загрузка PDF документов из директории с кэшированием FAISS индекса

        Args:
            directory: Директория с PDF файлами
            recursive: Рекурсивный обход
            chunk_size: Размер чанка
            chunk_overlap: Перекрытие чанков
            force_reload: Принудительная перезагрузка
        """
        logger.info(f"load_from_pdf_directory START: directory={directory}")

        _chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        _chunk_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

        # Загружаем документы через corpus_loader с метаданными
        logger.info(f"Calling load_from_pdf_directory_with_metadata...")
        doc_dict = self.corpus_loader.load_from_pdf_directory_with_metadata(
            directory,
            recursive=recursive,
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
            force_reload=force_reload
        )

        doc_count = len(doc_dict) if doc_dict else 0
        logger.info(f"load_from_pdf_directory_with_metadata returned {doc_count} items")
        if not doc_dict:
            logger.warning("No documents found for loading")
            return

        # Преобразуем dict в список документов
        from langchain_core.documents import Document
        documents = [
            Document(
                page_content=item.get('text', ''),
                metadata=item.get('metadata', {})
            )
            for item in doc_dict.values()
        ]

        logger.info(f"Transformed into {len(documents)} Document objects")
        if documents:
            logger.debug(f"First document preview: {documents[0].page_content[:100]}")
        else:
            logger.warning("WARNING: documents list is empty!")

        logger.info(f"Calling create_from_documents with {len(documents)} documents...")
        try:
            # Создаем FAISS индекс из документов
            self.vector_store_manager.create_from_documents(documents)
            logger.info(f"create_from_documents completed successfully")
        except Exception as e:
            logger.error(f"ERROR in create_from_documents: {e}", exc_info=True)
            raise

        logger.info(f"After create_from_documents: is_initialized={self.vector_store_manager.is_initialized}, has_vector_store={self.vector_store_manager.vector_store is not None}")

        self.vector_store_initialized = True
        logger.info(f"Set vector_store_initialized=True")
        logger.info(f"Created {len(documents)} documents/chunks")
        logger.info(f"load_from_pdf_directory COMPLETE")
    
    def load_from_pdf_directory_with_metadata(self, 
                                             directory: Path, 
                                             recursive: bool = True,
                                             chunk_size: int = None,
                                             chunk_overlap: int = None,
                                             force_reload: bool = False) -> None:
        """
        Загрузка PDF документов из директории с сохранением метаданных и кэшированием FAISS индекса
        """
        logger.info(f"Загрузка PDF из директории с метаданными: {directory}")

        # Проверяем, не менялся ли тип эмбеддингов
        cache_file = self.vector_store_manager.persist_dir / "embedding_type.txt"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                stored_type = f.read().strip()
            if stored_type != self.vector_store_manager.embedding_type:
                logger.warning(
                    f"Тип эмбеддингов изменился с {stored_type} "
                    f"на {self.vector_store_manager.embedding_type}"
                )
                logger.warning("Очищаем кэш для предотвращения ошибок...")
                self.clear_vector_cache()

        # Сохраняем текущий тип
        with open(cache_file, 'w') as f:
            f.write(self.vector_store_manager.embedding_type)        
    

        _chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        _chunk_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        
        # Загружаем документы через data_loader
        logger.info(f"[load_from_pdf_directory_with_metadata] Вызываю corpus_loader.load_from_pdf_directory_with_metadata")
        documents = self.corpus_loader.load_from_pdf_directory_with_metadata(
            directory,
            recursive=recursive,
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
            force_reload=force_reload
        )

        logger.info(f"[load_from_pdf_directory_with_metadata] corpus_loader вернул {len(documents) if documents else 0} документов")
        print(f"[DEBUG] documents type: {type(documents)}, len: {len(documents) if documents else 0}")

        if not documents:
            logger.warning("Не найдено документов для загрузки")
            return

        # Сохраняем метаданные отдельно
        self.documents_metadata = {}
        texts_for_vectorstore = {}
        metadata_for_vectorstore = {}

        # Применяем фильтрацию по токенам (удаляем чанки <30 токенов)
        # Преобразуем в LangChainDocument для фильтрации
        docs_list = []
        doc_ids_list = []

        for doc_id, data in documents.items():
            if isinstance(data, dict) and 'text' in data:
                doc = Document(
                    page_content=data['text'],
                    metadata=data.get('metadata', {})
                )
            else:
                doc = Document(
                    page_content=data if isinstance(data, str) else str(data),
                    metadata={}
                )
            docs_list.append(doc)
            doc_ids_list.append(doc_id)

        # Фильтруем по минимальному количеству токенов
        filtered_docs = filter_documents_by_token_count(
            docs_list,
            min_tokens=30,
            language=None  # Auto-detect
        )

        # Восстанавливаем отфильтрованные ID
        filtered_ids = set()
        for filtered_doc in filtered_docs:
            # Найдем исходный ID по содержимому
            for idx, orig_doc in enumerate(docs_list):
                if orig_doc.page_content == filtered_doc.page_content:
                    filtered_ids.add(doc_ids_list[idx])
                    break

        # Создаем новый словарь с отфильтрованными документами
        filtered_documents = {
            doc_id: data for doc_id, data in documents.items()
            if doc_id in filtered_ids
        }

        logger.info(
            f"📊 Фильтрация по токенам: {len(documents)} → {len(filtered_documents)} документов "
            f"(удалено {len(documents) - len(filtered_documents)} низкокачественных чанков)"
        )

        # Используем отфильтрованные документы
        documents = filtered_documents

        if not documents:
            logger.warning("⚠️ После фильтрации по токенам не осталось документов")
            return

        for doc_id, data in documents.items():
            if isinstance(data, dict) and 'metadata' in data:
                self.documents_metadata[doc_id] = data['metadata']
                texts_for_vectorstore[doc_id] = data['text']
                metadata_for_vectorstore[doc_id] = data['metadata']
            else:
                texts_for_vectorstore[doc_id] = data

        # Создаем FAISS индекс с кэшированием (передаём полные метаданные)
        from_cache = self.vector_store_manager.create_from_texts_with_cache(
            texts_for_vectorstore,
            force_reload=force_reload,
            metadata_dict=metadata_for_vectorstore
        )
        
        self.vector_store_initialized = True
        
        if from_cache:
            logger.info(f"📦 Загружено {len(texts_for_vectorstore)} документов/чанков из кэша FAISS")
        else:
            logger.info(f"✅ Создано {len(texts_for_vectorstore)} документов/чанков с метаданными")
    
    def load_from_sample_corpus(self, force_reload: bool = False) -> None:
        """
        Загрузка примеров документов с кэшированием FAISS индекса
        
        Args:
            force_reload: Принудительная перезагрузка
        """
        logger.info("Загрузка примеров документов")
        
        documents = self.corpus_loader.load_sample_corpus()
        
        chunked_docs = self.corpus_loader.split_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        from_cache = self.vector_store_manager.create_from_texts_with_cache(
            chunked_docs, 
            force_reload=force_reload
        )
        
        self.vector_store_initialized = True
        
        if from_cache:
            logger.info(f"📦 Загружено {len(chunked_docs)} чанков из кэша FAISS")
        else:
            logger.info(f"✅ Создано {len(chunked_docs)} чанков")
    
    def load_vector_store(self, hash_key: str = None) -> bool:
        """
        Загрузка FAISS индекса с диска
        
        Args:
            hash_key: Ключ для идентификации
        
        Returns:
            Успешность загрузки
        """
        if self.vector_store_manager.load_from_disk(hash_key):
            self.vector_store_initialized = True
            return True
        return False
    
    def clear_vector_cache(self, directory: Path = None):
        """
        Очистка кэша FAISS индекса
        
        Args:
            directory: Если указан, очищает кэш для конкретной директории
        """
        import shutil
        
        if directory:
            # Удаляем кэш FAISS для этой директории
            for item in self.vector_store_manager.persist_dir.glob("*"):
                if directory.name in str(item):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            logger.info(f"Кэш FAISS очищен для {directory}")
        else:
            # Очищаем весь кэш
            shutil.rmtree(self.vector_store_manager.persist_dir)
            self.vector_store_manager.persist_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Кэш FAISS полностью очищен")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Получение информации о кэше FAISS индекса
        
        Returns:
            Словарь с информацией о кэше
        """
        cache_files = list(self.vector_store_manager.persist_dir.glob("*"))
        cache_size = sum(f.stat().st_size for f in cache_files if f.is_file()) / 1024 / 1024
        
        return {
            'cache_dir': str(self.vector_store_manager.persist_dir),
            'num_cached_items': len(cache_files),
            'current_hash': self.vector_store_manager.current_hash,
            'cache_size_mb': cache_size,
            'is_initialized': self.vector_store_initialized
        }
    
    def _build_graph(self):
        """Построение графа LangGraph для RAG пайплайна"""
        from langchain_core.prompts import ChatPromptTemplate
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a helpful assistant. Answer the question based on the provided context. "
                "If you don't know the answer, say that you don't know."
            )),
            ("user", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
        ])
        
        # Получаем LLM
        #llm = self.llm_manager.get_llm()
        self.llm = self.llm_manager.get_llm() 
                
        # Определяем функцию поиска (Strategy pattern — делегируем self.retriever)
        def retrieve(state: RAGState):
            """Поиск релевантных документов"""
            docs = self.retriever.search(
                state["question"],
                k=model_config.default_k_retrieve
            )
            return {"context": docs}
        
        # Определяем функцию генерации
        def generate(state: RAGState):
            """Генерация ответа на основе контекста"""
            # Диагностика перед генерацией
            logger.debug(f"🔍 [generate] Начало генерации ответа")
            logger.debug(f"🔍 [generate] Контекст: {len(state['context'])} документов")
            logger.debug(f"🔍 [generate] Вопрос: {state['question'][:100]}...")

            docs_content = "\n\n".join(doc.page_content for doc in state["context"])

            if len(docs_content) > model_config.max_context_length:
                docs_content = docs_content[:model_config.max_context_length] + "..."

            logger.debug(f"🔍 [generate] Размер контекста: {len(docs_content)} символов")

            formatted_prompt = self.prompt.format_messages(
                question=state["question"],
                context=docs_content
            )

            logger.debug(f"🔍 [generate] Размер prompt: {len(str(formatted_prompt))} символов")

            with emit("llm.call", resource="gigachat", query_len=len(state["question"]), context_len=len(docs_content)):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    llm_start = time.time()
                    timeout_val = None if self.llm_manager.model_type == "local" else 60.0
                    logger.debug(f"🔍 [generate] Вызов LLM: model_type={self.llm_manager.model_type}, timeout={timeout_val}")
                    future = executor.submit(self.llm_manager.invoke_with_retry, formatted_prompt, timeout=timeout_val)
                    try:
                        result_timeout = 300.0 if self.llm_manager.model_type == "local" else 70.0
                        response = future.result(timeout=result_timeout)
                        llm_elapsed = time.time() - llm_start
                        logger.info(f"⏱️ LLM вызов завершен за {llm_elapsed:.1f} сек (тип: {self.llm_manager.model_type}, timeout: {timeout_val})")
                    except concurrent.futures.TimeoutError:
                        llm_elapsed = time.time() - llm_start
                        logger.error(f"❌ LLM вызов превысил timeout ({llm_elapsed:.1f}s, тип: {self.llm_manager.model_type})")
                        raise TimeoutError("LLM call exceeded timeout (60s per attempt, 3 attempts max)")
            if hasattr(response, 'content'):
                answer_text = response.content
            else:
                answer_text = str(response)

            return {"answer": format_answer(answer_text)}

        # Определяем функцию рендера (BKL-002: обеспечить полноту трассировки)
        def render(state: RAGState):
            """Форматирование ответа для вывода пользователю"""
            with emit("response.render", resource="pipeline", answer_len=len(state["answer"])):
                # Гарантируем, что ответ отформатирован и готов для пользователя
                return {"answer": state["answer"]}


        # Создаем граф с полной последовательностью activities
        graph_builder = StateGraph(RAGState).add_sequence([retrieve, generate, render])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()
        
        logger.info("LangGraph граф построен")
    
    def process_query(self, query: str, k: int = None, progress_callback=None, llm_model_name: str = None) -> GenerationResult:
        """Обработка запроса через RAG пайплайн

        Args:
            query: Запрос пользователя
            k: Количество документов для поиска
            progress_callback: Функция обратного вызова для отслеживания прогресса
                             Принимает (stage: str, message: str, progress: float)
            llm_model_name: Название LLM модели для отображения в прогрессе
        """
        def _progress(stage: str, message: str, progress: float = None):
            """Вспомогательная функция для отправки обновлений прогресса"""
            if progress_callback:
                progress_callback(stage, message, progress)
            print(f"🔍 [{stage}] {message}")

        _progress("init", "Начало обработки запроса")

        if not self.vector_store_initialized:
            raise ValueError("FAISS индекс не инициализирован. Сначала загрузите документы.")

        _progress("init", "Индекс инициализирован", 0.1)

        if self.graph is None:
            _progress("graph", "Построение графа обработки...")
            self._build_graph()
            _progress("graph", "граф готов", 0.2)

        original_k = model_config.default_k_retrieve
        if k:
            model_config.default_k_retrieve = k

        start_time = time.time()

        try:
            _progress("retrieval", "Поиск релевантных документов...", 0.3)

            # Получаем документы с реальными scores от FAISS
            docs_with_scores = self.vector_store_manager.similarity_search_with_scores(
                query, k=k or model_config.default_k_retrieve
            )
            logger.debug(f"🎯 Получены {len(docs_with_scores)} документов с scores")

            # Преобразуем в список документов и список scores
            docs = [doc for doc, _ in docs_with_scores]
            real_scores = [score for _, score in docs_with_scores]

            _progress("retrieval", f"Найдено {len(docs)} документов", 0.5)

            if docs:
                clean_text = docs[0].page_content[:100].replace('\n', ' ').replace('\r', ' ')
                top_score = real_scores[0] if real_scores else 0
                _progress("retrieval", f"Топ: '{clean_text}...' (score={top_score:.4f})", 0.6)

            # Подсчет токенов для запроса
            prompt_tokens = self.token_counter.count_text_tokens(query)
            model_display = llm_model_name if llm_model_name else "LLM"
            _progress("generation", f"Генерация ответа с помощью {model_display}...", 0.65)

            # Увеличиваем timeout для медленных моделей
            response = self.graph.invoke({"question": query}, config={"recursion_limit": 50})
            _progress("generation", "Ответ получен", 0.95)
            logger.debug (f"🔍 logger.debug: Граф выполнен, ответ получен")
            
            context_docs = response.get("context", [])
            context_text = "\n\n".join(doc.page_content for doc in context_docs)
            
            generation_time = time.time() - start_time
            
            # Формируем retrieved_docs с реальными scores
            retrieved_docs_list = [
                {
                    'doc_id': doc.metadata.get('source', f"doc_{i}"),
                    'score': real_scores[i] if i < len(real_scores) else 0.0,
                    'text': doc.page_content,
                    'page': doc.metadata.get('page', doc.metadata.get('page_number', None)),
                    'source_file': doc.metadata.get('source_file', doc.metadata.get('source', f"doc_{i}")),
                }
                for i, doc in enumerate(context_docs)
            ]

            result = GenerationResult(
                query_id="temp_id",
                query_text=query,
                context=context_text,
                answer=response["answer"],
                retrieval_results=RetrievalResult(
                    query_id="temp_id",
                    query_text=query,
                    retrieved_docs=retrieved_docs_list,
                    scores=real_scores[:len(context_docs)],
                    retrieval_time=0
                ),
                generation_time=generation_time,
                tokens_generated=len(response["answer"].split())
            )
            # УДОБНЫЙ ВЫВОД В ЛОГ
            preview = response['answer'][:200].replace('\n', ' ')
            logger.debug(f"""
            {'='*50}
            ✅ RAG RESULT
            {'='*50}
            📝 Query: {query[:80]}...
            🤖 Answer: {preview}...
            📚 Found docs: {len(response.get('context', []))}
            ⏱️  Time: {generation_time:.2f} sec
            🔢 Tokens: {len(response['answer'].split())}
            {'='*50}
            """)

            # Детали о найденных документах
            if logger.isEnabledFor(logging.DEBUG):
                for i, doc in enumerate(response.get('context', [])[:3], 1):
                    preview = doc.page_content[:200].replace('\n', ' ')
                    logger.debug(
                        f"  📄 Doc {i}: Источник '{doc.metadata.get('source', 'unknown')}'"
                        f" - {preview}..."
                    )

            # 🔍 ДИАГНОСТИКА: Полный анализ что вернулось от retriever
            print(f"\n🔍 === ДИАГНОСТИКА RETRIEVER ===")
            for i, doc in enumerate(response.get('context', [])[:3], 1):
                print(f"  Doc {i}:")
                print(f"    - source: {doc.metadata.get('source', 'NOT FOUND')}")
                print(f"    - page: {doc.metadata.get('page', 'NOT FOUND')}")
                print(f"    - page_number: {doc.metadata.get('page_number', 'NOT FOUND')}")
                print(f"    - metadata keys: {list(doc.metadata.keys())}")
            print(f"🔍 === END ДИАГНОСТИКА ===\n")            


            if self.gigachat_client:
                self.token_counter.add_request_with_balance(
                    prompt=query,
                    response=result.answer,
                    response_metadata=getattr(self.llm, 'response_metadata', None),
                    client=self.gigachat_client
                )
            else:
                self.token_counter.add_request(query, result.answer)

            return result
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            import traceback
            print(f"🔍 DEBUG: Ошибка: {e}")
            traceback.print_exc()
            
            return GenerationResult(
                query_id="temp_id",
                query_text=query,
                context="",
                answer=f"Ошибка: {str(e)}",
                retrieval_results=RetrievalResult(
                    query_id="temp_id",
                    query_text=query,
                    retrieved_docs=[],
                    scores=[]
                ),
                generation_time=0
            )
        finally:
            if k:
                model_config.default_k_retrieve = original_k
    
    def get_token_stats(self) -> Dict:
        """Получение статистики токенов"""
        return self.token_counter.get_stats()
                        
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики пайплайна"""
        stats = {
            'vector_store_initialized': self.vector_store_initialized,
            'vector_store_type': 'FAISS',
            'graph_built': self.graph is not None,
            'llm_model': self.llm_manager.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        # Добавляем информацию о кэше
        cache_info = self.get_cache_info()
        stats.update({
            'cache_size_mb': cache_info['cache_size_mb'],
            'num_cached_items': cache_info['num_cached_items']
        })
        
        return stats


def create_pipeline_from_config(retrieval_type: RetrievalType = RetrievalType.DENSE,
                               documents: Dict[str, str] = None,
                               **kwargs) -> RAGPipeline:
    """
    Создание RAG пайплайна из конфигурации
    
    Args:
        retrieval_type: Тип поиска (для совместимости)
        documents: Словарь документов
        **kwargs: Дополнительные параметры
    
    Returns:
        Настроенный RAGPipeline
    """
    pipeline = RAGPipeline(
        vector_store_type=kwargs.get('vector_store_type', 'faiss'),
                        
        chunk_size=kwargs.get('chunk_size', data_config.chunk_size),
        chunk_overlap=kwargs.get('chunk_overlap', data_config.chunk_overlap),
        embedding_model=kwargs.get('embedding_model', model_config.embedding_model_name)        
    )
    
    if documents:
        pipeline.load_documents_from_dict(documents)
    
    return pipeline
