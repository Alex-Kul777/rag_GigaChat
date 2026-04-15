"""
rag_core.py - Facade: RAG пайплайн (LangChain + LangGraph).

Классы VectorStoreManager и LLMManager вынесены в отдельные модули:
  vector_store.py — VectorStoreManager
  llm_manager.py  — LLMManager
Re-экспорт сохранён для обратной совместимости.
"""
import logging
import time
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

from config import (
    model_config, data_config, vectorstore_config,
    experiment_config, logging_config, gigachat_config
)

from models import RetrievalResult, GenerationResult, RetrievalType
from data_loader import CorpusLoader, DocumentLoader, TextSplitter
from token_counter import TokenCounter
from vector_store import VectorStoreManager       # noqa: F401 (re-export)
from llm_manager import LLMManager                # noqa: F401 (re-export)
from retriever import BaseRetriever, DenseRetriever, make_retriever  # noqa: F401

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, logging_config.log_level),
    format=logging_config.log_format,
    datefmt=logging_config.log_date_format
)
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, logging_config.log_level))


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
                 embedding_type: str = "gigachat",
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
        self.llm_manager = llm_manager or LLMManager(
            model_name=model_config.llm_model_name if llm_type == "local" else None,
            model_type=llm_type
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
        logger.info(f"Загрузка PDF из директории: {directory}")
        
        _chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        _chunk_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        
        # Загружаем документы через data_loader
        documents = self.corpus_loader.load_from_pdf_directory(
            directory,
            recursive=recursive,
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
            force_reload=force_reload
        )
        
        if not documents:
            logger.warning("Не найдено документов для загрузки")
            return
        
        # Создаем FAISS индекс с кэшированием
        from_cache = self.vector_store_manager.create_from_texts_with_cache(
            documents, 
            force_reload=force_reload
        )
        
        self.vector_store_initialized = True
        
        if from_cache:
            logger.info(f"📦 Загружено {len(documents)} документов/чанков из кэша FAISS")
        else:
            logger.info(f"✅ Создано {len(documents)} документов/чанков")
    
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
        documents = self.corpus_loader.load_from_pdf_directory_with_metadata(
            directory,
            recursive=recursive,
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
            force_reload=force_reload
        )
        
        if not documents:
            logger.warning("Не найдено документов для загрузки")
            return
        
        # Сохраняем метаданные отдельно
        self.documents_metadata = {}
        texts_for_vectorstore = {}
        
        for doc_id, data in documents.items():
            if isinstance(data, dict) and 'metadata' in data:
                self.documents_metadata[doc_id] = data['metadata']
                texts_for_vectorstore[doc_id] = data['text']
            else:
                texts_for_vectorstore[doc_id] = data
        
        # Создаем FAISS индекс с кэшированием
        from_cache = self.vector_store_manager.create_from_texts_with_cache(
            texts_for_vectorstore, 
            force_reload=force_reload
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
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            
            formatted_prompt = self.prompt.format_messages(
                question=state["question"],
                context=docs_content
            )
            
            response = self.llm.invoke(formatted_prompt)
            if hasattr(response, 'content'):
                answer_text = response.content
            else:
                answer_text = str(response)

            return {"answer": answer_text}
                
        
        # Создаем граф
        graph_builder = StateGraph(RAGState).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()
        
        logger.info("LangGraph граф построен")
    
    def process_query(self, query: str, k: int = None) -> GenerationResult:
        """Обработка запроса через RAG пайплайн"""
        print("🔍 DEBUG: Начало process_query")
        
        if not self.vector_store_initialized:
            raise ValueError("FAISS индекс не инициализирован. Сначала загрузите документы.")
        
        print(f"🔍 DEBUG: vector_store_initialized = {self.vector_store_initialized}")
        
        if self.graph is None:
            print("🔍 DEBUG: Строим граф...")
            self._build_graph()
        
        original_k = model_config.default_k_retrieve
        if k:
            model_config.default_k_retrieve = k
        
        start_time = time.time()
        
        try:
            print(f"🔍 DEBUG: Выполняем поиск для запроса: {query[:50]}...")
            
            # Сначала проверим поиск отдельно
            docs = self.vector_store_manager.similarity_search(
                query, k=k or model_config.default_k_retrieve
            )
            print(f"🔍 DEBUG: Найдено {len(docs)} документов")
            
            if docs:
                clean_text = docs[0].page_content[:100].replace('\n', ' ').replace('\r', ' ')
                print(f"🔍 DEBUG: Первый документ: {clean_text}...")

            # Подсчет токенов для запроса
            prompt_tokens = self.token_counter.count_text_tokens(query)

            print("🔍 DEBUG: Запускаем граф...")
            response = self.graph.invoke({"question": query})
            print(f"🔍 DEBUG: Граф выполнен, ответ получен")
            logger.debug (f"🔍 logger.debug: Граф выполнен, ответ получен")
            
            context_docs = response.get("context", [])
            context_text = "\n\n".join(doc.page_content for doc in context_docs)
            
            generation_time = time.time() - start_time
            
            result = GenerationResult(
                query_id="temp_id",
                query_text=query,
                context=context_text,
                answer=response["answer"],
                retrieval_results=RetrievalResult(
                    query_id="temp_id",
                    query_text=query,
                    retrieved_docs=[
                        {
                            'doc_id': doc.metadata.get('source', f"doc_{i}"),
                            'score': 1.0,
                            'text': doc.page_content,
                            'page': doc.metadata.get('page', doc.metadata.get('page_number', None)),
                        }
                        for i, doc in enumerate(context_docs)
                    ],
                    scores=[1.0] * len(context_docs),
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
