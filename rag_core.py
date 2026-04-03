"""
rag_core.py - RAG пайплайн с использованием LangChain и LangGraph
Использует загрузчик документов из data_loader.py
"""
import logging
import time
from typing import List, Dict, Optional, Any, TypedDict
from pathlib import Path
from tqdm import tqdm

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatOpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
import torch

try:
    from langchain_gigachat.chat_models import GigaChat
    from langchain_gigachat.embeddings import GigaChatEmbeddings
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False
    print("Warning: langchain-gigachat not installed. Install with: pip install langchain-gigachat")
    

from config import model_config, data_config, vectorstore_config, experiment_config, logging_config, gigachat_config

from models import RetrievalResult, GenerationResult, RetrievalType
from data_loader import CorpusLoader, DocumentLoader, TextSplitter

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

class VectorStoreManager:
    """
    Менеджер векторного хранилища на основе FAISS с кэшированием
    Поддерживает разные эмбеддинги (HuggingFace, GigaChat)
    """
    
    def __init__(self, 
                 embedding_model: str = None,
                 embedding_type: str = "gigachat",
                 persist_dir: Path = None):        
        """
        Инициализация менеджера векторного хранилища
        
        Args:
            embedding_model: Модель эмбеддингов
            persist_dir: Директория для сохранения индекса
        """
                        
        self.embedding_model = embedding_model or model_config.embedding_model_name
        self.embedding_type = embedding_type
        self.persist_dir = Path(persist_dir) if persist_dir else vectorstore_config.persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Инициализация эмбеддингов
        self.embeddings = self._init_embeddings()
        
        self.vector_store = None
        self.is_initialized = False
        self.current_hash = None
        
        logger.info(f"VectorStoreManager инициализирован. Тип эмбеддингов: {embedding_type}, Директория: {self.persist_dir}")

    def _init_embeddings(self):
        """Инициализация модели эмбеддингов"""
        if self.embedding_type == "gigachat":
            if not GIGACHAT_AVAILABLE:
                raise ImportError("langchain-gigachat не установлен")
            if not gigachat_config.api_key:
                raise ValueError("GigaChat API ключ не настроен")
            
            return GigaChatEmbeddings(
                credentials=gigachat_config.api_key,
                scope=gigachat_config.scope,
                verify_ssl_certs=gigachat_config.verify_ssl_certs
            )
        else:
            # HuggingFace эмбеддинги
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': model_config.device},
                encode_kwargs={'normalize_embeddings': True}
            )
                
    def _get_hash(self, documents: Dict[str, str]) -> str:
        """
        Генерация хеша для документов
        
        Args:
            documents: Словарь документов
        
        Returns:
            Хеш для идентификации набора документов
        """
        import hashlib
        
        # Создаем строку из всех документов
        docs_str = ""
        for doc_id, text in sorted(documents.items()):
            docs_str += f"{doc_id}:{len(text)}"
        
        return hashlib.md5(docs_str.encode()).hexdigest()[:16]
    
    def save_to_disk(self, hash_key: str = None) -> bool:
        """
        Сохранение FAISS индекса на диск
        
        Args:
            hash_key: Ключ для идентификации (если None, используем текущий)
        
        Returns:
            Успешность сохранения
        """
        if not self.is_initialized or self.vector_store is None:
            logger.warning("Векторное хранилище не инициализировано")
            return False
        
        try:
            save_path = self.persist_dir / f"faiss_index_{hash_key or self.current_hash or 'default'}"
            self.vector_store.save_local(str(save_path))
            logger.info(f"FAISS индекс сохранен: {save_path}")
            return True
                
        except Exception as e:
            logger.error(f"Ошибка сохранения FAISS индекса: {e}")
            return False
    
    def load_from_disk(self, hash_key: str = None) -> bool:
        """
        Загрузка FAISS индекса с диска
        
        Args:
            hash_key: Ключ для идентификации
        
        Returns:
            Успешность загрузки
        """
        try:
            load_path = self.persist_dir / f"faiss_index_{hash_key or 'default'}"
            if load_path.exists():
                self.vector_store = FAISS.load_local(
                    str(load_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.is_initialized = True
                self.current_hash = hash_key
                logger.info(f"FAISS индекс загружен из {load_path}")
                return True
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки FAISS индекса: {e}")
        
        return False
    
    def check_cache_exists(self, hash_key: str = None) -> bool:
        """
        Проверка существования кэша FAISS индекса
        
        Args:
            hash_key: Ключ для идентификации
        
        Returns:
            Существует ли кэш
        """
        load_path = self.persist_dir / f"faiss_index_{hash_key or 'default'}"
        return load_path.exists()
    
    def create_from_texts_with_cache(self, texts: Dict[str, str], force_reload: bool = False) -> bool:
        """
        Создание FAISS индекса из текстов с использованием кэша
        """
        # Генерируем хеш для этого набора документов
        doc_hash = self._get_hash(texts)

        # Проверяем существование кэша
        if not force_reload and self.check_cache_exists(doc_hash):
            logger.info(f"📦 Загрузка FAISS индекса из кэша (хеш: {doc_hash})")
            if self.load_from_disk(doc_hash):
                return True
            else:
                logger.warning("Не удалось загрузить из кэша, создаем заново")

        logger.info(f"🔄 Создание FAISS индекса из {len(texts)} документов...")
        start_time = time.time()

        # Подготовка документов с ограничением длины
        documents = []
        max_chars = 1500  # ~375 токенов (оставляем запас для безопасности)

        for doc_id, text in tqdm(texts.items(), desc="Обработка текстов"):
            if len(text) > max_chars:
                text = text[:max_chars]
                logger.debug(f"Документ {doc_id} обрезан до {max_chars} символов")

            doc = Document(
                page_content=text,
                metadata={"source": doc_id}
            )
            documents.append(doc)

        # Разбиваем на маленькие батчи для эмбеддингов
        batch_size = 3  # Начинаем с маленького батча

        for attempt in range(3):  # Максимум 3 попытки
            try:
                logger.info(f"Попытка {attempt + 1}: создание эмбеддингов батчами по {batch_size}")

                # Используем стандартный метод FAISS
                self.vector_store = FAISS.from_documents(documents, self.embeddings)

                self.is_initialized = True
                self.current_hash = doc_hash
                self.save_to_disk(doc_hash)

                elapsed = time.time() - start_time
                logger.info(f"✅ FAISS индекс создан за {elapsed:.2f} сек")
                return False

            except Exception as e:
                error_msg = str(e)
                if "Tokens limit exceeded" in error_msg and attempt < 2:
                    # Уменьшаем размер документов и батча
                    batch_size = max(1, batch_size // 2)
                    max_chars = max(500, max_chars - 300)

                    logger.warning(f"Ошибка токенов, уменьшаем размер: max_chars={max_chars}, batch_size={batch_size}")

                    # Обрезаем документы еще сильнее
                    for doc in documents:
                        if len(doc.page_content) > max_chars:
                            doc.page_content = doc.page_content[:max_chars]
                    continue
                raise e

        raise RuntimeError("Не удалось создать FAISS индекс после нескольких попыток")
    
    def create_from_texts_with_cacheOldVersion20260403(self, texts: Dict[str, str], force_reload: bool = False) -> bool:
        """
        Создание FAISS индекса из текстов с использованием кэша
        
        Args:
            texts: Словарь {doc_id: text}
            force_reload: Принудительная перезагрузка (игнорировать кэш)
        
        Returns:
            Загружено ли из кэша (True) или создано заново (False)
        """
        # Генерируем хеш для этого набора документов
        doc_hash = self._get_hash(texts)
        
        # Проверяем существование кэша
        if not force_reload and self.check_cache_exists(doc_hash):
            logger.info(f"📦 Загрузка FAISS индекса из кэша (хеш: {doc_hash})")
            if self.load_from_disk(doc_hash):
                return True
            else:
                logger.warning("Не удалось загрузить из кэша, создаем заново")
        
        # Создаем заново
        logger.info(f"🔄 Создание FAISS индекса из {len(texts)} документов...")
        start_time = time.time()
        
        # Преобразуем в документы LangChain
        documents = []
        for doc_id, text in texts.items():
            doc = Document(
                page_content=text,
                metadata={"source": doc_id}
            )
            documents.append(doc)
        
        # Создаем FAISS индекс
        #self.vector_store = FAISS.from_documents(documents, self.embeddings)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Пробуем создать эмбеддинги
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                break
            except Exception as e:
                if "Tokens limit exceeded" in str(e) and attempt < max_retries - 1:
                    # Уменьшаем размер документов
                    for doc in documents:
                        if len(doc.page_content) > 2000:
                            doc.page_content = doc.page_content[:2000]
                            logger.warning(f"Документ обрезан до 2000 символов (попытка {attempt + 2})")
                    continue
                raise e

        self.is_initialized = True
        self.current_hash = doc_hash
        
        # Сохраняем в кэш
        self.save_to_disk(doc_hash)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ FAISS индекс создан за {elapsed:.2f} сек")
        
        return False
    
    def create_from_documents(self, documents: List[Document]) -> None:
        """
        Создание FAISS индекса из документов (без кэша)
        
        Args:
            documents: Список документов LangChain
        """
        logger.info(f"Создание FAISS индекса из {len(documents)} документов...")
        start_time = time.time()
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        self.is_initialized = True
        elapsed = time.time() - start_time
        logger.info(f"FAISS индекс создан за {elapsed:.2f} сек")
    
    def create_from_texts(self, texts: Dict[str, str]) -> None:
        """
        Создание FAISS индекса из текстов (без кэша)
        
        Args:
            texts: Словарь {doc_id: text}
        """
        documents = []
        for doc_id, text in texts.items():
            doc = Document(
                page_content=text,
                metadata={"source": doc_id}
            )
            documents.append(doc)
        
        self.create_from_documents(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Поиск похожих документов
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
        
        Returns:
            Список документов
        """
        if not self.is_initialized:
            raise ValueError("FAISS индекс не инициализирован")
        
        start_time = time.time()
        docs = self.vector_store.similarity_search(query, k=k)
        elapsed = time.time() - start_time
        
        logger.debug(f"Поиск завершен. Найдено: {len(docs)}, Время: {elapsed:.3f} сек")
        return docs
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Добавление документов в FAISS индекс
        
        Args:
            documents: Список документов для добавления
        """
        if not self.is_initialized:
            self.create_from_documents(documents)
        else:
            self.vector_store.add_documents(documents)
            logger.info(f"Добавлено {len(documents)} документов")
            
            # Обновляем кэш после добавления
            if self.current_hash:
                self.save_to_disk(self.current_hash)


class LLMManager:
    """
    Менеджер языковых моделей
    Поддерживает локальные модели (HuggingFace) и OpenAI
    """
    
    def __init__(self, model_name: str = None, model_type: str = "local"):
        """
        Инициализация менеджера LLM
        
        Args:
            model_name: Название модели
            model_type: Тип модели ("local", "openai", "gigachat")
        """
        self.model_type = model_type
        self.model_name = model_name or model_config.llm_model_name
        self.llm = None
        self.is_initialized = False
        
        logger.info(f"LLMManager инициализирован. Модель: {model_name}")

    def load_gigachat_model(self) -> BaseLLM:
        """
        Загрузка GigaChat модели
        
        Returns:
            LangChain LLM объект
        """
        if not GIGACHAT_AVAILABLE:
            raise ImportError("langchain-gigachat не установлен. Установите: pip install langchain-gigachat")
        
        if not gigachat_config.api_key:
            raise ValueError("GigaChat API ключ не настроен в конфигурации")
        
        logger.info(f"Загрузка GigaChat модели: {gigachat_config.model}")
        
        try:
            self.llm = GigaChat(
                credentials=gigachat_config.api_key,
                verify_ssl_certs=gigachat_config.verify_ssl_certs,
                scope=gigachat_config.scope,
                model=gigachat_config.model,
                timeout=gigachat_config.timeout,
                max_retries=gigachat_config.max_retries
            )
            self.is_initialized = True
            logger.info("GigaChat модель загружена")
            return self.llm
            
        except Exception as e:
            logger.error(f"Ошибка загрузки GigaChat модели: {e}")
            raise

    def load_local_model(self) -> BaseLLM:
        """
        Загрузка локальной модели через HuggingFace

        Returns:
            LangChain LLM объект
        """
        logger.info(f"Загрузка локальной модели: {self.model_name}")
        print(f"🔍 DEBUG: Загрузка модели {self.model_name}")

        try:
            # Определяем тип данных
                        
            torch_dtype = torch.float16 if model_config.device == "cuda" else torch.float32
            print(f"🔍 DEBUG: Используем dtype: {torch_dtype}, device: {model_config.device}")

            # Загружаем модель
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                dtype=torch_dtype,
                device_map="auto" if model_config.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True  # Экономия памяти
            )
            print("🔍 DEBUG: Модель загружена")

            # Загружаем токенизатор
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            print("🔍 DEBUG: Токенизатор загружен")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # ИСПРАВЛЕНИЕ: переименовываем переменную pipeline в text_gen_pipeline
            print("🔍 DEBUG: Создаем pipeline...")
            text_gen_pipeline = hf_pipeline(  # ← переименовано с pipeline на text_gen_pipeline
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=model_config.max_new_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                do_sample=True,
                device=0 if model_config.device == "cuda" else -1
            )
            print("🔍 DEBUG: Pipeline создан")

            # Создаем LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)  # ← используем новое имя
            self.is_initialized = True

            print("✅ DEBUG: Модель успешно загружена")
            return self.llm

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            print(f"❌ DEBUG: Ошибка загрузки: {e}")
            import traceback
            traceback.print_exc()
            raise    
    
    def load_openai_model(self, api_key: str, model_name: str = "gpt-3.5-turbo") -> BaseLLM:
        """
        Загрузка OpenAI модели
        
        Args:
            api_key: API ключ OpenAI
            model_name: Название модели
        
        Returns:
            LangChain LLM объект
        """
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=model_config.temperature,
            max_tokens=model_config.max_new_tokens
        )
        self.is_initialized = True
        logger.info(f"OpenAI модель загружена: {model_name}")
        return self.llm
    

    def get_llm(self) -> BaseLLM:
        """Получение LLM (загружает если нужно)"""
        if not self.is_initialized:
            if self.model_type == "gigachat":
                self.load_gigachat_model()
            elif self.model_type == "openai":
                self.load_openai_model(model_config.openai_api_key)
            else:
                self.load_local_model()
        return self.llm
    
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
                 llm_type: str = "gigachat"):
        """
        Инициализация RAG пайплайна
        
        Args:
            chunk_size: Размер чанка для разделения документов
            chunk_overlap: Перекрытие между чанками
            embedding_model: Модель эмбеддингов
            embedding_type: Тип эмбеддингов ("huggingface", "gigachat")
            llm_type: Тип LLM ("local", "gigachat", "openai")
        """    
        chunk_size = chunk_size or data_config.chunk_size
        chunk_overlap = chunk_overlap or data_config.chunk_overlap
        
        self.vector_store_manager = VectorStoreManager(
            embedding_model=embedding_model or model_config.embedding_model_name,
            embedding_type=embedding_type,
            persist_dir=vectorstore_config.persist_dir
        )
        self.llm_manager = LLMManager(
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
        self.token_counter = TokenCounter()

        self.gigachat_client = None  # Для хранения клиента GigaChat

        logger.info(f"RAGPipeline инициализирован. chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, llm_type={llm_type}, embedding_type={embedding_type}")

    
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
        
    def load_documents_from_dict(self, documents_dict: Dict[str, str], force_reload: bool = False) -> None:
        """
        Загрузка документов из словаря с кэшированием FAISS индекса
        
        Args:
            documents_dict: Словарь {doc_id: text}
            force_reload: Принудительная перезагрузка
        """
        logger.info(f"Загрузка {len(documents_dict)} документов из словаря")
        
        # Разделяем на чанки
        chunked_docs = self.corpus_loader.split_documents(
            documents_dict,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
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
                logger.warning(f"Тип эмбеддингов изменился с {stored_type} на {self.vector_store_manager.embedding_type}")
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
            ("system", "You are a helpful assistant. Answer the question based on the provided context. If you don't know the answer, say that you don't know."),
            ("user", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
        ])
        
        # Получаем LLM
        #llm = self.llm_manager.get_llm()
        self.llm = self.llm_manager.get_llm() 
                
        # Определяем функцию поиска
        def retrieve(state: RAGState):
            """Поиск релевантных документов"""
            docs = self.vector_store_manager.similarity_search(
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
            docs = self.vector_store_manager.similarity_search(query, k=k or model_config.default_k_retrieve)
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
                            'text': doc.page_content
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
                    logger.debug(f"  📄 Doc {i}: Источник '{doc.metadata.get('source', 'unknown')}' - {preview}...")            


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

# token_counter.py
import tiktoken
from typing import List, Dict, Any
from collections import defaultdict

class TokenCounter:
    """Счетчик токенов для RAG системы"""
    
    def __init__(self):
        self.reset()
        self.balance_history = []  # История баланса
        self.last_balance = None
        self.last_balance_time = None

        # Инициализируем tokenizer для разных моделей
        try:
            # Для GigaChat/OpenAI
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoder = None
    
    def reset(self):
        """Сброс счетчиков"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.num_requests = 0
        self.details = []


    def calculate_balance_delta(self, initial_balance: Optional[Dict], final_balance: Optional[Dict]) -> Dict[str, Any]:
        """
        Расчет расхода по дельте баланса

        Args:
            initial_balance: Начальный баланс (может быть None)
            final_balance: Конечный баланс (может быть None)

        Returns:
            Словарь с дельтой
        """
        delta = {
            'error': None,
            'has_data': False,
            'timestamp': datetime.now().isoformat()
        }

        # Проверяем, что оба баланса получены
        if initial_balance is None:
            delta['error'] = 'initial_balance is None'
            delta['has_data'] = False
            logger.warning("initial_balance is None, cannot calculate delta")
            return delta

        if final_balance is None:
            delta['error'] = 'final_balance is None'
            delta['has_data'] = False
            logger.warning("final_balance is None, cannot calculate delta")
            return delta

        # Проверяем, что это словари
        if not isinstance(initial_balance, dict):
            delta['error'] = f'initial_balance is not a dict: {type(initial_balance)}'
            logger.warning(delta['error'])
            return delta

        if not isinstance(final_balance, dict):
            delta['error'] = f'final_balance is not a dict: {type(final_balance)}'
            logger.warning(delta['error'])
            return delta

        # Пробуем разные возможные поля баланса
        balance_fields = ['balance', 'available', 'total', 'amount', 'value', 'credits']

        for field in balance_fields:
            if field in initial_balance and field in final_balance:
                try:
                    initial_value = float(initial_balance[field])
                    final_value = float(final_balance[field])
                    delta[field] = initial_value - final_value
                    delta['has_data'] = True
                except (ValueError, TypeError) as e:
                    logger.debug(f"Не удалось преобразовать поле {field}: {e}")

        # Поля для токенов
        token_fields = ['tokens_used', 'tokens', 'total_tokens', 'usage']

        for field in token_fields:
            if field in final_balance:
                try:
                    delta[field] = final_balance.get(field, 0)
                    delta['has_data'] = True
                except (ValueError, TypeError):
                    pass
                
        if not delta.get('has_data'):
            delta['error'] = 'No valid balance fields found'
            delta['available_fields_initial'] = list(initial_balance.keys())
            delta['available_fields_final'] = list(final_balance.keys())
            logger.warning(f"No valid balance fields found. Initial fields: {list(initial_balance.keys())}, Final fields: {list(final_balance.keys())}")

        return delta
        
    
    def add_request_with_balance(self, 
                                 prompt: str, 
                                 response: str = None, 
                                 response_metadata: Dict = None,
                                 client = None) -> int:
        """
        Добавление запроса с учетом баланса
        
        Args:
            prompt: Текст запроса
            response: Текст ответа
            response_metadata: Метаданные ответа от API
            client: GigaChat клиент (для получения баланса)
        
        Returns:
            Количество использованных токенов
        """
        # Получаем баланс до запроса
        balance_before = None
        if client:
            balance_before = self.get_balance_info(client)
        
        # Добавляем запрос
        tokens_used = self.add_request(prompt, response, response_metadata)
        
        # Получаем баланс после запроса
        balance_after = None
        if client:
            balance_after = self.get_balance_info(client)
        
        # Рассчитываем дельту
        if balance_before and balance_after:
            delta = self.calculate_balance_delta(balance_before, balance_after)
            logger.info(f"Расход по балансу: {delta}")
            
            # Сохраняем информацию о дельте
            self.details[-1]['balance_delta'] = delta
        
        return tokens_used
    
    def get_balance_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по балансу
        
        Returns:
            Словарь со статистикой баланса
        """
        if not self.balance_history:
            return {'has_balance_data': False}
        
        first_balance = self.balance_history[0]
        last_balance = self.balance_history[-1]
        
        # Рассчитываем общий расход
        total_delta = self.calculate_balance_delta(first_balance, last_balance)
        
        return {
            'has_balance_data': True,
            'first_balance': first_balance,
            'last_balance': last_balance,
            'total_delta': total_delta,
            'num_balance_checks': len(self.balance_history),
            'balance_history': self.balance_history
        }
    
    def get_stats_for_json(self) -> Dict[str, Any]:
        """
        Получение статистики в формате для JSON (с балансом)
        """
        stats = {
            'num_requests': self.num_requests,
            'total_prompt_tokens': self.prompt_tokens,
            'total_completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_request': self.total_tokens / self.num_requests if self.num_requests > 0 else 0,
            'estimated_cost_usd': self.estimate_cost()
        }
        
        # Добавляем статистику баланса
        balance_stats = self.get_balance_statistics()
        if balance_stats.get('has_balance_data'):
            stats['balance'] = {
                'first_balance': balance_stats.get('first_balance'),
                'last_balance': balance_stats.get('last_balance'),
                'total_delta': balance_stats.get('total_delta'),
                'num_checks': balance_stats.get('num_balance_checks')
            }
        
        return stats
            
    def get_balance_info(self, client) -> Optional[Dict[str, Any]]:
        """
        Получение информации о балансе из GigaChat клиента
        
        Args:
            client: GigaChat клиент
        
        Returns:
            Словарь с информацией о балансе или None
        """
        try:
            balance_obj = None
            
            # Пробуем разные методы получения баланса
            if hasattr(client, 'get_balance'):
                balance_obj = client.get_balance()
            elif hasattr(client, 'balance'):
                balance_obj = client.balance
            elif hasattr(client, 'get_account_balance'):
                balance_obj = client.get_account_balance()
            
            if balance_obj is None:
                logger.warning("Не удалось получить баланс: ни один метод не сработал")
                return None
            
            # Преобразуем в словарь
            if hasattr(balance_obj, 'model_dump'):
                balance_dict = balance_obj.model_dump()
            elif hasattr(balance_obj, 'dict'):
                balance_dict = balance_obj.dict()
            elif hasattr(balance_obj, '__dict__'):
                balance_dict = vars(balance_obj)
            else:
                balance_dict = {'balance': str(balance_obj)}
            
            # Добавляем временную метку
            balance_dict['timestamp'] = datetime.now().isoformat()
            
            # Сохраняем в историю
            self.balance_history.append(balance_dict)
            self.last_balance = balance_dict
            self.last_balance_time = datetime.now()
            
            logger.info(f"Баланс получен: {balance_dict}")
            return balance_dict
            
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return None
            
    def count_text_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте"""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Приблизительный подсчет (1 токен ≈ 4 символа для русского)
            return len(text) // 4
    
    def add_request(self, prompt: str, response: str = None, response_metadata: Dict = None):
        """
        Добавление информации о запросе
        
        Args:
            prompt: Текст запроса
            response: Текст ответа (опционально)
            response_metadata: Метаданные ответа от API
        """
        self.num_requests += 1
        
        # Если есть метаданные от API - используем их
        if response_metadata and 'token_usage' in response_metadata:
            token_usage = response_metadata['token_usage']
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            total_tokens = token_usage.get('total_tokens', 0)
        else:
            # Подсчитываем вручную
            prompt_tokens = self.count_text_tokens(prompt)
            completion_tokens = self.count_text_tokens(response) if response else 0
            total_tokens = prompt_tokens + completion_tokens
        
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        
        self.details.append({
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'prompt_preview': prompt[:100]
        })
        
        return total_tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        return {
            'num_requests': self.num_requests,
            'total_prompt_tokens': self.prompt_tokens,
            'total_completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_request': self.total_tokens / self.num_requests if self.num_requests > 0 else 0,
            'estimated_cost_usd': self.estimate_cost()
        }
    
    def estimate_cost(self, model: str = "gigachat") -> float:
        """
        Оценка стоимости (примерные цены)
        GigaChat: ~0.0001$ за 1000 токенов
        """
        if model == "gigachat":
            return self.total_tokens * 0.0001 / 1000
        else:
            return self.total_tokens * 0.002 / 1000  # OpenAI
    
    def print_summary(self):
        """Вывод сводки"""
        stats = self.get_stats()
        print(f"""
        {'='*50}
        📊 СТАТИСТИКА ТОКЕНОВ ЗА ЭКСПЕРИМЕНТ
        {'='*50}
        📝 Количество запросов: {stats['num_requests']}
        🔢 Всего токенов: {stats['total_tokens']:,}
        📤 Prompt токены: {stats['total_prompt_tokens']:,}
        📥 Completion токены: {stats['total_completion_tokens']:,}
        📊 Среднее токенов на запрос: {stats['avg_tokens_per_request']:.0f}
        💰 Оценочная стоимость: ${stats['estimated_cost_usd']:.4f}
        {'='*50}
        """)
    
    def save_to_file(self, filepath: str):
        """Сохранение статистики в файл"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': self.get_stats(),
                'details': self.details
            }, f, indent=2, ensure_ascii=False)

    def get_stats_for_json(self) -> Dict[str, Any]:
        """
        Получение статистики в формате для JSON
        """
        return {
            'num_requests': self.num_requests,
            'total_prompt_tokens': self.prompt_tokens,
            'total_completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_request': self.total_tokens / self.num_requests if self.num_requests > 0 else 0,
            'estimated_cost_usd': self.estimate_cost()
        }

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


""" if __name__ == "__main__": """
"""     import time """
"""      """
"""     print("=" * 60) """
"""     print("Тестирование RAG пайплайна с LangChain и data_loader") """
"""     print("=" * 60) """
"""      """
"""     # Создание пайплайна """
"""     pipeline = RAGPipeline( """
"""         chunk_size=data_config.chunk_size, """
"""         chunk_overlap=data_config.chunk_overlap, """
"""         embedding_model=model_config.embedding_model_name) """
"""      """
"""     # Вариант 1: Загрузка из примера """
"""     #print("\n📁 Вариант 1: Загрузка из примера...") """
"""     #pipeline.load_from_sample_corpus() """
"""      """
"""     #Вариант 2: Загрузка из PDF директории (раскомментируйте для использования) """
"""     pdf_dir = Path("data/domain_1_AI/books") """
"""     if pdf_dir.exists(): """
"""         print(f"\n📁 Вариант 2: Загрузка PDF из {pdf_dir}...") """
"""         pipeline.load_from_pdf_directory_with_metadata(pdf_dir, recursive=True, force_reload=False) """
"""      """
"""     # Обработка запроса """
"""     query = "Что такое нейросети и как они работают?" """
"""     print(f"\n❓ Запрос: {query}") """
"""     print("-" * 60) """
"""      """
"""     start_time = time.time() """
"""     result = pipeline.process_query(query, k=3) """
"""     elapsed = time.time() - start_time """
"""      """
"""     print(f"\n🤖 Ответ:\n{result.answer}") """
"""     print(f"\n📚 Найденные документы: {len(result.retrieval_results.retrieved_docs)}") """
"""     print(f"⏱️  Время обработки: {elapsed:.2f} сек") """
"""      """
"""     # Статистика """
"""     print(f"\n📊 Статистика:") """
"""     for key, value in pipeline.get_stats().items(): """
"""         print(f"  {key}: {value}") """
"""      """
"""     print("\n✅ Тестирование завершено") """
if __name__ == "__main__1":
    # Использование с GigaChat
    pipeline = RAGPipeline(
        chunk_size=data_config.chunk_size,
        chunk_overlap=data_config.chunk_overlap,
        embedding_type="gigachat",  # Используем GigaChat эмбеддинги
        llm_type="gigachat"          # Используем GigaChat LLM
    )
    
    # Загрузка документов
    pdf_dir = Path("data/domain_1_AI/books")
    if pdf_dir.exists():
        pipeline.load_from_pdf_directory_with_metadata(pdf_dir, recursive=True, force_reload=False)
    
    # Обработка запроса
    result = pipeline.process_query("Что такое нейросети?", k=3)
    print(result.answer)

if __name__ == "__main__":
    # Использование с GigaChat
    strTest = "Tes of TEST"
    logger.info(f" logger.info strTest: {strTest}")

    current_level = logging.getLevelName(logger.level)
    logger.info(f"Текущий уровень логгера: {current_level}, strTest: {strTest}")

    current_level = logging.getLevelName(logger.getEffectiveLevel())
    logger.info(f"📊 Уровень getEffectiveLevel логгера: {current_level} | logger.info strTest: {strTest}")    

    pipeline = RAGPipeline(
        chunk_size=data_config.chunk_size,
        chunk_overlap=data_config.chunk_overlap,
        #embedding_type="gigachat",  # Используем GigaChat эмбеддинги
        #llm_type="gigachat"          # Используем GigaChat LLM
    )
    token_counter = pipeline.token_counter

    # Загрузка документов
    #pdf_dir = Path("data/domain_2_Debug/books")
    pdf_dir = Path("data/domain_7_UAV/books")

    if pdf_dir.exists():
        pipeline.load_from_pdf_directory_with_metadata(pdf_dir, recursive=True, force_reload=False)
    
    # Обработка запроса
    #query = "Что такое самолёт системы утка?"
    query = "Что такое точка фокуса для самолёта и как её расчитать?"
    result = pipeline.process_query(query, k=5)
    #print(f"result.answer: {result.answer}")

    preview = result.answer.replace('\n', ' ')

    print(f"""
    {'='*50}
    ✅ RAG RESULT
    {'='*50}
    📝 Query: {result.query_text}
    🤖 Answer: {preview}...
    📚 Found docs: {len(result.retrieval_results.retrieved_docs)}
    ⏱️  Time: {result.generation_time:.2f} sec
    🔢 Tokens: {result.tokens_generated}
    {'='*50}
    """)
    # Детали о найденных документах
    
    for i, doc in enumerate(result.retrieval_results.retrieved_docs, 1):
        doc_id = doc.get('doc_id', 'unknown')
        doc_score = doc.get('score', 0)
        doc_preview = doc.get('text', '').replace('\n', ' ')
        print(f"  📄 Doc {i}: {doc_id} (score: {doc_score:.3f}) - {doc_preview}...")

    # Выводим статистику токенов
    token_counter.print_summary()


    # Сохраняем в файл
    token_counter.save_to_file("token_stats.json")        
