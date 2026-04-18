"""
vector_store.py - Менеджер векторного хранилища FAISS
"""
import logging
import time
from typing import List, Dict, Optional, Any
from pathlib import Path

from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from langchain_gigachat.embeddings import GigaChatEmbeddings
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False

from rag_gigachat.config import model_config, vectorstore_config, gigachat_config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Менеджер векторного хранилища на основе FAISS с кэшированием.
    Поддерживает разные эмбеддинги (HuggingFace, GigaChat).
    """

    def __init__(self,
                 embedding_model: str = None,
                 embedding_type: str = "gigachat",
                 persist_dir: Path = None):
        """
        Инициализация менеджера векторного хранилища

        Args:
            embedding_model: Модель эмбеддингов
            embedding_type: Тип эмбеддингов ("huggingface", "gigachat")
            persist_dir: Директория для сохранения индекса
        """
        self.embedding_model = embedding_model or model_config.embedding_model_name
        self.embedding_type = embedding_type
        self.persist_dir = Path(persist_dir) if persist_dir else vectorstore_config.persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = self._init_embeddings()

        self.vector_store = None
        self.is_initialized = False
        self.current_hash = None

        logger.info(
            f"VectorStoreManager инициализирован. "
            f"Тип эмбеддингов: {embedding_type}, Директория: {self.persist_dir}"
        )

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
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': model_config.device},
                encode_kwargs={'normalize_embeddings': True}
            )

    def _get_hash(self, documents: Dict[str, str]) -> str:
        """
        Генерация хеша для набора документов

        Args:
            documents: Словарь документов

        Returns:
            16-символьный MD5-хеш
        """
        import hashlib
        docs_str = "".join(f"{doc_id}:{len(text)}" for doc_id, text in sorted(documents.items()))
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
            hash_suffix = hash_key or self.current_hash or 'default'
            save_path = self.persist_dir / f"faiss_index_{hash_suffix}"
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
            True если кэш существует
        """
        load_path = self.persist_dir / f"faiss_index_{hash_key or 'default'}"
        return load_path.exists()

    def create_from_texts_with_cache(
            self, texts: Dict[str, str], force_reload: bool = False,
            metadata_dict: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
        """Создание FAISS индекса из текстов с использованием кэша

        Args:
            texts: Словарь {doc_id: text}
            force_reload: Принудительная перезагрузка
            metadata_dict: Опциональный словарь метаданных {doc_id: metadata_dict}
        """
        # Валидация входных данных
        if not texts:
            logger.warning("⚠️ Получен пустой словарь текстов")
            raise ValueError("Нельзя создать FAISS индекс без текстов")

        # Фильтруем пустые тексты
        non_empty_texts = {doc_id: text for doc_id, text in texts.items() if text and text.strip()}
        if not non_empty_texts:
            logger.warning("⚠️ Все тексты пусты или содержат только пробелы")
            raise ValueError("После фильтрации нет валидных текстов для эмбеддинга")

        doc_hash = self._get_hash(non_empty_texts)

        if not force_reload and self.check_cache_exists(doc_hash):
            logger.info(f"📦 Загрузка FAISS индекса из кэша (хеш: {doc_hash})")
            if self.load_from_disk(doc_hash):
                return True
            else:
                logger.warning("Не удалось загрузить из кэша, создаем заново")

        logger.info(f"🔄 Создание FAISS индекса из {len(non_empty_texts)} документов...")
        start_time = time.time()

        documents = []
        max_chars = 1500  # ~375 токенов

        for doc_id, text in tqdm(non_empty_texts.items(), desc="Обработка текстов"):
            if len(text) > max_chars:
                text = text[:max_chars]
                logger.debug(f"Документ {doc_id} обрезан до {max_chars} символов")

            # Создаём метаданные: базовый source + дополнительные из metadata_dict
            meta = {"source": doc_id}
            if metadata_dict and doc_id in metadata_dict:
                meta.update(metadata_dict[doc_id])

            documents.append(Document(page_content=text, metadata=meta))

        # Финальная проверка перед созданием FAISS индекса
        if not documents:
            logger.error("❌ Список документов пуст перед созданием FAISS индекса")
            raise RuntimeError("Не удалось подготовить документы для эмбеддинга")

        batch_size = 3

        for attempt in range(3):
            try:
                logger.info(f"Попытка {attempt + 1}: создание эмбеддингов из {len(documents)} документов")
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                self.is_initialized = True
                self.current_hash = doc_hash
                self.save_to_disk(doc_hash)
                elapsed = time.time() - start_time
                logger.info(f"✅ FAISS индекс создан за {elapsed:.2f} сек")
                return True

            except IndexError as e:
                logger.error(f"❌ IndexError при создании FAISS: {e}")
                logger.error(f"Количество документов: {len(documents)}, Попытка: {attempt + 1}")
                if attempt < 2:
                    logger.warning("Пробуем с меньшим размером текстов...")
                    max_chars = max(500, max_chars - 300)
                    for doc in documents:
                        if len(doc.page_content) > max_chars:
                            doc.page_content = doc.page_content[:max_chars]
                    continue
                raise
            except Exception as e:
                error_msg = str(e)
                if "Tokens limit exceeded" in error_msg and attempt < 2:
                    batch_size = max(1, batch_size // 2)
                    max_chars = max(500, max_chars - 300)
                    logger.warning(
                        f"Ошибка токенов, уменьшаем размер: "
                        f"max_chars={max_chars}, batch_size={batch_size}"
                    )
                    for doc in documents:
                        if len(doc.page_content) > max_chars:
                            doc.page_content = doc.page_content[:max_chars]
                    continue
                raise e

        raise RuntimeError("Не удалось создать FAISS индекс после нескольких попыток")

    def create_from_documents(self, documents: List[Document]) -> None:
        """
        Создание FAISS индекса из документов (без кэша)

        Args:
            documents: Список документов LangChain
        """
        if not documents:
            raise ValueError("Список документов не может быть пустым")

        non_empty_docs = [d for d in documents if d.page_content and d.page_content.strip()]
        if not non_empty_docs:
            raise ValueError("После фильтрации нет валидных документов для эмбеддинга")

        logger.info(f"Создание FAISS индекса из {len(non_empty_docs)} документов...")
        start_time = time.time()
        self.vector_store = FAISS.from_documents(non_empty_docs, self.embeddings)
        self.is_initialized = True
        elapsed = time.time() - start_time
        logger.info(f"✅ FAISS индекс создан за {elapsed:.2f} сек")

    def create_from_texts(self, texts: Dict[str, str]) -> None:
        """
        Создание FAISS индекса из текстов (без кэша)

        Args:
            texts: Словарь {doc_id: text}
        """
        if not texts:
            raise ValueError("Словарь текстов не может быть пустым")

        non_empty_texts = {doc_id: text for doc_id, text in texts.items() if text and text.strip()}
        if not non_empty_texts:
            raise ValueError("После фильтрации нет валидных текстов")

        documents = [
            Document(page_content=text, metadata={"source": doc_id})
            for doc_id, text in non_empty_texts.items()
        ]
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

        logger.debug(f"🔍 ПОИСК: query='{query[:50]}...', k={k}")
        logger.debug(f"📊 Поиск завершен. Найдено: {len(docs)}, Время: {elapsed:.3f} сек")
        return docs

    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Поиск похожих документов с оценками релевантности

        Args:
            query: Поисковый запрос
            k: Количество результатов

        Returns:
            Список кортежей (Document, score)
        """
        if not self.is_initialized:
            raise ValueError("FAISS индекс не инициализирован")

        start_time = time.time()
        logger.debug(f"🔍 ПОИСК: query='{query[:50]}...', k={k}")

        # Получаем документы со scores
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        elapsed = time.time() - start_time

        # Логируем результаты
        if docs_with_scores:
            scores = [score for _, score in docs_with_scores]
            logger.debug(f"📊 RAW SCORES: {scores}")

            for i, (doc, score) in enumerate(docs_with_scores[:k], 1):
                logger.debug(f"  📄 Doc {i}: score={score:.4f}, source={doc.metadata.get('source', 'unknown')}")

        logger.debug(f"⏱️ Поиск завершен за {elapsed:.3f} сек, найдено {len(docs_with_scores)} документов")
        return docs_with_scores

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
            if self.current_hash:
                self.save_to_disk(self.current_hash)
