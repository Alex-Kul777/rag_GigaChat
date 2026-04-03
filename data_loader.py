"""
data_loader.py - Загрузчик данных для RAG системы с кэшированием
Поддерживает:
- Загрузку PDF через LangChain с кэшированием
- Сохранение загруженных документов в JSON
- Флаг принудительной перезагрузки
"""
import json
import csv
import logging
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import random

# LangChain imports
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.document_loaders import DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document as LangChainDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not installed. Install with: pip install langchain langchain-community")

import pandas as pd
import numpy as np

from models import TestSample

# Импортируем конфигурацию
from config import data_config, model_config, logging_config    


# Настройка логирования
logger = logging.getLogger(__name__)

@dataclass
class CorpusStats:
    """Статистика по корпусу документов"""
    num_documents: int
    total_chars: int
    avg_doc_length: float
    min_doc_length: int
    max_doc_length: int
    unique_terms: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __str__(self) -> str:
        return f"""
        Статистика корпуса:
        - Всего документов: {self.num_documents}
        - Всего символов: {self.total_chars:,}
        - Средняя длина: {self.avg_doc_length:.1f} симв.
        - Мин. длина: {self.min_doc_length}
        - Макс. длина: {self.max_doc_length}
        - Уникальных терминов: {self.unique_terms:,}
        """


class DocumentCache:
    """
    Кэш для загруженных документов
    Сохраняет документы в JSON и pickle форматах
    """
    
    def __init__(self, cache_dir: Path = Path("data/cache")):
        """
        Инициализация кэша
        
        Args:
            cache_dir: Директория для кэша
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DocumentCache инициализирован. Директория: {self.cache_dir}")
    
    def _get_cache_key(self, directory: Path, recursive: bool, chunk_size: int = None) -> str:
        """
        Генерация ключа кэша на основе параметров загрузки
        
        Args:
            directory: Директория с документами
            recursive: Рекурсивный обход
            chunk_size: Размер чанка (если используется)
        
        Returns:
            Уникальный ключ для кэша
        """
        # Создаем хеш от параметров
        cache_str = f"{directory}_{recursive}_{chunk_size}"
        # Добавляем информацию о файлах в директории
        if directory.exists():
            files = sorted([str(f.relative_to(directory)) for f in directory.rglob("*") if f.is_file()])
            cache_str += "_".join(files)
        
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def save(self, documents: Dict[str, str], 
            directory: Path, 
            recursive: bool, 
            chunk_size: int = None,
            metadata: Dict[str, Any] = None) -> Path:
        """
        Сохранение документов в кэш
        
        Args:
            documents: Словарь документов
            directory: Исходная директория
            recursive: Рекурсивный обход
            chunk_size: Размер чанка
            metadata: Дополнительные метаданные
        
        Returns:
            Путь к сохраненному файлу
        """
        cache_key = self._get_cache_key(directory, recursive, chunk_size)
        
        # Сохраняем в JSON
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        # Сохраняем метаданные
        metadata_file = self.cache_dir / f"{cache_key}_meta.json"
        meta_data = {
            'cache_key': cache_key,
            'directory': str(directory),
            'recursive': recursive,
            'chunk_size': chunk_size,
            'timestamp': datetime.now().isoformat(),
            'num_documents': len(documents),
            'metadata': metadata or {}
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Документы сохранены в кэш: {cache_file}")
        return cache_file
    
    def load(self, directory: Path, recursive: bool, chunk_size: int = None) -> Optional[Dict[str, str]]:
        """
        Загрузка документов из кэша

        Returns:
            Словарь документов или None если кэш не найден
        """
        cache_key = self._get_cache_key(directory, recursive, chunk_size)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            logger.debug(f"Кэш не найден для {directory}")
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)

            logger.info(f"📦 Загружено {len(documents)} документов из кэша: {cache_file.name}")
            return documents
        except Exception as e:
            logger.error(f"Ошибка загрузки из кэша: {e}")
            return None    
    
    def exists(self, directory: Path, recursive: bool, chunk_size: int = None) -> bool:
        """Проверка наличия кэша"""
        cache_key = self._get_cache_key(directory, recursive, chunk_size)
        cache_file = self.cache_dir / f"{cache_key}.json"
        return cache_file.exists()
    
    def clear(self, directory: Path = None):
        """
        Очистка кэша
        
        Args:
            directory: Если указан, удаляет кэш только для этой директории
        """
        if directory:
            # Удаляем все кэши для этой директории
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'metadata' in data:
                            if data.get('directory') == str(directory):
                                cache_file.unlink()
                                logger.info(f"Удален кэш: {cache_file}")
                except:
                    pass
        else:
            # Очищаем все кэши
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Кэш полностью очищен")

class DocumentLoader:
    """
    Загрузчик документов с использованием LangChain и извлечением метаданных
    """
    
    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache = DocumentCache(cache_dir)
        logger.info("DocumentLoader инициализирован с LangChain и кэшированием")
    
    def extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Извлечение метаданных из PDF файла
        
        Args:
            pdf_path: Путь к PDF файлу
        
        Returns:
            Словарь с метаданными
        """
        try:
            import PyPDF2
            metadata = {}
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Извлекаем метаданные документа
                if reader.metadata:
                    metadata = {
                        'title': reader.metadata.get('/Title', pdf_path.stem),
                        'author': reader.metadata.get('/Author', 'Неизвестен'),
                        'subject': reader.metadata.get('/Subject', ''),
                        'keywords': reader.metadata.get('/Keywords', ''),
                        'creator': reader.metadata.get('/Creator', ''),
                        'producer': reader.metadata.get('/Producer', ''),
                        'creation_date': reader.metadata.get('/CreationDate', ''),
                        'modification_date': reader.metadata.get('/ModDate', ''),
                    }
                
                # Добавляем дополнительную информацию
                metadata.update({
                    'filename': pdf_path.name,
                    'filepath': str(pdf_path),
                    'num_pages': len(reader.pages),
                    'file_size': pdf_path.stat().st_size,
                    'file_modified': datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
                })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Ошибка извлечения метаданных из {pdf_path}: {e}")
            return {
                'filename': pdf_path.name,
                'filepath': str(pdf_path),
                'error': str(e)
            }
    
    def load_pdf_with_metadata(self, pdf_path: Path) -> List[LangChainDocument]:
        """
        Загрузка PDF файла с извлечением метаданных
        
        Args:
            pdf_path: Путь к PDF файлу
        
        Returns:
            Список документов LangChain с метаданными
        """
        logger.info(f"DEBUG MODE: {logging_config.log_level.upper() == 'DEBUG'}")
        try:
            # Извлекаем метаданные документа
            logger.info(f"Извлекаем метаданные документа: {pdf_path}")
            doc_metadata = self.extract_pdf_metadata(pdf_path)
            
            # Загружаем PDF через LangChain
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            logger.info(f"documents = loader.load(): {len(documents)}")
            # Добавляем метаданные к каждой странице
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'doc_title': doc_metadata.get('title', ''),
                    'doc_author': doc_metadata.get('author', ''),
                    'doc_subject': doc_metadata.get('subject', ''),
                    'doc_keywords': doc_metadata.get('keywords', ''),
                    'filename': doc_metadata.get('filename', ''),
                    'filepath': doc_metadata.get('filepath', ''),
                    'total_pages': doc_metadata.get('num_pages', 0),
                    'page_number': i + 1
                })

            #print ( f"load_pdf_with_metadata , logging_config.log_level.upper() = {logging_config.log_level.upper()}")
            # ========== НОВЫЙ КОД: Сохранение текстовых файлов при DEBUG ==========
            # Проверяем уровень логирования
            if logging_config.log_level.upper() == "DEBUG":
                # Создаем директорию для текстовых файлов
                # Извлекаем путь к директории с PDF
                pdf_parent_dir = pdf_path.parent
                # Формируем путь для текстовых файлов
                temp_dir = Path("temp") / pdf_path.parent.relative_to(Path("."))
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Сохраняем каждую страницу в отдельный текстовый файл
                for i, doc in enumerate(documents):
                    # Имя файла: имя_PDF_страница_N.txt
                    txt_filename = f"{pdf_path.stem}_page_{i+1}.txt"
                    txt_path = temp_dir / txt_filename

                    try:
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            # Записываем метаданные
                            f.write(f"=== МЕТАДАННЫЕ ДОКУМЕНТА ===\n")
                            f.write(f"Файл: {pdf_path.name}\n")
                            f.write(f"Страница: {i+1} из {len(documents)}\n")
                            f.write(f"Заголовок: {doc_metadata.get('title', 'Не указан')}\n")
                            f.write(f"Автор: {doc_metadata.get('author', 'Не указан')}\n")
                            f.write(f"Дата создания: {doc_metadata.get('creation_date', 'Не указана')}\n")
                            f.write(f"\n=== СОДЕРЖИМОЕ СТРАНИЦЫ ===\n\n")
                            # Записываем текст страницы
                            f.write(doc.page_content)

                        logger.debug(f"Сохранен текстовый файл: {txt_path}")

                    except Exception as e:
                        logger.warning(f"Не удалось сохранить текстовый файл {txt_path}: {e}")

                logger.info(f"💾 Сохранено {len(documents)} текстовых файлов в: {temp_dir}")

            logger.info(f"Загружен PDF: {pdf_path.name}, {len(documents)} страниц, автор: {doc_metadata.get('author', 'неизвестен')}")
            return documents
            
        except Exception as e:
            logger.error(f"Ошибка загрузки PDF {pdf_path}: {e}")
            return []
    
    def load_directory_with_metadata(self, 
                                     directory: Path, 
                                     glob_pattern: str = "**/*.pdf",
                                     recursive: bool = True,
                                     force_reload: bool = False) -> List[LangChainDocument]:
        """
        Загрузка всех PDF из директории с метаданными
        """
        # Проверяем кэш
        if not force_reload:
            cache_key = self.cache._get_cache_key(directory, recursive, None)
            langchain_cache = self.cache.cache_dir / f"{cache_key}_with_metadata.pkl"
            
            if langchain_cache.exists():
                try:
                    with open(langchain_cache, 'rb') as f:
                        documents = pickle.load(f)
                    logger.info(f"📦 Загружено {len(documents)} документов из кэша (с метаданными)")
                    return documents
                except Exception as e:
                    logger.warning(f"Ошибка загрузки кэша: {e}")
        
        logger.info(f"📁 Загрузка PDF из директории: {directory} (с извлечением метаданных)")
        
        documents = []
        
        # Рекурсивный обход файлов
        if recursive:
            pdf_files = list(directory.rglob(glob_pattern))
        else:
            pdf_files = list(directory.glob(glob_pattern))
        
        for pdf_file in tqdm(pdf_files, desc="Обработка PDF файлов"):
            docs = self.load_pdf_with_metadata(pdf_file)
            documents.extend(docs)
        
        # Сохраняем в кэш
        cache_key = self.cache._get_cache_key(directory, recursive, None)
        langchain_cache = self.cache.cache_dir / f"{cache_key}_with_metadata.pkl"
        with open(langchain_cache, 'wb') as f:
            pickle.dump(documents, f)
        
        logger.info(f"✅ Загружено {len(documents)} страниц из {len(pdf_files)} PDF файлов")
        return documents


class TextSplitter:
    """
    Разделитель текста на чанки с использованием LangChain
    """
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 separators: List[str] = None):
        """
        Инициализация сплиттера
        
        Args:
            chunk_size: Размер чанка в символах
            chunk_overlap: Перекрытие между чанками
            separators: Разделители для разбиения
        """
        chunk_size = chunk_size or data_config.chunk_size
        chunk_overlap = chunk_overlap or data_config.chunk_overlap
        separators = separators or data_config.chunk_separators
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        
        logger.info(f"TextSplitter инициализирован: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def split_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """
        Разделение документов на чанки
        
        Args:
            documents: Список документов LangChain
        
        Returns:
            Список чанков
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Разделено {len(documents)} документов на {len(chunks)} чанков")
        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """
        Разделение текста на чанки
        
        Args:
            text: Исходный текст
        
        Returns:
            Список чанков
        """
        chunks = self.text_splitter.split_text(text)
        return chunks



    """
    Загрузчик корпуса документов с кэшированием
    """
class CorpusLoader:
    def __init__(self, data_dir: Path = Path("data/corpus"), cache_dir: Path = Path("data/cache")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.document_loader = DocumentLoader(cache_dir) if LANGCHAIN_AVAILABLE else None
        self.text_splitter = None
        self.cache = DocumentCache(cache_dir)
        
        # Словарь для хранения метаданных документов
        self.documents_metadata = {}
        
        logger.info(f"CorpusLoader инициализирован. Директория: {self.data_dir}")
    
    def load_from_pdf_directory_with_metadata(self, 
                                             directory: Path,
                                             recursive: bool = True,
                                             chunk_size: int = None,
                                             chunk_overlap: int = None,
                                             force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Загрузка PDF документов с метаданными
        
        Returns:
            Словарь {doc_id: {'text': text, 'metadata': metadata}}
        """

        current_level = logging.getLogger().getEffectiveLevel()
        level_name = logging.getLevelName(current_level)  # Эта строка должна быть

        print(f"=== ДИАГНОСТИКА ЛОГИРОВАНИЯ ===")
        print(f"logging_config.log_level: {logging_config.log_level}")
        print(f"Эффективный уровень логгера: {level_name} ({current_level})")
        print(f"logger.isEnabledFor(logging.INFO): {logger.isEnabledFor(logging.INFO)}")
        print(f"logger.isEnabledFor(logging.DEBUG): {logger.isEnabledFor(logging.DEBUG)}")
        print(f"=================================")

        # Используем значения из конфигурации
        force_reload = force_reload if force_reload is not None else data_config.force_reload
        logger.info(f"logger.info load_from_pdf_directory_with_metadata инициализирован. force_reload: {force_reload}")
        print(f"print load_from_pdf_directory_with_metadata инициализирован. force_reload: {force_reload}")
        chunk_size = chunk_size or data_config.chunk_size
        chunk_overlap = chunk_overlap or data_config.chunk_overlap

        if not self.document_loader:
            raise ImportError("LangChain не установлен")
        
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Директория не существует: {directory}")
            return {}
        
        # Проверяем кэш
        if not force_reload:
            cached_docs = self.cache.load(directory, recursive, chunk_size)
            if cached_docs is not None:
                logger.info(f"📦 Используем кэш с метаданными: {len(cached_docs)} документов")
                return cached_docs
            else:    
                logger.info(f"load_from_pdf_directory_with_metadata: Не используем кэш")
        
        logger.info(f"📁 Загрузка PDF из: {directory} с метаданными")
        
        # Загружаем документы с метаданными
        documents = self.document_loader.load_directory_with_metadata(
            directory, "**/*.pdf", recursive, force_reload
        )
        
        if not documents:
            logger.warning("Не найдено PDF документов")
            return {}
        
        # Опциональное разделение на чанки
        if chunk_size:
            splitter = TextSplitter(chunk_size, chunk_overlap or 50)
            documents = splitter.split_documents(documents)
        
        # Преобразование в словарь с сохранением метаданных
        result = {}
        for doc in documents:
            source = Path(doc.metadata.get('source', 'unknown')).stem
            page = doc.metadata.get('page_number', 0)
            
            if chunk_size:
                doc_id = f"{source}_page_{page}"
            else:
                doc_id = source
            
            # Сохраняем текст и метаданные
            result[doc_id] = {
                'text': doc.page_content,
                'metadata': {
                    'source_file': doc.metadata.get('filename', source),
                    'source_path': doc.metadata.get('filepath', ''),
                    'author': doc.metadata.get('doc_author', 'Неизвестен'),
                    'title': doc.metadata.get('doc_title', source),
                    'subject': doc.metadata.get('doc_subject', ''),
                    'keywords': doc.metadata.get('doc_keywords', ''),
                    'page_number': doc.metadata.get('page_number', 0),
                    'total_pages': doc.metadata.get('total_pages', 0),
                    'chunk_id': doc.metadata.get('chunk_id', 0) if chunk_size else None
                }
            }
        
        # Сохраняем в кэш
        self.cache.save(result, directory, recursive, chunk_size, {
            'num_pages': len(documents),
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'has_metadata': True
        })
        
        logger.info(f"✅ Загружено {len(result)} документов/чанков с метаданными")
        return result
    
    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """
        Получение метаданных документа по ID
        
        Args:
            doc_id: ID документа
        
        Returns:
            Словарь с метаданными
        """
        if doc_id in self.documents_metadata:
            return self.documents_metadata[doc_id]
        return {}
    
    def search_by_author(self, author: str) -> List[str]:
        """
        Поиск документов по автору
        
        Args:
            author: Имя автора
        
        Returns:
            Список ID документов
        """
        results = []
        for doc_id, data in self.documents_metadata.items():
            if author.lower() in data.get('metadata', {}).get('author', '').lower():
                results.append(doc_id)
        return results
    
    def get_statistics_by_author(self) -> Dict[str, int]:
        """
        Получение статистики по авторам
        
        Returns:
            Словарь {author: количество документов}
        """
        stats = {}
        for doc_id, data in self.documents_metadata.items():
            author = data.get('metadata', {}).get('author', 'Неизвестен')
            stats[author] = stats.get(author, 0) + 1
        return stats
    
    def clear_cache(self, directory: Path = None):
        """
        Очистка кэша для документов
        
        Args:
            directory: Если указан, очищает кэш только для этой директории
        """
        self.cache.clear(directory)
        logger.info(f"Кэш очищен для {directory if directory else 'всех директорий'}")    

    def compute_stats(self, documents: Dict[str, Any]) -> CorpusStats:
        """
        Вычисление статистики по корпусу документов

        Args:
            documents: Словарь документов. Поддерживает два формата:
                - {doc_id: text_string}
                - {doc_id: {'text': text_string, 'metadata': {...}}}
        """
        # Извлекаем тексты в зависимости от формата
        texts = []
        for doc_id, doc_content in documents.items():
            if isinstance(doc_content, dict):
                # Формат с метаданными
                text = doc_content.get('text', '')
                texts.append(text)
            elif isinstance(doc_content, str):
                # Простой формат
                texts.append(doc_content)
            else:
                # Неизвестный формат
                texts.append(str(doc_content))

        doc_lengths = [len(text) for text in texts]

        # Собираем уникальные слова из первых 100 документов для статистики
        all_words = set()
        for text in texts[:100]:
            try:
                words = text.lower().split()
                all_words.update(words)
            except Exception as e:
                logger.warning(f"Ошибка обработки текста для статистики: {e}")
                continue
            
        stats = CorpusStats(
            num_documents=len(documents),
            total_chars=sum(doc_lengths),
            avg_doc_length=np.mean(doc_lengths) if doc_lengths else 0,
            min_doc_length=min(doc_lengths) if doc_lengths else 0,
            max_doc_length=max(doc_lengths) if doc_lengths else 0,
            unique_terms=len(all_words)
        )

        logger.info("Статистика корпуса вычислена")
        return stats
    
class TestDataLoader:
    """Загрузчик тестовых данных"""
    
    def __init__(self, data_dir: Path = Path("data/tests")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)


    def load_from_json(self, json_path: Path) -> Dict[str, TestSample]:
        """
        Загрузка тестовых примеров из JSON файла
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            test_samples = {}

            # Формат: словарь с ключами q000, q001, ...
            if isinstance(data, dict):
                for qid, item in data.items():
                    # Проверяем, что item - словарь
                    if isinstance(item, dict):
                        # Извлекаем данные
                        query = item.get('query', '')
                        relevant_docs = item.get('relevant_docs', [])
                        reference_answer = item.get('reference_answer', '')

                        # Создаем TestSample с query_id
                        test_samples[qid] = TestSample(
                            query_id=qid,  # <-- ВАЖНО: передаем query_id
                            query=query,
                            relevant_docs=relevant_docs,
                            reference_answer=reference_answer
                        )

            elif isinstance(data, list):
                for item in data:
                    qid = item.get('query_id', f"q{len(test_samples):03d}")
                    test_samples[qid] = TestSample(
                        query_id=qid,  # <-- ВАЖНО: передаем query_id
                        query=item.get('query', ''),
                        relevant_docs=item.get('relevant_docs', []),
                        reference_answer=item.get('reference_answer', '')
                    )

            logger.info(f"✅ Загружено {len(test_samples)} тестовых примеров")

            # Выводим первый пример для проверки
            if test_samples:
                first_qid = list(test_samples.keys())[0]
                first = test_samples[first_qid]
                logger.info(f"   Первый пример: {first_qid}")
                logger.info(f"   Вопрос: {first.query[:100]}...")
                logger.info(f"   Релевантные документы: {first.relevant_docs}")

            return test_samples

        except Exception as e:
            logger.error(f"Ошибка загрузки JSON: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def create_sample_dataset(self, num_samples: int = 5) -> Dict[str, TestSample]:
        """Создание примера тестового датасета"""
        sample_queries = [
            {
                'query': "Что такое нейросети и как они работают?",
                'relevant_docs': ["doc_1"],
                'reference_answer': "Нейросети - это математические модели, вдохновленные структурой человеческого мозга.",
                'category': 'technical',
                'difficulty': 0.5
            },
            {
                'query': "Какие преимущества у RAG перед обычными LLM?",
                'relevant_docs': ["doc_3"],
                'reference_answer': "RAG позволяет использовать актуальную информацию из внешних источников, уменьшает галлюцинации.",
                'category': 'technical',
                'difficulty': 0.6
            },
        ]
        
        samples = {}
        for i, q in enumerate(sample_queries[:num_samples]):
            qid = f"sample_q{i+1}"
            samples[qid] = TestSample(
                query_id=qid,
                query=q['query'],
                relevant_docs=q['relevant_docs'],
                reference_answer=q['reference_answer'],
                category=q['category'],
                difficulty=q['difficulty']
            )
        
        self.save_to_json(samples, self.data_dir / "sample_testset.json")
        logger.info(f"Создано {len(samples)} тестовых примеров")
        return samples
    
    def save_to_json(self, samples: Dict[str, TestSample], filepath: Path):
        """Сохранение тестового набора в JSON"""
        data = {}
        for qid, sample in samples.items():
            data[qid] = {
                'query': sample.query,
                'relevant_docs': sample.relevant_docs,
                'reference_answer': sample.reference_answer,
                'category': sample.category,
                'difficulty': sample.difficulty
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Тестовый набор сохранен в {filepath}")



def main(args=None):
    import argparse
    from config import data_config, model_config    
    import time


    # Пример использования с конфигурацией
    from config import get_config_summary, data_config, model_config

    parser = argparse.ArgumentParser(description="Запуск загрузчика")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Директория с PDF документами")

    # Если передан список аргументов, используем его, иначе берем из sys.argv
    if args is not None:
        parsed_args = parser.parse_args(args)  # Используем другую переменную
    else:
        parsed_args = parser.parse_args()

    print(get_config_summary())


    print("=" * 60)
    print("Тестирование загрузчика с кэшированием")
    print("=" * 60)
    
    loader = CorpusLoader()
    pdf_dir = Path(parsed_args.pdf_dir)
    
    if pdf_dir.exists():
        # Очищаем кэш для чистого теста
        print("\n🧹 Очистка кэша...")
        loader.clear_cache(pdf_dir)
        
        # ПЕРВЫЙ ЗАПУСК - загрузка из PDF (создание кэша)
        print("\n📁 ПЕРВЫЙ ЗАПУСК - загрузка из PDF (создание кэша)...")
        start_time = time.time()
        
        documents = loader.load_from_pdf_directory_with_metadata(
            pdf_dir,
            recursive=True,
            chunk_size=500,
            chunk_overlap=50,
            force_reload=True  # Принудительная загрузка
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Загружено {len(documents)} документов за {elapsed:.2f} сек")


        first_load_time = time.time() - start_time
        print(f"✅ Загружено {len(documents)} документов/чанков за {first_load_time:.2f} сек")
        
        # ВТОРОЙ ЗАПУСК - загрузка из кэша
        print("\n📦 ВТОРОЙ ЗАПУСК - загрузка из кэша...")
        start_time = time.time()
        
        documents = loader.load_from_pdf_directory_with_metadata(
            pdf_dir,
            recursive=True,
            chunk_size=500,
            chunk_overlap=50,
            force_reload=False  # Используем кэш
        )
        
        second_load_time = time.time() - start_time
        print(f"✅ Загружено {len(documents)} документов/чанков за {second_load_time:.2f} сек")
        
        # Сравнение времени
        print(f"\n⏱️  Сравнение времени:")
        print(f"  Первая загрузка (PDF): {first_load_time:.2f} сек")
        print(f"  Вторая загрузка (кэш): {second_load_time:.2f} сек")

        if second_load_time < 0.001:
            print(f"  ℹ️  Вторая загрузка выполнена очень быстро (< 0.001 сек)")        
        elif second_load_time < first_load_time:
            speedup = first_load_time / second_load_time
            print(f"  🚀 Ускорение: {speedup:.1f}x")
        else:
            print(f"  ⚠️ Кэш не сработал! Время одинаковое")
        
        
        # Статистика
        if documents:
            stats = loader.compute_stats(documents)
            print(stats)
        
        # Пример документа
        if documents:
            sample_id = list(documents.keys())[0]
            sample_data = documents[sample_id]
            print(f"\n📄 Пример документа ({sample_id}):")

            # Извлекаем текст в зависимости от формата
            if isinstance(sample_data, dict):
                text = sample_data.get('text', '')
                if text:
                    print(f"  {text[:200]}...")
                else:
                    print(f"  (Нет текста)")
            elif isinstance(sample_data, str):
                print(f"  {sample_data[:200]}...")
            else:
                print(f"  Тип данных: {type(sample_data)}")        

        # Вывод информации о первых 5 документах
        print("\n📊 МЕТАДАННЫЕ ДОКУМЕНТОВ:")
        print("-" * 60)

        for doc_id, data in list(documents.items())[:5]:
            print(f"\n📄 Документ: {doc_id}")

            if isinstance(data, dict):
                # Формат с метаданными
                if 'metadata' in data:
                    metadata = data['metadata']
                    text = data.get('text', '')
                    print(f"  Название: {metadata.get('title', 'Не указано')}")
                    print(f"  Автор: {metadata.get('author', 'Не указан')}")
                    print(f"  Файл: {metadata.get('source_file', 'Неизвестно')}")
                    print(f"  Страниц: {metadata.get('total_pages', 0)}")
                    print(f"  Размер текста: {len(text):,} символов")
                else:
                    # Формат без метаданных, но с текстом в поле 'text'
                    text = data.get('text', '')
                    print(f"  Размер текста: {len(text):,} символов")
                    if text:
                        print(f"  Начало текста: {text[:100]}...")
            elif isinstance(data, str):
                # Простой формат - строка текста
                print(f"  Размер текста: {len(data):,} символов")
                print(f"  Начало текста: {data[:100]}...")
            else:
                print(f"  Тип данных: {type(data)}")
                print(f"  Размер: {len(str(data)):,} символов")
        
    else:
        print(f"❌ Директория не найдена: {pdf_dir}")
    
    print("\n✅ Тестирование завершено")    

if __name__ == "__main__":

    # После настройки логирования
    import logging

    print("\n" + "="*60)
    print("СОСТОЯНИЕ ВСЕХ ЛОГГЕРОВ:")
    print("="*60)

    root_logger = logging.getLogger()
    print(f"Корневой логгер: {root_logger.name}")
    print(f"  Уровень: {logging.getLevelName(root_logger.getEffectiveLevel())}")
    print(f"  Обработчиков: {len(root_logger.handlers)}")

    # Проверяем всех потомков
    for name in logging.root.manager.loggerDict:
        logger_obj = logging.getLogger(name)
        if logger_obj.level != 0:  # 0 означает не установлен (используется родительский)
            print(f"Логгер '{name}': уровень = {logging.getLevelName(logger_obj.level)}")

    #main()
    #main(["--pdf_dir", "data/domain_3_WikiEval_1row/books"])        
    #main(["--pdf_dir", "data/domain_4_WikiEval_2row/books"])        
    main(["--pdf_dir", "data/domain_7_UAV/books"])        
    