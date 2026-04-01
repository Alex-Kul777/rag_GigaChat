"""
config.py - Централизованная конфигурация для RAG системы
"""
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import List, Optional, Dict
import torch

import os
from dotenv import load_dotenv

#№env_path = Path(".env")
#load_dotenv(env_path)
load_dotenv()


@dataclass
class GigaChatConfig:
    """Конфигурация GigaChat"""
    api_key: str = os.getenv("GIGACHAT_API_KEY", "")
    enabled: bool = True
    model: str = "GigaChat-2-Max"
    scope: str = "GIGACHAT_API_B2B"
    verify_ssl_certs: bool = False
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3

# Добавьте глобальный экземпляр
gigachat_config = GigaChatConfig()

@dataclass
class ModelConfig:
    """Конфигурация моделей"""
    
    # LLM модели
    #llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    #llm_model_name: str = "ai-forever/rugpt3small_based_on_gpt2"
    llm_model_name: str = "GigaChat-2-Max"
    
    # Embedding модели
    #embedding_model_name: str = "intfloat/multilingual-e5-base"
    #embedding_model_name: str = "intfloat/multilingual-e5-small"
    embedding_model_name: str = "GigaChat-2-Max"
    
    # Параметры генерации
    max_new_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Параметры поиска
    default_k_retrieve: int = 5
    #default_k_retrieve: int = 3
    max_context_length: int = 2000
    
    # Устройство
    device: str = "cpu"  # Принудительно CPU для экономии памяти
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Режим работы
    mode: str = "ui"
    use_retriever: bool = True
    
    # Квантование (для экономии памяти)
    use_8bit_quantization: bool = False
    use_4bit_quantization: bool = False


@dataclass
class DataConfig:
    """Конфигурация данных"""
    
    # Директории
    data_dir: Path = Path("data")
    corpus_dir: Path = Path("data/corpus")
    cache_dir: Path = Path("data/cache")
    vectorstore_dir: Path = Path("data/vectorstore")
    experiments_dir: Path = Path("experiments")
    logs_dir: Path = Path("logs")
    metadata_dir: Path = Path("data/metadata")
    
      
    # 🆕 Или можно добавить несколько путей для разных доменов
    documents_dirs: Dict[str, Path] = field(default_factory=lambda: {
        "debug": Path("data/domain_2_Debug/books"),
        "ai": Path("data/domain_1_AI/books"),
        "test": Path("data/test_docs")
    })

    # Параметры загрузки PDF
    pdf_max_pages: Optional[int] = 1000  # Максимум страниц на PDF (None = без ограничений)
    pdf_max_doc_size: int = 50000000  # Максимальный размер документа в символах
    pdf_method: str = "auto"  # "auto", "pypdf2", "pdfplumber"
    
    # Параметры разделения на чанки
    chunk_size: int = 800
    chunk_overlap: int = 100
    chunk_separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ".", "!", "?", ",", " ", ""])
    
    # Параметры кэширования
    cache_enabled: bool = True
    #force_reload: bool = True
    force_reload: bool = False
    
    # Форматы файлов
    supported_extensions: List[str] = field(default_factory=lambda: ['.pdf', '.txt', '.json', '.csv'])
    
    def __post_init__(self):
        """Создание директорий после инициализации"""
        for dir_path in [self.data_dir, self.corpus_dir, self.cache_dir, 
                         self.vectorstore_dir, self.experiments_dir, 
                         self.logs_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class VectorStoreConfig:
    """Конфигурация векторного хранилища"""
    
    # Тип хранилища
    vector_store_type: str = "faiss"  # только faiss
    
    # Параметры FAISS
    faiss_index_type: str = "flat"  # "flat", "ivf", "hnsw"
    faiss_nlist: int = 100  # для IVF индекса
    faiss_nprobe: int = 10  # для поиска
    
    # Сохранение
    persist_dir: Path = Path("data/vectorstore")
    save_on_update: bool = True


@dataclass
class ExperimentConfig:
    """Конфигурация экспериментов"""
    
    # Параметры оценки
    ks_eval: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    batch_size: int = 32
    
    # Сохранение результатов
    save_results: bool = True
    save_detailed_predictions: bool = True
    
    # Логирование
    detailed_logging: bool = True
    log_level: str = "INFO"


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    
    #log_level: str = "INFO"
    log_level: str = "DEBUG"
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_date_format: str = '%Y-%m-%d %H:%M:%S'
    log_to_file: bool = True
    log_to_console: bool = True
    log_file_name: str = "logs/rag_app.log"


# Глобальные экземпляры конфигураций
model_config = ModelConfig()
data_config = DataConfig()
vectorstore_config = VectorStoreConfig()
experiment_config = ExperimentConfig()
logging_config = LoggingConfig()

# Список модулей, для которых нужно показывать логи
OUR_MODULES = [
    'excel_reporter', 'experiment', 'evaluator', 
    'data_loader', 'rag_core', 'models', 'config'
]

# Список библиотек, логи которых нужно подавить
SILENCE_LIBRARIES = [
    'matplotlib', 'datasets', 'ragas', 'asyncio',
    'urllib3', 'requests', 'huggingface_hub', 'filelock',
    'PIL', 'PIL.PngImagePlugin', 'fontTools', 'PdfReader'
]


def configure_logging():
    """Настройка логирования с подавлением логов сторонних библиотек"""
    # Получаем уровень из конфигурации
    log_level = getattr(logging, logging_config.log_level.upper(), logging.INFO)
    
    # Устанавливаем уровень для корневого логгера ИЗ КОНФИГУРАЦИИ
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)  # ← ИСПРАВЛЕНО: используем log_level вместо WARNING
    
    # Очищаем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Настраиваем обработчики
    formatter = logging.Formatter(
        logging_config.log_format,
        logging_config.log_date_format
    )
    
    log_level = getattr(logging, logging_config.log_level.upper(), logging.INFO)
    
    # Консольный обработчик (только для наших модулей)
    if logging_config.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(ModuleFilter(OUR_MODULES))
        root_logger.addHandler(console_handler)
    
    # Файловый обработчик (все логи)
    if logging_config.log_to_file:
        log_file = Path(logging_config.log_file_name)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Подавляем логи от сторонних библиотек
    for lib in SILENCE_LIBRARIES:
        logging.getLogger(lib).setLevel(logging.WARNING)


class ModuleFilter(logging.Filter):
    """Фильтр для показа логов только из указанных модулей"""
    def __init__(self, modules):
        self.modules = modules
    
    def filter(self, record):
        # Показываем логи только из наших модулей
        for module in self.modules:
            if record.name.startswith(module):
                return True
        # Подавляем все остальные
        return False


# Автоматически настраиваем логирование при импорте
configure_logging()

def update_config_from_args(args):
    """Обновление конфигурации из аргументов командной строки"""
    if hasattr(args, 'model_name') and args.model_name:
        model_config.llm_model_name = args.model_name
    
    if hasattr(args, 'embedding_model') and args.embedding_model:
        model_config.embedding_model_name = args.embedding_model
    
    if hasattr(args, 'k_retrieve') and args.k_retrieve:
        model_config.default_k_retrieve = args.k_retrieve
    
    if hasattr(args, 'chunk_size') and args.chunk_size:
        data_config.chunk_size = args.chunk_size
    
    if hasattr(args, 'chunk_overlap') and args.chunk_overlap:
        data_config.chunk_overlap = args.chunk_overlap
    
    if hasattr(args, 'force_reload') and args.force_reload:
        data_config.force_reload = args.force_reload


def get_config_summary() -> str:
    """Получение сводки конфигурации"""
    summary = f"""
    ==================================================
    КОНФИГУРАЦИЯ RAG СИСТЕМЫ
    ==================================================
    
    МОДЕЛИ:
      LLM: {model_config.llm_model_name}
      Embeddings: {model_config.embedding_model_name}
      Device: {model_config.device}
      Max tokens: {model_config.max_new_tokens}
      Temperature: {model_config.temperature}
    
    ДАННЫЕ:
      Data dir: {data_config.data_dir}
      Cache dir: {data_config.cache_dir}
      Chunk size: {data_config.chunk_size}
      Chunk overlap: {data_config.chunk_overlap}
      PDF max pages: {data_config.pdf_max_pages}
    
    ВЕКТОРНОЕ ХРАНИЛИЩЕ:
      Type: {vectorstore_config.vector_store_type}
      Persist dir: {vectorstore_config.persist_dir}
    
    ЭКСПЕРИМЕНТЫ:
      Ks eval: {experiment_config.ks_eval}
      Save results: {experiment_config.save_results}
    ==================================================
    """
    return summary


# Экспорт для удобного импорта
__all__ = [
    'model_config',
    'data_config', 
    'vectorstore_config',
    'experiment_config',
    'logging_config',
    'update_config_from_args',
    'get_config_summary'
]