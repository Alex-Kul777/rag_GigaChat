"""RAG GigaChat - Retrieval-Augmented Generation System with GigaChat LLM"""

__version__ = "1.0.0"

# Public API - основные классы и конфигурация
from .config import (
    model_config,
    data_config,
    vectorstore_config,
    experiment_config,
    logging_config,
    gigachat_config,
)
from .core.rag_pipeline import RAGPipeline, VectorStoreManager, LLMManager
from .data.data_loader import CorpusLoader, DocumentLoader, TestDataLoader
from .token_counter import TokenCounter
from .models import RetrievalType, RetrievalResult, GenerationResult

__all__ = [
    "RAGPipeline",
    "VectorStoreManager",
    "LLMManager",
    "TokenCounter",
    "CorpusLoader",
    "DocumentLoader",
    "TestDataLoader",
    "model_config",
    "data_config",
    "vectorstore_config",
    "experiment_config",
    "logging_config",
    "gigachat_config",
    "RetrievalType",
    "RetrievalResult",
    "GenerationResult",
]
