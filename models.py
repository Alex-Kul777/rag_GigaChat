"""
models.py - Модели данных для RAG системы
Содержит dataclasses и Enum для типизации данных
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime


class RetrievalType(str, Enum):
    """Типы методов поиска"""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    
    def __str__(self):
        return self.value
    
    @property
    def description(self) -> str:
        """Описание метода поиска"""
        descriptions = {
            "dense": "Векторный поиск на основе эмбеддингов (FAISS)",
            "sparse": "Лексический поиск на основе BM25",
            "hybrid": "Гибридный поиск, объединяющий dense и sparse методы"
        }
        return descriptions[self.value]


class ExperimentStatus(str, Enum):
    """Статусы эксперимента"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Document:
    """Модель документа"""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Валидация после инициализации"""
        if not self.doc_id or not self.text:
            raise ValueError("doc_id и text обязательны")
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'doc_id': self.doc_id,
            'text': self.text,
            'metadata': self.metadata
        }
    
    def preview(self, length: int = 100) -> str:
        """Предпросмотр текста документа"""
        return self.text[:length] + "..." if len(self.text) > length else self.text


@dataclass
class RetrievalResult:
    """Результат поиска"""
    query_id: str
    query_text: str
    retrieved_docs: List[Dict[str, Any]]  # список {doc_id, score, text}
    scores: List[float]
    retrieval_time: float = 0.0
    retrieval_type: Optional[RetrievalType] = None
    
    def __post_init__(self):
        """Инициализация после создания"""
        if not self.scores and self.retrieved_docs:
            self.scores = [doc.get('score', 0.0) for doc in self.retrieved_docs]
    
    @property
    def top_doc(self) -> Optional[Dict[str, Any]]:
        """Самый релевантный документ"""
        return self.retrieved_docs[0] if self.retrieved_docs else None
    
    def top_k_docs(self, k: int = 3) -> List[Dict[str, Any]]:
        """Топ-K документов"""
        return self.retrieved_docs[:k]
    
    def get_doc_ids(self) -> List[str]:
        """Получить список ID документов"""
        return [doc['doc_id'] for doc in self.retrieved_docs]
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'query_id': self.query_id,
            'query_text': self.query_text,
            'retrieved_docs': self.retrieved_docs,
            'scores': self.scores,
            'retrieval_time': self.retrieval_time,
            'retrieval_type': self.retrieval_type.value if self.retrieval_type else None
        }


@dataclass
class GenerationResult:
    """Результат генерации"""
    query_id: str
    query_text: str
    context: str
    answer: str
    retrieval_results: RetrievalResult
    generation_time: float = 0.0
    prompt: Optional[str] = None
    tokens_generated: int = 0
    
    @property
    def full_response(self) -> str:
        """Полный ответ с контекстом (для отладки)"""
        return f"""
        Вопрос: {self.query_text}
        
        Контекст:
        {self.context}
        
        Ответ:
        {self.answer}
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'query_id': self.query_id,
            'query_text': self.query_text,
            'context': self.context,
            'answer': self.answer,
            'retrieval_results': self.retrieval_results.to_dict(),
            'generation_time': self.generation_time,
            'tokens_generated': self.tokens_generated
        }


@dataclass
class TestSample:
    """Тестовый пример для экспериментов"""
    query_id: str
    query: str
    relevant_docs: List[str]  # ID релевантных документов
    reference_answer: Optional[str] = None  # Эталонный ответ
    context: Optional[str] = None  # Контекст для генерации
    category: Optional[str] = None  # Категория вопроса (для анализа)
    difficulty: Optional[float] = None  # Сложность вопроса (0-1)
    
    def __post_init__(self):
        """Валидация после инициализации"""
        if not self.query_id or not self.query:
            raise ValueError("query_id и query обязательны")
        if not isinstance(self.relevant_docs, list):
            self.relevant_docs = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'query_id': self.query_id,
            'query': self.query,
            'relevant_docs': self.relevant_docs,
            'reference_answer': self.reference_answer,
            'category': self.category,
            'difficulty': self.difficulty
        }


@dataclass
class RetrievalMetrics:
    """Метрики качества поиска (совместимый с evaluator.py формат)"""
    map: float = 0.0
    mrr: float = 0.0
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для JSON"""
        return {
            'map': self.map,
            'mrr': self.mrr,
            'precision_at_k': self.precision_at_k,
            'recall_at_k': self.recall_at_k,
            'ndcg_at_k': self.ndcg_at_k
        }
    
    def summary(self) -> str:
        """Краткое текстовое представление метрик"""
        lines = [
            f"MAP: {self.map:.4f}",
            f"MRR: {self.mrr:.4f}"
        ]
        
        for k, v in self.precision_at_k.items():
            lines.append(f"P@{k}: {v:.4f}")
        
        for k, v in self.recall_at_k.items():
            lines.append(f"R@{k}: {v:.4f}")
        
        return "\n".join(lines)


@dataclass
class GenerationMetrics:
    """Метрики качества генерации"""
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bleu: float = 0.0
    bert_score: float = 0.0
    semantic_similarity: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Преобразование в словарь"""
        return {
            'rouge1': self.rouge1,
            'rouge2': self.rouge2,
            'rougeL': self.rougeL,
            'bleu': self.bleu,
            'bert_score': self.bert_score,
            'semantic_similarity': self.semantic_similarity
        }
    
    def summary(self) -> str:
        """Краткое текстовое представление метрик"""
        return f"""
        ROUGE-1: {self.rouge1:.4f}
        ROUGE-2: {self.rouge2:.4f}
        ROUGE-L: {self.rougeL:.4f}
        BLEU: {self.bleu:.4f}
        BERTScore: {self.bert_score:.4f}
        Semantic Similarity: {self.semantic_similarity:.4f}
        """


@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента"""
    experiment_name: str
    retrieval_type: RetrievalType
    k_retrieve: int
    ks_eval: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    use_reranker: bool = False
    reranker_model_name: Optional[str] = None
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    batch_size: int = 32
    device: str = "cpu"
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model_name: str = "intfloat/multilingual-e5-base"
    llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 200
    temperature: float = 0.7
    
    def __post_init__(self):
        """Валидация конфигурации"""
        if self.retrieval_type == RetrievalType.HYBRID:
            if self.dense_weight + self.sparse_weight != 1.0:
                # Нормализация весов
                total = self.dense_weight + self.sparse_weight
                if total > 0:
                    self.dense_weight /= total
                    self.sparse_weight /= total
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'experiment_name': self.experiment_name,
            'retrieval_type': self.retrieval_type.value,
            'k_retrieve': self.k_retrieve,
            'ks_eval': self.ks_eval,
            'use_reranker': self.use_reranker,
            'reranker_model_name': self.reranker_model_name,
            'dense_weight': self.dense_weight,
            'sparse_weight': self.sparse_weight,
            'batch_size': self.batch_size,
            'device': self.device,
            'embedding_model_name': self.embedding_model_name,
            'llm_model_name': self.llm_model_name,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature
        }


@dataclass
class ExperimentResult:
    """Результат эксперимента"""
    experiment_id: str
    timestamp: str
    config: Dict[str, Any]
    retrieval_metrics: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    detailed_predictions: Dict[str, Any]
    execution_time: float
    status: ExperimentStatus = ExperimentStatus.COMPLETED
    
    error_message: Optional[str] = None
    advanced_metrics: Optional[Dict[str, Any]] = None  # Добавлено поле   
    
    def __post_init__(self):
        """Инициализация после создания"""
        if isinstance(self.status, str):
            self.status = ExperimentStatus(self.status)

        if self.advanced_metrics is None:
            self.advanced_metrics = {}            
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь (для JSON)"""
        return {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'config': self.config,
            'retrieval_metrics': self.retrieval_metrics,
            'generation_metrics': self.generation_metrics,
            'detailed_predictions': self.detailed_predictions,
            'execution_time': self.execution_time,
            'status': self.status.value if hasattr(self.status, 'value') else str(self.status),
            'error_message': self.error_message,
            'advanced_metrics': self.advanced_metrics if self.advanced_metrics else {}           
        }
    
    @property
    def map_score(self) -> float:
        """MAP score (для быстрого доступа)"""
        return self.retrieval_metrics.get('map', 0.0)
    
    @property
    def mrr_score(self) -> float:
        """MRR score (для быстрого доступа)"""
        return self.retrieval_metrics.get('mrr', 0.0)
    
    def summary(self) -> str:
        """Краткое текстовое представление результатов"""
        lines = [
            f"Эксперимент: {self.experiment_id}",
            f"Статус: {self.status.value}",
            f"Время выполнения: {self.execution_time:.2f} сек",
            f"\nМетрики поиска:",
            f"  MAP: {self.map_score:.4f}",
            f"  MRR: {self.mrr_score:.4f}"
        ]
        
        # Precision@k
        precisions = self.retrieval_metrics.get('precision_at_k', {})
        if precisions:
            lines.append("  Precision@k:")
            for k, v in precisions.items():
                lines.append(f"    P@{k}: {v:.4f}")
        
        # Recall@k
        recalls = self.retrieval_metrics.get('recall_at_k', {})
        if recalls:
            lines.append("  Recall@k:")
            for k, v in recalls.items():
                lines.append(f"    R@{k}: {v:.4f}")
        
        # Метрики генерации
        if self.generation_metrics:
            lines.append(f"\nМетрики генерации:")
            for metric, value in self.generation_metrics.items():
                lines.append(f"  {metric}: {value:.4f}")
        
        return "\n".join(lines)


@dataclass
class QueryRequest:
    """Запрос на обработку"""
    query: str
    k: int = 5
    retrieval_type: Optional[RetrievalType] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    
    def validate(self) -> bool:
        """Валидация запроса"""
        if not self.query or not isinstance(self.query, str):
            return False
        if self.k < 1 or self.k > 100:
            return False
        if self.temperature is not None and (self.temperature < 0 or self.temperature > 2):
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'query': self.query,
            'k': self.k,
            'retrieval_type': self.retrieval_type.value if self.retrieval_type else None,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }


@dataclass
class QueryResponse:
    """Ответ на запрос"""
    query: str
    answer: str
    retrieved_docs: List[Dict[str, Any]]
    context: str
    retrieval_time: float
    generation_time: float
    total_time: float
    tokens_generated: int
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'query': self.query,
            'answer': self.answer,
            'retrieved_docs': self.retrieved_docs,
            'context': self.context,
            'retrieval_time': self.retrieval_time,
            'generation_time': self.generation_time,
            'total_time': self.total_time,
            'tokens_generated': self.tokens_generated,
            'success': self.success,
            'error_message': self.error_message
        }
    
    @property
    def summary(self) -> str:
        """Краткое представление ответа"""
        return f"""
        Вопрос: {self.query}
        Ответ: {self.answer[:200]}...
        Время обработки: {self.total_time:.2f} сек
        Найдено документов: {len(self.retrieved_docs)}
        """


# Вспомогательные функции
def create_test_sample_from_dict(data: Dict[str, Any]) -> TestSample:
    """Создание TestSample из словаря"""
    return TestSample(
        query_id=data.get('query_id', ''),
        query=data.get('query', ''),
        relevant_docs=data.get('relevant_docs', []),
        reference_answer=data.get('reference_answer'),
        context=data.get('context'),
        category=data.get('category'),
        difficulty=data.get('difficulty')
    )


def create_experiment_config_from_args(args) -> ExperimentConfig:
    """Создание конфигурации эксперимента из аргументов командной строки"""
    return ExperimentConfig(
        experiment_name=args.experiment_name,
        retrieval_type=RetrievalType(args.retrieval_type),
        k_retrieve=args.k_retrieve,
        ks_eval=[1, 3, 5, 10],
        use_reranker=args.use_reranker if hasattr(args, 'use_reranker') else False,
        dense_weight=args.dense_weight if hasattr(args, 'dense_weight') else 0.5,
        sparse_weight=args.sparse_weight if hasattr(args, 'sparse_weight') else 0.5,
        device=args.device if hasattr(args, 'device') else "cpu"
    )