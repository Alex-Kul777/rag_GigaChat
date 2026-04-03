"""
evaluator.py - Модуль для оценки качества RAG системы
Содержит классы для оценки поиска и генерации ответов
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import pandas as pd

# Для метрик генерации
try:
    from rouge_score import rouge_scorer
    from bert_score import BERTScorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score bert-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not installed. Install with: pip install nltk")

# Импорты для RAGAS
try:
    import asyncio    
    from tqdm import tqdm    
    from ragas.dataset_schema import SingleTurnSample    
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRelevance
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: ragas not installed. Install with: pip install ragas")

from config import data_config, model_config, logging_config

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Метрики качества поиска"""
    map: float = 0.0
    mrr: float = 0.0
    precision_at_k: Dict[int, float] = None
    recall_at_k: Dict[int, float] = None
    ndcg_at_k: Dict[int, float] = None
    
    def __post_init__(self):
        if self.precision_at_k is None:
            self.precision_at_k = {}
        if self.recall_at_k is None:
            self.recall_at_k = {}
        if self.ndcg_at_k is None:
            self.ndcg_at_k = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'map': self.map,
            'mrr': self.mrr,
            'precision_at_k': self.precision_at_k,
            'recall_at_k': self.recall_at_k,
            'ndcg_at_k': self.ndcg_at_k
        }

@dataclass
class AdvancedMetrics:
    """Продвинутые метрики RAG"""

    """Продвинутые метрики RAG (RAGAS)"""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_relevancy: float = 0.0
    detailed_results: Dict[str, Any] = None

    
    def __post_init__(self):
        if self.detailed_results is None:
            self.detailed_results = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'faithfulness': self.faithfulness,
            'answer_relevancy': self.answer_relevancy,
            'context_relevancy': self.context_relevancy,
            'detailed_results': self.detailed_results
        }

class WikiEvalEvaluator:
    def __init__(self, pipeline, giga_api_key: str):
        # Отложенный импорт
        from rag_core import RAGPipeline
        
        # Проверка типа (опционально)
        if not isinstance(pipeline, RAGPipeline):
            raise TypeError(f"Expected RAGPipeline, got {type(pipeline)}")
        
        self.pipeline = pipeline
        self.giga_api_key = giga_api_key
       
        self.token_counter = pipeline.token_counter  # Используем счетчик из пайплайна
    
    def evaluate_batch(self, dataset, k: int = 3, limit: int = None) -> pd.DataFrame:
        results = []
        
        for i, item in enumerate(dataset):
            if limit and i >= limit:
                break
            
            # Сохраняем количество токенов до запроса
            tokens_before = self.token_counter.total_tokens
            
            try:
                eval_result = self.evaluate_single(item, k=k)
                results.append(eval_result)
                
                # Добавляем информацию о токенах в результат
                tokens_used = self.token_counter.total_tokens - tokens_before
                eval_result['tokens_used'] = tokens_used
                
            except Exception as e:
                logger.error(f"Ошибка при обработке примера {i}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def print_token_statistics(self):
        """Вывод статистики токенов"""
        self.token_counter.print_summary()

class RAGEvaluator:
    """
    Класс для оценки RAG системы
    Поддерживает оценку:
    - Поиска (MAP, MRR, Precision@k, Recall@k, NDCG@k)
    - Генерации (ROUGE, BLEU, BERTScore)
    - Продвинутые метрики (Faithfulness, AnswerRelevancy, ContextRelevancy)
    """
    def __init__(self, 
                 use_bert_score: bool = False,
                 llm=None,
                 embeddings=None):
            
        """
        Инициализация оценщика
        
        Args:
            use_bert_score: Использовать ли BERTScore для оценки
            llm: LLM модель для RAGAS метрик (LangChain совместимая)
            embeddings: Эмбеддинги для RAGAS метрик            
        """
        self.use_bert_score = use_bert_score and ROUGE_AVAILABLE
        self.llm = llm
        self.embeddings = embeddings

        # Инициализируем ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
        
        # Инициализируем BERTScore
        if self.use_bert_score:
            try:
                self.bert_scorer = BERTScorer(
                    model_type='bert-base-multilingual-cased',
                    lang='ru',
                    device='cpu',
                    rescale_with_baseline=True
                )
            except Exception as e:
                logger.warning(f"Не удалось инициализировать BERTScore: {e}")
                self.use_bert_score = False
                self.bert_scorer = None
        else:
            self.bert_scorer = None

        # Инициализируем RAGAS метрики
        self._init_ragas_metrics()
   
    def _init_ragas_metrics(self):
        """Инициализация RAGAS метрик"""
        self.ragas_available = RAGAS_AVAILABLE and self.llm is not None
        logger.warning(f"Инициализация RAGAS метрик. self.ragas_available = {RAGAS_AVAILABLE} ")        
        if self.ragas_available:
            try:
                # Оборачиваем LLM и эмбеддинги для RAGAS
                self.ragas_llm = LangchainLLMWrapper(self.llm)
                self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings) if self.embeddings else None
                
                # Инициализируем метрики
                self.faithfulness_metric = Faithfulness(llm=self.llm)
                self.answer_relevancy_metric = AnswerRelevancy(llm=self.llm)
                self.context_relevancy_metric = ContextRelevance(llm=self.llm)
                
                logger.info("RAGAS метрики успешно инициализированы")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать RAGAS метрики: {e}")
                self.ragas_available = False
        else:
            if not RAGAS_AVAILABLE:
                logger.warning("ragas не установлен. Продвинутые метрики недоступны")
            elif self.llm is None:
                logger.warning("LLM не предоставлен. Продвинутые метрики недоступны")

    def evaluate_retrieval(self, predictions: Dict[str, List[str]], 
                            ground_truth: Dict[str, List[str]], 
                            ks: List[int] = [1, 3, 5, 10]) -> RetrievalMetrics:
        """
        Оценка качества поиска
        """
        map_scores = []
        mrr_scores = []
        precision_at_k = {k: [] for k in ks}
        recall_at_k = {k: [] for k in ks}
        ndcg_at_k = {k: [] for k in ks}

        for qid, pred_docs in predictions.items():
            if qid not in ground_truth:
                continue
            
            true_docs = ground_truth[qid]

            # Нормализация имен документов для сравнения
            normalized_pred = self._normalize_doc_names(pred_docs)
            normalized_true = self._normalize_doc_names(true_docs)

            # Преобразуем в множество для быстрого поиска
            relevant_set = set(normalized_true)

            # Отладочный вывод для проверки
            logger.debug(f"Query {qid}: релевантных документов = {len(relevant_set)}")
            logger.debug(f"  Релевантные: {relevant_set}")
            logger.debug(f"  Найденные: {normalized_pred[:5]}")

            # Вычисляем метрики
            ap = self._average_precision(normalized_pred, relevant_set)
            rr = self._reciprocal_rank(normalized_pred, relevant_set)

            map_scores.append(ap)
            mrr_scores.append(rr)

            for k in ks:
                p = self._precision_at_k(normalized_pred, relevant_set, k)
                r = self._recall_at_k(normalized_pred, relevant_set, k)
                ndcg = self._ndcg_at_k(normalized_pred, relevant_set, k)

                # Отладочный вывод
                logger.debug(f"  k={k}: P={p:.3f}, R={r:.3f}, NDCG={ndcg:.3f}")

                precision_at_k[k].append(p)
                recall_at_k[k].append(r)
                ndcg_at_k[k].append(ndcg)

        return RetrievalMetrics(
            map=sum(map_scores) / len(map_scores) if map_scores else 0,
            mrr=sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
            precision_at_k={k: sum(v)/len(v) if v else 0 for k, v in precision_at_k.items()},
            recall_at_k={k: sum(v)/len(v) if v else 0 for k, v in recall_at_k.items()},
            ndcg_at_k={k: sum(v)/len(v) if v else 0 for k, v in ndcg_at_k.items()}
        )

    def _normalize_doc_names(self, docs: List[str]) -> List[str]:
        """
        Нормализация имен документов для сопоставления

        Пример:
        - "PSLV_C56.pdf" -> "PSLV_C56"
        - "PSLV_C56_page_1" -> "PSLV_C56"
        - "PSLV_C56_page_2" -> "PSLV_C56"
        """
        normalized = []
        for doc in docs:
            # Удаляем расширение .pdf
            doc = doc.replace('.pdf', '')
            # Убираем суффиксы страниц
            if '_page_' in doc:
                doc = doc.split('_page_')[0]
            normalized.append(doc)
        return normalized    
    
    def _average_precision(self, retrieved: List[str], relevant: set) -> float:
        """
        Вычисление Average Precision для одного запроса
        AP всегда в диапазоне [0, 1]
        """
        if not relevant:
            return 0.0

        num_hits = 0
        sum_precisions = 0.0

        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                num_hits += 1
                sum_precisions += num_hits / i

        ap = sum_precisions / len(relevant) if relevant else 0.0
        return min(ap, 1.0)

    def _reciprocal_rank(self, retrieved: List[str], relevant: set) -> float:
        """
        Вычисление Reciprocal Rank для одного запроса
        RR всегда в диапазоне [0, 1]
        """
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return min(1.0 / i, 1.0)
        return 0.0

    def _precision_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """
        Вычисление Precision@k
        Precision@k всегда в диапазоне [0, 1]

        Формула: Precision@k = (число релевантных документов в топ-k) / min(k, len(retrieved))
        Если retrieved короче k, используем длину retrieved.
        """
        if k <= 0:
            return 0.0

        # Если нет предсказаний, возвращаем 0
        if not retrieved:
            return 0.0

        # Берем первые k элементов (или меньше, если retrieved короче)
        top_k = retrieved[:k]

        # Подсчитываем количество релевантных документов в топ-k
        hits = sum(1 for doc in top_k if doc in relevant)

        # Знаменатель - фактическое количество рассмотренных элементов
        denominator = min(k, len(retrieved))

        # Вычисляем precision
        precision = hits / denominator if denominator > 0 else 0.0

        # Ограничиваем значением 1.0
        return min(precision, 1.0)


    def _recall_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """
        Вычисление Recall@k
        Recall@k всегда в диапазоне [0, 1]
        """
        if not relevant:
            return 0.0

        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in relevant)
        recall = hits / len(relevant)
        return min(recall, 1.0)

    def _ndcg_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """
        Вычисление NDCG@k (Normalized Discounted Cumulative Gain)
        NDCG всегда в диапазоне [0, 1]

        Формула: 
        - DCG = sum(gain_i / log2(i+1))
        - IDCG = sum(ideal_gains / log2(i+1)), где ideal_gains = [1] * min(len(relevant), k)
        - NDCG = DCG / IDCG
        """
        if k <= 0 or not relevant:
            return 0.0

        # Вычисляем DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k], 1):
            gain = 1 if doc in relevant else 0
            dcg += gain / np.log2(i + 1)

        # Вычисляем IDCG (Ideal DCG)
        # В идеальном случае все релевантные документы находятся на первых позициях
        # Количество идеальных позиций = min(количество релевантных, k)
        ideal_positions = min(len(relevant), k)
        idcg = 0.0
        for i in range(1, ideal_positions + 1):
            # gain = 1 для каждой идеальной позиции
            idcg += 1 / np.log2(i + 1)

        if idcg > 0:
            ndcg = dcg / idcg
            return min(ndcg, 1.0)
        return 0.0


    def evaluate_generation(self, 
                           predictions: Dict[str, str], 
                           references: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Оценка качества генерации для каждого запроса
        
        Args:
            predictions: Словарь {query_id: generated_answer}
            references: Словарь {query_id: reference_answer}
        
        Returns:
            Словарь с метриками для каждого запроса
        """
        if not predictions or not references:
            logger.warning("Нет данных для оценки генерации")
            return {}
        
        results = {}
        
        for qid, pred in predictions.items():
            if qid not in references:
                logger.warning(f"Query {qid} отсутствует в references")
                continue
            
            ref = references[qid]
            metrics = self._compute_generation_metrics(pred, ref)
            results[qid] = metrics
        
        return results
    
    def _compute_generation_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Вычисление метрик для одной пары ответ-эталон
        
        Args:
            prediction: Сгенерированный ответ
            reference: Эталонный ответ
        
        Returns:
            Словарь с метриками
        """
        metrics = {}
        
        # ROUGE метрики
        if self.rouge_scorer is not None:
            try:
                scores = self.rouge_scorer.score(reference, prediction)
                metrics['rouge1'] = scores['rouge1'].fmeasure
                metrics['rouge2'] = scores['rouge2'].fmeasure
                metrics['rougeL'] = scores['rougeL'].fmeasure
            except Exception as e:
                logger.warning(f"Ошибка вычисления ROUGE: {e}")
                metrics['rouge1'] = metrics['rouge2'] = metrics['rougeL'] = 0.0
        
        # BLEU метрика
        if NLTK_AVAILABLE:
            try:
                reference_tokens = reference.split()
                prediction_tokens = prediction.split()
                
                smoothing = SmoothingFunction().method1
                bleu_score = sentence_bleu(
                    [reference_tokens], 
                    prediction_tokens, 
                    smoothing_function=smoothing
                )
                metrics['bleu'] = bleu_score
            except Exception as e:
                logger.warning(f"Ошибка вычисления BLEU: {e}")
                metrics['bleu'] = 0.0
        else:
            metrics['bleu'] = 0.0
        
        # BERTScore
        if self.use_bert_score and self.bert_scorer is not None:
            try:
                P, R, F1 = self.bert_scorer.score([prediction], [reference])
                metrics['bert_score_precision'] = P.item()
                metrics['bert_score_recall'] = R.item()
                metrics['bert_score_f1'] = F1.item()
            except Exception as e:
                logger.warning(f"Ошибка вычисления BERTScore: {e}")
                metrics['bert_score_f1'] = 0.0
        else:
            metrics['bert_score_f1'] = 0.0
        
        return metrics
    
    def calculate_average_metrics(self, metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Усреднение метрик по всем запросам
        
        Args:
            metrics_dict: Словарь с метриками для каждого запроса
        
        Returns:
            Усредненные метрики
        """
        if not metrics_dict:
            return {}
        
        avg_metrics = {}
        
        # Собираем все доступные метрики
        all_metrics = set()
        for qid, metrics in metrics_dict.items():
            all_metrics.update(metrics.keys())
        
        # Усредняем каждую метрику
        for metric in all_metrics:
            values = [metrics_dict[qid][metric] for qid in metrics_dict if metric in metrics_dict[qid]]
            if values:
                avg_metrics[metric] = np.mean(values)
            else:
                avg_metrics[metric] = 0.0
        
        return avg_metrics
    
    def evaluate_full_pipeline(self,
                              retrieval_results: Dict[str, List[str]],
                              generation_results: Dict[str, str],
                              ground_truth_retrieval: Dict[str, List[str]],
                              ground_truth_generation: Dict[str, str],
                              ks: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Полная оценка RAG пайплайна (поиск + генерация)
        
        Args:
            retrieval_results: Результаты поиска
            generation_results: Результаты генерации
            ground_truth_retrieval: Эталонные релевантные документы
            ground_truth_generation: Эталонные ответы
            ks: Значения k для метрик поиска
        
        Returns:
            Словарь с результатами оценки
        """
        results = {}
        
        # Оценка поиска
        retrieval_metrics = self.evaluate_retrieval(
            retrieval_results, 
            ground_truth_retrieval, 
            ks
        )
        results['retrieval'] = retrieval_metrics.to_dict()
        
        # Оценка генерации
        if generation_results and ground_truth_generation:
            gen_metrics_per_query = self.evaluate_generation(
                generation_results,
                ground_truth_generation
            )
            avg_gen_metrics = self.calculate_average_metrics(gen_metrics_per_query)
            results['generation'] = avg_gen_metrics
            results['generation_detailed'] = gen_metrics_per_query
        
        return results


# ==================== ПРОДВИНУТЫЕ МЕТРИКИ RAGAS ====================
    
    def evaluate_ragas_metrics(self,
                               questions: List[str],
                               answers: List[str],
                               contexts: List[List[str]],
                               ground_truths: List[str] = None) -> AdvancedMetrics:
        """
        Оценка продвинутых метрик с помощью RAGAS
        
        Args:
            questions: Список вопросов
            answers: Список сгенерированных ответов
            contexts: Список списков контекстов
            ground_truths: Список эталонных ответов (опционально)
            
        Returns:
            AdvancedMetrics: Объект с метриками
        """
        if not self.ragas_available:
            logger.warning("RAGAS метрики недоступны. Возвращаются нулевые значения.")
            return AdvancedMetrics()
        
        if len(questions) != len(answers) != len(contexts):
            logger.error("Длины списков не совпадают!")
            return AdvancedMetrics()
        
        # Создаем датасет в формате RAGAS
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts
        }
        
        if ground_truths:
            data['ground_truth'] = ground_truths
        
        dataset = Dataset.from_dict(data)
                                    
        
        try:
            # Инициализируем метрики
            faithfulness_metric = Faithfulness(llm=self.ragas_llm)
            answer_relevancy_metric = AnswerRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings)
            context_relevancy_metric = ContextRelevance(llm=self.ragas_llm)            
            # Вычисляем метрики
            results = {
                'faithfulness': [],
                'answer_relevancy': [],
                'context_relevancy': []
            }
            
            for i, (question, answer, context_list) in enumerate(tqdm(zip(questions, answers, contexts), 
                                                                   total=len(questions), 
                                                                   desc="Прогресс вычисления метрик")):
            
                try:
                    # Формируем образец для оценки
                    sample = SingleTurnSample(
                        user_input=question,
                        response=answer,
                        retrieved_contexts=context_list
                    )
                    # Вычисляем метрики (используем asyncio.run для каждого, хотя возможно параллелить)
                    faith_score = asyncio.run(faithfulness_metric.single_turn_ascore(sample))
                    answer_score = asyncio.run(answer_relevancy_metric.single_turn_ascore(sample))
                    context_score = asyncio.run(context_relevancy_metric.single_turn_ascore(sample))
                    # Сохраняем результаты
                    # Сохраняем результаты
                    results['faithfulness'].append(faith_score)
                    results['answer_relevancy'].append(answer_score)
                    results['context_relevancy'].append(context_score)

                except Exception as e:
                    logger.error(f"Ошибка при обработке примера {i}: {e}")
                    results['faithfulness'].append(0.0)
                    results['answer_relevancy'].append(0.0)
                    results['context_relevancy'].append(0.0)


            # Вычисляем средние значения
            avg_faithfulness = sum(results['faithfulness']) / len(results['faithfulness']) if results['faithfulness'] else 0.0
            avg_answer = sum(results['answer_relevancy']) / len(results['answer_relevancy']) if results['answer_relevancy'] else 0.0
            avg_context = sum(results['context_relevancy']) / len(results['context_relevancy']) if results['context_relevancy'] else 0.0

            # Создаем DataFrame для детальных результатов
            df = pd.DataFrame({
                'Faithfulness': results['faithfulness'],
                'Answer_Relevance': results['answer_relevancy'],
                'Context_Relevance': results['context_relevancy']
            })

            logger.info(f"RAGAS метрики: Faith={avg_faithfulness:.3f}, "
                       f"Answer={avg_answer:.3f}, Context={avg_context:.3f}")

            return AdvancedMetrics(
                faithfulness=avg_faithfulness,
                answer_relevancy=avg_answer,
                context_relevancy=avg_context,
                detailed_results={
                    'individual_results': results,
                    'dataframe': df.to_dict()
                }
            )

            
        except Exception as e:
            logger.error(f"Ошибка при вычислении RAGAS метрик: {e}")
            return AdvancedMetrics()
   
    def evaluate_all_metrics(self,
                            predictions_retrieval: Dict[str, List[str]],
                            ground_truth_retrieval: Dict[str, List[str]],
                            questions: List[str],
                            answers: List[str],
                            contexts: List[List[str]],
                            ground_truths: List[str] = None,
                            ks: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Полная оценка всех метрик
        
        Returns:
            Словарь со всеми метриками
        """
        results = {}
        
        # Метрики поиска
        retrieval_metrics = self.evaluate_retrieval(
            predictions_retrieval, 
            ground_truth_retrieval, 
            ks
        )
        results['retrieval'] = retrieval_metrics.to_dict()
        
        # Метрики генерации
        if answers and ground_truths:
            gen_metrics = {}
            for qid, answer in enumerate(answers):
                if ground_truths and qid < len(ground_truths):
                    gen_metrics[f'q{qid:03d}'] = self._compute_generation_metrics(
                        answer, 
                        ground_truths[qid]
                    )
            results['generation'] = self.calculate_average_metrics(gen_metrics)
        
        # RAGAS метрики
        if questions and answers and contexts:
            ragas_metrics = self.evaluate_ragas_metrics(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths
            )
            results['ragas'] = ragas_metrics.to_dict()
        
        return results
                
class RAGMetricsCalculator:
    """
    Утилитарный класс для расчета дополнительных метрик RAG
    """
    
    @staticmethod
    def response_length_distribution(responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Анализ распределения длины ответов
        
        Args:
            responses: Словарь ответов
        
        Returns:
            Статистика по длине ответов
        """
        lengths = [len(resp.split()) for resp in responses.values()]
        
        return {
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_length': np.mean(lengths) if lengths else 0,
            'std_length': np.std(lengths) if lengths else 0,
            'total_responses': len(responses)
        }
    
    @staticmethod
    def retrieval_coverage(predictions: Dict[str, List[str]], 
                          ground_truth: Dict[str, List[str]]) -> float:
        """
        Вычисление покрытия - сколько релевантных документов было найдено хотя бы раз
        
        Args:
            predictions: Предсказанные документы
            ground_truth: Релевантные документы
        
        Returns:
            Процент найденных релевантных документов
        """
        all_relevant = set()
        for docs in ground_truth.values():
            all_relevant.update(docs)
        
        all_found = set()
        for docs in predictions.values():
            all_found.update(docs)
        
        found_relevant = all_found & all_relevant
        return len(found_relevant) / len(all_relevant) if all_relevant else 0.0
    
    @staticmethod
    def novelty_score(predictions: Dict[str, List[str]], 
                     ground_truth: Dict[str, List[str]]) -> float:
        """
        Вычисление новизны - сколько найденных документов не было в обучающей выборке
        
        Args:
            predictions: Предсказанные документы
            ground_truth: Релевантные документы
        
        Returns:
            Процент новых документов
        """
        all_predicted = set()
        for docs in predictions.values():
            all_predicted.update(docs)
        
        all_relevant = set()
        for docs in ground_truth.values():
            all_relevant.update(docs)
        
        novel_docs = all_predicted - all_relevant
        return len(novel_docs) / len(all_predicted) if all_predicted else 0.0

# ==================== ТЕСТИРОВАНИЕ МЕТРИКИ PRECISION@K ====================

def test_precision_at_k():
    """
    Функция для проверки корректности вычисления метрики Precision@K
    """
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ МЕТРИКИ PRECISION@K")
    print("=" * 70)
    
    # Создаем экземпляр оценщика
    evaluator = RAGEvaluator()
    
    # Тестовые данные
    test_cases = {
        # Запрос 0: из учебника (релевантные на позициях 1, 3, 5)
        "q0": {
            "predictions": ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"],
            "ground_truth": ["doc_A", "doc_C", "doc_E"]
        },
        # Запрос 1: Идеальный случай – все релевантные на первых позициях
        "q1": {
            "predictions": ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"],
            "ground_truth": ["doc_A", "doc_B", "doc_C"]
        },
        # Запрос 2: Релевантные документы разбросаны
        "q2": {
            "predictions": ["doc_X", "doc_Y", "doc_Z", "doc_W", "doc_V"],
            "ground_truth": ["doc_Z", "doc_W", "doc_X"]
        },
        # Запрос 3: Только один релевантный документ (на позиции 3)
        "q3": {
            "predictions": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"],
            "ground_truth": ["doc_3"]
        },
        # Запрос 4: Релевантных документов нет
        "q4": {
            "predictions": ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"],
            "ground_truth": ["doc_F", "doc_G"]
        },
        # Запрос 5: Все предсказания релевантны
        "q5": {
            "predictions": ["doc_R1", "doc_R2", "doc_R3", "doc_R4", "doc_R5"],
            "ground_truth": ["doc_R1", "doc_R2", "doc_R3", "doc_R4", "doc_R5"]
        },
        # Запрос 6: Частичное совпадение (2 из 5 релевантны)
        "q6": {
            "predictions": ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"],
            "ground_truth": ["doc_A", "doc_D"]
        },
        # Запрос 7: Пустые списки
        "q7": {
            "predictions": [],
            "ground_truth": []
        },
        # Запрос 8: Предсказаний меньше, чем K
        "q8": {
            "predictions": ["doc_A", "doc_B"],
            "ground_truth": ["doc_A", "doc_B", "doc_C"]
        }
    }
    
    # Ожидаемые значения Precision@K
    expected = {
        "q0": {1: 1.0, 3: 2/3, 5: 3/5},
        "q1": {1: 1.0, 3: 3/3, 5: 3/5},
        "q2": {1: 1.0, 3: 2/3, 5: 3/5},
        "q3": {1: 0.0, 3: 1/3, 5: 1/5},
        "q4": {1: 0.0, 3: 0.0, 5: 0.0},
        "q5": {1: 1.0, 3: 1.0, 5: 1.0},
        "q6": {1: 1.0, 3: 1/3, 5: 2/5},
        "q7": {1: 0.0, 3: 0.0, 5: 0.0},
        "q8": {1: 1.0, 3: 2/2, 5: 2/2}
    }
    
    # Счетчики для статистики
    total_tests = 0
    passed_tests = 0
    
    # Заголовок таблицы
    print(f"\n{'ID':<6} {'Predictions':<35} {'Ground Truth':<20} {'P@1':<8} {'P@3':<8} {'P@5':<8}")
    print("-" * 95)
    
    for qid, data in test_cases.items():
        pred = data["predictions"]
        truth = {qid: data["ground_truth"]}
        pred_dict = {qid: pred}
        
        # Вычисляем метрики
        metrics = evaluator.evaluate_retrieval(pred_dict, truth, ks=[1, 3, 5])
        p_at_k = metrics.precision_at_k
        
        # Получаем ожидаемые значения
        exp = expected[qid]
        
        # Вычисляем релевантные позиции для отображения
        relevant_positions = [i+1 for i, doc in enumerate(pred) if doc in data["ground_truth"]]
        
        # Формируем строку с predictions (обрезаем для длинных)
        pred_str = str(pred[:5])
        truth_str = str(data["ground_truth"])
        
        # Проверяем значения
        p1_status = "✓" if abs(p_at_k.get(1, 0.0) - exp.get(1, 0.0)) < 1e-6 else "✗"
        p3_status = "✓" if abs(p_at_k.get(3, 0.0) - exp.get(3, 0.0)) < 1e-6 else "✗"
        p5_status = "✓" if abs(p_at_k.get(5, 0.0) - exp.get(5, 0.0)) < 1e-6 else "✗"
        
        # Обновляем счетчики
        if p1_status == "✓":
            passed_tests += 1
        total_tests += 1
        if p3_status == "✓":
            passed_tests += 1
        total_tests += 1
        if p5_status == "✓":
            passed_tests += 1
        total_tests += 1
        
        # Выводим результат
        print(f"{qid:<6} {pred_str:<35} {truth_str:<20} "
              f"{p_at_k.get(1, 0.0):.4f}{p1_status:<3} "
              f"{p_at_k.get(3, 0.0):.4f}{p3_status:<3} "
              f"{p_at_k.get(5, 0.0):.4f}{p5_status:<3}")
    
    # Вывод итоговой статистики
    print("-" * 95)
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"   Всего проверок: {total_tests}")
    print(f"   Успешно: {passed_tests}")
    print(f"   Ошибок: {total_tests - passed_tests}")
    print(f"   Успеваемость: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! Метрика Precision@K работает корректно.")
    else:
        print("\n❌ ЕСТЬ ОШИБКИ! Проверьте реализацию метода _precision_at_k.")
    
    # Демонстрация недостатка Precision@K (не учитывает порядок)
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ НЕДОСТАТКА PRECISION@K")
    print("=" * 70)
    
    # Модель A: релевантные документы на первых позициях
    model_a_pred = ["rel1", "rel2", "rel3", "non1", "non2"]
    model_b_pred = ["non1", "non2", "rel1", "rel2", "rel3"]
    ground_truth_set = {"rel1", "rel2", "rel3"}
    
    p5_a = evaluator._precision_at_k(model_a_pred, ground_truth_set, 5)
    p5_b = evaluator._precision_at_k(model_b_pred, ground_truth_set, 5)
    
    print(f"\nМодель A (релевантные на позициях 1,2,3): {model_a_pred}")
    print(f"Модель B (релевантные на позициях 3,4,5): {model_b_pred}")
    print(f"\nPrecision@5 для модели A: {p5_a:.4f}")
    print(f"Precision@5 для модели B: {p5_b:.4f}")
    print(f"\n⚠️ Precision@5 ОДИНАКОВ для обеих моделей ({p5_a:.4f}),")
    print("   хотя Модель A лучше, так как релевантные документы находятся выше!")
    print("   Это основной недостаток метрики Precision@K - она не учитывает порядок.")

def test_recall_at_k():
    """
    Функция для проверки корректности вычисления метрики Recall@K
    """
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ МЕТРИКИ RECALL@K")
    print("=" * 70)
    
    # Создаем экземпляр оценщика
    evaluator = RAGEvaluator()
    
    # Тестовые данные для Recall
    test_cases_recall = {
        # Запрос 0: 3 релевантных документа, на позициях 1, 3, 5
        "q0_recall": {
            "predictions": ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"],
            "ground_truth": ["doc_A", "doc_C", "doc_E"],
            "description": "3 релевантных на позициях 1, 3, 5"
        },
        # Запрос 1: 3 релевантных документа, все в топ-3
        "q1_recall": {
            "predictions": ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"],
            "ground_truth": ["doc_A", "doc_B", "doc_C"],
            "description": "3 релевантных в топ-3"
        },
        # Запрос 2: 3 релевантных документа, разбросаны
        "q2_recall": {
            "predictions": ["doc_X", "doc_Y", "doc_Z", "doc_W", "doc_V"],
            "ground_truth": ["doc_Z", "doc_W", "doc_X"],
            "description": "3 релевантных на позициях 1, 3, 4"
        },
        # Запрос 3: 1 релевантный документ на позиции 3
        "q3_recall": {
            "predictions": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"],
            "ground_truth": ["doc_3"],
            "description": "1 релевантный на позиции 3"
        },
        # Запрос 4: Нет релевантных документов
        "q4_recall": {
            "predictions": ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"],
            "ground_truth": ["doc_F", "doc_G"],
            "description": "Нет релевантных в предсказаниях"
        },
        # Запрос 5: Все предсказания релевантны
        "q5_recall": {
            "predictions": ["doc_R1", "doc_R2", "doc_R3", "doc_R4", "doc_R5"],
            "ground_truth": ["doc_R1", "doc_R2", "doc_R3", "doc_R4", "doc_R5"],
            "description": "Все 5 предсказаний релевантны"
        },
        # Запрос 6: 2 релевантных на позициях 1 и 4
        "q6_recall": {
            "predictions": ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"],
            "ground_truth": ["doc_A", "doc_D"],
            "description": "2 релевантных на позициях 1 и 4"
        },
        # Запрос 7: Пустые списки
        "q7_recall": {
            "predictions": [],
            "ground_truth": [],
            "description": "Пустые списки"
        },
        # Запрос 8: Предсказаний меньше, чем K
        "q8_recall": {
            "predictions": ["doc_A", "doc_B"],
            "ground_truth": ["doc_A", "doc_B", "doc_C"],
            "description": "2 предсказания, 3 релевантных"
        }
    }
    
    # Ожидаемые значения Recall@K
    # Формула: Recall@K = hits / total_relevant
    expected_recall = {
        "q0_recall": {1: 1/3, 3: 2/3, 5: 3/3},
        "q1_recall": {1: 1/3, 3: 3/3, 5: 3/3},
        "q2_recall": {1: 1/3, 3: 2/3, 5: 3/3},
        "q3_recall": {1: 0/1, 3: 1/1, 5: 1/1},
        "q4_recall": {1: 0/2, 3: 0/2, 5: 0/2},
        "q5_recall": {1: 1/5, 3: 3/5, 5: 5/5},
        "q6_recall": {1: 1/2, 3: 1/2, 5: 2/2},
        "q7_recall": {1: 0.0, 3: 0.0, 5: 0.0},
        "q8_recall": {1: 1/3, 3: 2/3, 5: 2/3}
    }
    
    # Счетчики для статистики
    total_tests = 0
    passed_tests = 0
    
    # Заголовок таблицы
    print(f"\n{'ID':<14} {'Predictions':<35} {'Ground Truth':<20} {'R@1':<8} {'R@3':<8} {'R@5':<8}")
    print("-" * 95)
    
    for qid, data in test_cases_recall.items():
        pred = data["predictions"]
        truth = {qid: data["ground_truth"]}
        pred_dict = {qid: pred}
        
        # Вычисляем метрики
        metrics = evaluator.evaluate_retrieval(pred_dict, truth, ks=[1, 3, 5])
        r_at_k = metrics.recall_at_k
        
        # Получаем ожидаемые значения
        exp = expected_recall[qid]
        
        # Вычисляем релевантные позиции для отображения
        relevant_positions = [i+1 for i, doc in enumerate(pred) if doc in data["ground_truth"]]
        total_relevant = len(data["ground_truth"])
        
        # Формируем строки
        pred_str = str(pred[:5])
        truth_str = str(data["ground_truth"])
        
        # Проверяем значения
        r1_status = "✓" if abs(r_at_k.get(1, 0.0) - exp.get(1, 0.0)) < 1e-6 else "✗"
        r3_status = "✓" if abs(r_at_k.get(3, 0.0) - exp.get(3, 0.0)) < 1e-6 else "✗"
        r5_status = "✓" if abs(r_at_k.get(5, 0.0) - exp.get(5, 0.0)) < 1e-6 else "✗"
        
        # Обновляем счетчики
        if r1_status == "✓":
            passed_tests += 1
        total_tests += 1
        if r3_status == "✓":
            passed_tests += 1
        total_tests += 1
        if r5_status == "✓":
            passed_tests += 1
        total_tests += 1
        
        # Выводим результат
        print(f"{qid:<14} {pred_str:<35} {truth_str:<20} "
              f"{r_at_k.get(1, 0.0):.4f}{r1_status:<3} "
              f"{r_at_k.get(3, 0.0):.4f}{r3_status:<3} "
              f"{r_at_k.get(5, 0.0):.4f}{r5_status:<3}")
    
    # Вывод итоговой статистики
    print("-" * 95)
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ RECALL@K:")
    print(f"   Всего проверок: {total_tests}")
    print(f"   Успешно: {passed_tests}")
    print(f"   Ошибок: {total_tests - passed_tests}")
    print(f"   Успеваемость: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n✅ ВСЕ ТЕСТЫ RECALL@K ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("\n❌ ЕСТЬ ОШИБКИ! Проверьте реализацию метода _recall_at_k.")
    
    # Детальный вывод для наглядности
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЙ РАСЧЕТ RECALL@K НА ПРИМЕРЕ q0_recall")
    print("=" * 70)
    
    qid = "q0_recall"
    data = test_cases_recall[qid]
    pred = data["predictions"]
    truth = {qid: data["ground_truth"]}
    pred_dict = {qid: pred}
    
    metrics = evaluator.evaluate_retrieval(pred_dict, truth, ks=[1, 3, 5])
    r_at_k = metrics.recall_at_k
    
    print(f"\nЗапрос: {qid}")
    print(f"Предсказания: {pred}")
    print(f"Релевантные документы: {data['ground_truth']}")
    print(f"Всего релевантных: {len(data['ground_truth'])}")
    print(f"\nРелевантные позиции: ", end="")
    for i, doc in enumerate(pred, 1):
        if doc in data["ground_truth"]:
            print(f"позиция {i} ({doc})", end=" ")
    print()
    print(f"\nRecall@1 = {r_at_k.get(1, 0):.4f} (1 релевантный из {len(data['ground_truth'])} в топ-1)")
    print(f"Recall@3 = {r_at_k.get(3, 0):.4f} (2 релевантных из {len(data['ground_truth'])} в топ-3)")
    print(f"Recall@5 = {r_at_k.get(5, 0):.4f} (3 релевантных из {len(data['ground_truth'])} в топ-5)")


def test_mrr():
    """
    Функция для проверки корректности вычисления метрики MRR (Mean Reciprocal Rank)
    """
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ МЕТРИКИ MRR (Mean Reciprocal Rank)")
    print("=" * 70)
    
    # Создаем экземпляр оценщика
    evaluator = RAGEvaluator()
    
    # Тестовые данные для MRR
    test_cases_mrr = [
        {
            "id": "q1",
            "predictions": ["rel1", "rel2", "rel3", "non1", "non2"],
            "ground_truth": ["rel1", "rel2", "rel3"],
            "description": "Первый релевантный на позиции 1",
            "expected_rr": 1.0
        },
        {
            "id": "q2",
            "predictions": ["non1", "rel1", "rel2", "rel3", "non2"],
            "ground_truth": ["rel1", "rel2", "rel3"],
            "description": "Первый релевантный на позиции 2",
            "expected_rr": 1/2
        },
        {
            "id": "q3",
            "predictions": ["non1", "non2", "rel1", "rel2", "rel3"],
            "ground_truth": ["rel1", "rel2", "rel3"],
            "description": "Первый релевантный на позиции 3",
            "expected_rr": 1/3
        },
        {
            "id": "q4",
            "predictions": ["non1", "non2", "non3", "non4", "non5"],
            "ground_truth": ["rel1", "rel2", "rel3"],
            "description": "Нет релевантных документов",
            "expected_rr": 0.0
        },
        {
            "id": "q5",
            "predictions": ["rel1"],
            "ground_truth": ["rel1", "rel2"],
            "description": "Один релевантный на позиции 1",
            "expected_rr": 1.0
        },
        {
            "id": "q6",
            "predictions": [],
            "ground_truth": ["rel1", "rel2"],
            "description": "Пустые предсказания",
            "expected_rr": 0.0
        },
        {
            "id": "q7",
            "predictions": ["rel1", "non1", "rel2",  "non2", "rel3"],
            "ground_truth": ["rel1", "rel3", "rel5"],
            "description": "релевантный на позиции 1 3 5",
            "expected_rr": 1.0
        },        
        {
            "id": "q8",
            "predictions": ["non1", "rel1", "non2",  "non3", "rel2"],
            "ground_truth": ["rel1", "rel2"],
            "description": "релевантный на позиции 2 5",
            "expected_rr": 0.5
        },        
        {
            "id": "q9",
            "predictions": ["non1","non2","non3","non4", "rel1"],
            "ground_truth": ["rel1"],
            "description": "релевантный на позиции 5",
            "expected_rr": 0.2
        },                        
    ]
    
    print(f"\n{'ID':<6} {'Predictions':<40} {'Ground Truth':<20} {'RR':<8} {'Expected':<8} {'Status':<8}")
    print("-" * 95)
    
    passed = 0
    total = 0
    
    for tc in test_cases_mrr:
        pred = tc["predictions"]
        truth = {tc["id"]: tc["ground_truth"]}
        pred_dict = {tc["id"]: pred}
        
        # Вычисляем метрики
        metrics = evaluator.evaluate_retrieval(pred_dict, truth, ks=[1])
        mrr = metrics.mrr
        
        expected = tc["expected_rr"]
        status = "✓" if abs(mrr - expected) < 1e-6 else "✗"
        
        if status == "✓":
            passed += 1
        total += 1
        
        pred_str = str(pred[:5])
        truth_str = str(tc["ground_truth"])
        
        print(f"{tc['id']:<6} {pred_str:<40} {truth_str:<20} {mrr:.4f}    {expected:.4f}    {status}")
    
    print("-" * 95)
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ MRR:")
    print(f"   Всего проверок: {total}")
    print(f"   Успешно: {passed}")
    print(f"   Успеваемость: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✅ ВСЕ ТЕСТЫ MRR ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("\n❌ ЕСТЬ ОШИБКИ! Проверьте реализацию метода _reciprocal_rank.")

def test_ndcg_at_k():
    """
    Функция для проверки корректности вычисления метрики NDCG@K
    NDCG = DCG / IDCG, где:
    - DCG = сумма gain_i / log2(i+1)
    - IDCG = сумма ideal_gain_i / log2(i+1)
    - gain = 1 если документ релевантен, иначе 0
    """
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ МЕТРИКИ NDCG@K")
    print("=" * 70)
    
    # Создаем экземпляр оценщика
    evaluator = RAGEvaluator()
    
    # Тестовые данные для NDCG
    test_cases_ndcg = {
        # Запрос 0: Релевантные на позициях 1, 2, 3 (идеальный порядок)
        "q0_ndcg": {
            "predictions": ["rel1", "rel2", "rel3", "non1", "non2"],
            "ground_truth": ["rel1", "rel2", "rel3"],
            "description": "Идеальный порядок - все релевантные в топ-3"
        },
        # Запрос 1: Релевантные на позициях 1, 3, 5
        "q1_ndcg": {
            "predictions": ["rel1", "non1", "rel2", "non2", "rel3"],
            "ground_truth": ["rel1", "rel2", "rel3"],
            "description": "Релевантные на позициях 1, 3, 5"
        },
        # Запрос 2: Релевантные на позициях 3, 4, 5 (худший порядок)
        "q2_ndcg": {
            "predictions": ["non1", "non2", "rel1", "rel2", "rel3"],
            "ground_truth": ["rel1", "rel2", "rel3"],
            "description": "Релевантные на позициях 3, 4, 5"
        },
        # Запрос 3: Только один релевантный на позиции 1
        "q3_ndcg": {
            "predictions": ["rel1", "non1", "non2", "non3", "non4"],
            "ground_truth": ["rel1"],
            "description": "Один релевантный на позиции 1"
        },
        # Запрос 4: Только один релевантный на позиции 5
        "q4_ndcg": {
            "predictions": ["non1", "non2", "non3", "non4", "rel1"],
            "ground_truth": ["rel1"],
            "description": "Один релевантный на позиции 5"
        },
        # Запрос 5: Нет релевантных документов
        "q5_ndcg": {
            "predictions": ["non1", "non2", "non3", "non4", "non5"],
            "ground_truth": ["rel1", "rel2"],
            "description": "Нет релевантных документов"
        },
        # Запрос 6: Все предсказания релевантны (идеальный случай)
        "q6_ndcg": {
            "predictions": ["rel1", "rel2", "rel3", "rel4", "rel5"],
            "ground_truth": ["rel1", "rel2", "rel3", "rel4", "rel5"],
            "description": "Все 5 предсказаний релевантны"
        },
        # Запрос 7: 2 релевантных на позициях 2 и 4
        "q7_ndcg": {
            "predictions": ["non1", "rel1", "non2", "rel2", "non3"],
            "ground_truth": ["rel1", "rel2"],
            "description": "2 релевантных на позициях 2 и 4"
        },
        # Запрос 8: Релевантных больше, чем K (K=3, релевантных 5)
        "q8_ndcg": {
            "predictions": ["rel1", "rel2", "rel3", "rel4", "rel5"],
            "ground_truth": ["rel1", "rel2", "rel3", "rel4", "rel5"],
            "description": "5 релевантных, считаем NDCG@3"
        },
    }
    
    # Ожидаемые значения NDCG@K
    # Рассчитаны по формуле: DCG = sum(1/log2(i+1)), IDCG = sum(1/log2(i+1)) для идеального порядка

    expected_ndcg = {
        "q0_ndcg": {1: 1.0, 3: 1.0, 5: 1.0},
        "q1_ndcg": {
            1: 1.0,
            3: (1/np.log2(2) + 0 + 1/np.log2(4)) / (1/np.log2(2) + 1/np.log2(3) + 1/np.log2(4)),
            5: (1/np.log2(2) + 0 + 1/np.log2(4) + 0 + 1/np.log2(6)) / (1/np.log2(2) + 1/np.log2(3) + 1/np.log2(4))
        },
        "q2_ndcg": {
            1: 0.0,
            3: (0 + 0 + 1/np.log2(4)) / (1/np.log2(2) + 1/np.log2(3) + 1/np.log2(4)),
            5: (0 + 0 + 1/np.log2(4) + 1/np.log2(5) + 1/np.log2(6)) / (1/np.log2(2) + 1/np.log2(3) + 1/np.log2(4))
        },
        "q3_ndcg": {1: 1.0, 3: 1.0, 5: 1.0},
        "q4_ndcg": {
            1: 0.0,
            3: 0.0,
            5: (1/np.log2(6)) / (1/np.log2(2))
        },
        "q5_ndcg": {1: 0.0, 3: 0.0, 5: 0.0},
        "q6_ndcg": {1: 1.0, 3: 1.0, 5: 1.0},
        "q7_ndcg": {
            1: 0.0,
            3: (0 + 1/np.log2(3) + 0) / (1/np.log2(2) + 1/np.log2(3)),
            5: (0 + 1/np.log2(3) + 0 + 1/np.log2(5) + 0) / (1/np.log2(2) + 1/np.log2(3))
        },
        "q8_ndcg": {1: 1.0, 3: 1.0, 5: 1.0}
    }    
    # Счетчики для статистики
    total_tests = 0
    passed_tests = 0
    
    # Заголовок таблицы
    print(f"\n{'ID':<14} {'Predictions':<35} {'Ground Truth':<20} {'NDCG@1':<10} {'NDCG@3':<10} {'NDCG@5':<10}")
    print("-" * 105)
    
    for qid, data in test_cases_ndcg.items():
        pred = data["predictions"]
        truth = {qid: data["ground_truth"]}
        pred_dict = {qid: pred}
        
        # Вычисляем метрики
        metrics = evaluator.evaluate_retrieval(pred_dict, truth, ks=[1, 3, 5])
        ndcg_at_k = metrics.ndcg_at_k
        
        # Получаем ожидаемые значения
        exp = expected_ndcg[qid]
        
        # Формируем строки
        pred_str = str(pred[:5])
        truth_str = str(data["ground_truth"])
        
        # Проверяем значения с допуском 0.01 (из-за погрешностей округления)
        n1_got = ndcg_at_k.get(1, 0.0)
        n1_exp = exp.get(1, 0.0)
        n3_got = ndcg_at_k.get(3, 0.0)
        n3_exp = exp.get(3, 0.0)
        n5_got = ndcg_at_k.get(5, 0.0)
        n5_exp = exp.get(5, 0.0)
        
        n1_status = "✓" if abs(n1_got - n1_exp) < 0.01 else "✗"
        n3_status = "✓" if abs(n3_got - n3_exp) < 0.01 else "✗"
        n5_status = "✓" if abs(n5_got - n5_exp) < 0.01 else "✗"
        
        # Обновляем счетчики
        if n1_status == "✓":
            passed_tests += 1
        total_tests += 1
        if n3_status == "✓":
            passed_tests += 1
        total_tests += 1
        if n5_status == "✓":
            passed_tests += 1
        total_tests += 1
        
        # Выводим результат
        print(f"{qid:<14} {pred_str:<35} {truth_str:<20} "
              f"{n1_got:.4f}{n1_status:<4} "
              f"{n3_got:.4f}{n3_status:<4} "
              f"{n5_got:.4f}{n5_status:<4}")
    
    # Вывод итоговой статистики
    print("-" * 105)
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ NDCG@K:")
    print(f"   Всего проверок: {total_tests}")
    print(f"   Успешно: {passed_tests}")
    print(f"   Ошибок: {total_tests - passed_tests}")
    print(f"   Успеваемость: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n✅ ВСЕ ТЕСТЫ NDCG@K ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("\n❌ ЕСТЬ ОШИБКИ! Проверьте реализацию метода _ndcg_at_k.")
    
    # Детальный вывод для наглядности
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЙ РАСЧЕТ NDCG@K НА ПРИМЕРАХ")
    print("=" * 70)
    
    # Пример 1: Идеальный порядок (q0_ndcg)
    print("\n📌 ПРИМЕР 1: Идеальный порядок (все релевантные в топ-3)")
    data = test_cases_ndcg["q0_ndcg"]
    pred = data["predictions"]
    truth = {"q0": data["ground_truth"]}
    pred_dict = {"q0": pred}
    
    metrics = evaluator.evaluate_retrieval(pred_dict, truth, ks=[1, 3, 5])
    
    print(f"   Предсказания: {pred}")
    print(f"   Релевантные: {data['ground_truth']}")
    print(f"   NDCG@1 = {metrics.ndcg_at_k.get(1, 0):.4f} (должен быть 1.0)")
    print(f"   NDCG@3 = {metrics.ndcg_at_k.get(3, 0):.4f} (должен быть 1.0)")
    print(f"   NDCG@5 = {metrics.ndcg_at_k.get(5, 0):.4f} (должен быть 1.0)")
    
    # Пример 2: Релевантные на позициях 1, 3, 5 (q1_ndcg)
    print("\n📌 ПРИМЕР 2: Релевантные на позициях 1, 3, 5")
    data = test_cases_ndcg["q1_ndcg"]
    pred = data["predictions"]
    truth = {"q1": data["ground_truth"]}
    pred_dict = {"q1": pred}
    
    metrics = evaluator.evaluate_retrieval(pred_dict, truth, ks=[1, 3, 5])
    
    # Ручной расчет DCG и IDCG для проверки
    dcg_3 = 1/np.log2(2) + 0 + 1/np.log2(4)  # позиции 1,3 (1-indexed: 1,3)
    idcg_3 = 1/np.log2(2) + 1/np.log2(3) + 1/np.log2(4)
    ndcg_3_manual = dcg_3 / idcg_3
    
    dcg_5 = 1/np.log2(2) + 0 + 1/np.log2(4) + 0 + 1/np.log2(6)
    idcg_5 = 1/np.log2(2) + 1/np.log2(3) + 1/np.log2(4) + 1/np.log2(5) + 1/np.log2(6)
    ndcg_5_manual = dcg_5 / idcg_5
    
    print(f"   Предсказания: {pred}")
    print(f"   Релевантные: {data['ground_truth']}")
    print(f"   NDCG@3 = {metrics.ndcg_at_k.get(3, 0):.4f} (ручной расчет: {ndcg_3_manual:.4f})")
    print(f"   NDCG@5 = {metrics.ndcg_at_k.get(5, 0):.4f} (ручной расчет: {ndcg_5_manual:.4f})")
    
    # Пример 3: Худший порядок (q2_ndcg)
    print("\n📌 ПРИМЕР 3: Худший порядок (релевантные на позициях 3, 4, 5)")
    data = test_cases_ndcg["q2_ndcg"]
    pred = data["predictions"]
    truth = {"q2": data["ground_truth"]}
    pred_dict = {"q2": pred}
    
    metrics = evaluator.evaluate_retrieval(pred_dict, truth, ks=[1, 3, 5])
    
    dcg_3_worst = 0 + 0 + 1/np.log2(4)
    ndcg_3_worst_manual = dcg_3_worst / idcg_3
    
    print(f"   Предсказания: {pred}")
    print(f"   Релевантные: {data['ground_truth']}")
    print(f"   NDCG@3 = {metrics.ndcg_at_k.get(3, 0):.4f} (ручной расчет: {ndcg_3_worst_manual:.4f})")

if __name__ == "__main__":
    #test_precision_at_k()  # Тестирование Precision@K
    #test_recall_at_k()     # Тестирование Recall@K
    #test_mrr()             # Тестирование MRR
    test_ndcg_at_k()       # Тестирование NDCG@K 
