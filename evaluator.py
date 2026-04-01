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


class RAGEvaluator:
    """
    Класс для оценки RAG системы
    Поддерживает оценку:
    - Поиска (MAP, MRR, Precision@k, Recall@k, NDCG@k)
    - Генерации (ROUGE, BLEU, BERTScore)
    - Продвинутые метрики (Faithfulness, AnswerRelevancy, ContextRelevancy)
    """
    def __init__(self, 
                 use_bert_score: bool = True,
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
                self.answer_relevancy_metric = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
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

            # Вычисляем метрики используя существующие методы
            ap = self._average_precision(normalized_pred, relevant_set)
            rr = self._reciprocal_rank(normalized_pred, relevant_set)

            map_scores.append(ap)
            mrr_scores.append(rr)

            for k in ks:
                p = self._precision_at_k(normalized_pred, relevant_set, k)
                r = self._recall_at_k(normalized_pred, relevant_set, k)
                ndcg = self._ndcg_at_k(normalized_pred, relevant_set, k)

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
        
        Args:
            retrieved: Список найденных документов
            relevant: Множество релевантных документов
        
        Returns:
            Average Precision
        """
        if not relevant:
            return 0.0
        
        num_hits = 0
        sum_precisions = 0.0
        
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                num_hits += 1
                sum_precisions += num_hits / i
        
        return sum_precisions / len(relevant) if relevant else 0.0
    
    def _reciprocal_rank(self, retrieved: List[str], relevant: set) -> float:
        """
        Вычисление Reciprocal Rank для одного запроса
        
        Args:
            retrieved: Список найденных документов
            relevant: Множество релевантных документов
        
        Returns:
            Reciprocal Rank
        """
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / i
        return 0.0
    
    def _precision_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """
        Вычисление Precision@k
        
        Args:
            retrieved: Список найденных документов
            relevant: Множество релевантных документов
            k: Параметр k
        
        Returns:
            Precision@k
        """
        if k <= 0:
            return 0.0
        
        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in relevant)
        return hits / k
    
    def _recall_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """
        Вычисление Recall@k
        
        Args:
            retrieved: Список найденных документов
            relevant: Множество релевантных документов
            k: Параметр k
        
        Returns:
            Recall@k
        """
        if not relevant:
            return 0.0
        
        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in relevant)
        return hits / len(relevant)
    
    def _ndcg_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """
        Вычисление NDCG@k (Normalized Discounted Cumulative Gain)
        
        Args:
            retrieved: Список найденных документов
            relevant: Множество релевантных документов
            k: Параметр k
        
        Returns:
            NDCG@k
        """
        if k <= 0:
            return 0.0
        
        # Вычисляем DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k], 1):
            gain = 1 if doc in relevant else 0
            dcg += gain / np.log2(i + 1)
        
        # Вычисляем IDCG (идеальный DCG)
        ideal_ranking = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
        idcg = 0.0
        for i, gain in enumerate(ideal_ranking, 1):
            idcg += gain / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
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