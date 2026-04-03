"""
experiment.py - Модуль для проведения экспериментов с RAG системой
Обеспечивает запуск, управление и сохранение результатов экспериментов
"""
import json
import time
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, is_dataclass
from collections import defaultdict

# В начале файла experiment.py, после импортов
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_gigachat import GigaChat

from models import (
    TestSample, ExperimentConfig, ExperimentResult, 
    ExperimentStatus, RetrievalType, RetrievalMetrics
)
from rag_core import RAGPipeline
from data_loader import TestDataLoader
from evaluator import RAGEvaluator

# Импортируем конфигурации
from config import (
    model_config, data_config, vectorstore_config, 
    experiment_config, logging_config, gigachat_config
)

# Настройка логирования
# Создаем логгер для модуля
logger = logging.getLogger(__name__)

#№ogging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Запуск и управление экспериментами
    Поддерживает:
    - Запуск экспериментов с разными конфигурациями
    - Сохранение результатов в JSON и CSV
    - Сравнение экспериментов
    - Воспроизведение экспериментов
    """
    
    def __init__(self, experiments_dir: Path = Path("experiments")):
        """
        Инициализация менеджера экспериментов
        
        Args:
            experiments_dir: Директория для сохранения результатов
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Поддиректории
        self.results_dir = self.experiments_dir / "results"
        self.reports_dir = self.experiments_dir / "reports"
        self.checkpoints_dir = self.experiments_dir / "checkpoints"
        
        for dir_path in [self.results_dir, self.reports_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExperimentRunner инициализирован. Директория: {self.experiments_dir}")
    
    def _get_full_config(self, pipeline: RAGPipeline = None) -> Dict[str, Any]:
        """
        Получение полной конфигурации системы
        
        Args:
            pipeline: RAGPipeline для получения дополнительных параметров
            
        Returns:
            Dict: Полная конфигурация
        """
        full_config = {
            # Конфигурация моделей
            'model_config': {
                'llm_model_name': model_config.llm_model_name,
                'embedding_model_name': model_config.embedding_model_name,
                'max_new_tokens': model_config.max_new_tokens,
                'temperature': model_config.temperature,
                'top_p': model_config.top_p,
                'repetition_penalty': model_config.repetition_penalty,
                'default_k_retrieve': model_config.default_k_retrieve,
                'max_context_length': model_config.max_context_length,
                'device': model_config.device,
                'mode': model_config.mode,
                'use_retriever': model_config.use_retriever,
                'use_8bit_quantization': model_config.use_8bit_quantization,
                'use_4bit_quantization': model_config.use_4bit_quantization,
            },
            
            # Конфигурация данных
            'data_config': {
                'data_dir': str(data_config.data_dir),
                'corpus_dir': str(data_config.corpus_dir),
                'cache_dir': str(data_config.cache_dir),
                'vectorstore_dir': str(data_config.vectorstore_dir),
                'experiments_dir': str(data_config.experiments_dir),
                'logs_dir': str(data_config.logs_dir),
                'metadata_dir': str(data_config.metadata_dir),
                'documents_dirs': {k: str(v) for k, v in data_config.documents_dirs.items()},
                'pdf_max_pages': data_config.pdf_max_pages,
                'pdf_max_doc_size': data_config.pdf_max_doc_size,
                'pdf_method': data_config.pdf_method,
                'chunk_size': data_config.chunk_size,
                'chunk_overlap': data_config.chunk_overlap,
                'chunk_separators': data_config.chunk_separators,
                'cache_enabled': data_config.cache_enabled,
                'force_reload': data_config.force_reload,
                'supported_extensions': data_config.supported_extensions,
            },
            
            # Конфигурация векторного хранилища
            'vectorstore_config': {
                'vector_store_type': vectorstore_config.vector_store_type,
                'faiss_index_type': vectorstore_config.faiss_index_type,
                'faiss_nlist': vectorstore_config.faiss_nlist,
                'faiss_nprobe': vectorstore_config.faiss_nprobe,
                'persist_dir': str(vectorstore_config.persist_dir),
                'save_on_update': vectorstore_config.save_on_update,
            },
            
            # Конфигурация экспериментов
            'experiment_config': {
                'ks_eval': experiment_config.ks_eval,
                'batch_size': experiment_config.batch_size,
                'save_results': experiment_config.save_results,
                'save_detailed_predictions': experiment_config.save_detailed_predictions,
                'detailed_logging': experiment_config.detailed_logging,
                'log_level': experiment_config.log_level,
            },
            
            # Конфигурация логирования
            'logging_config': {
                'log_level': logging_config.log_level,
                'log_format': logging_config.log_format,
                'log_date_format': logging_config.log_date_format,
                'log_to_file': logging_config.log_to_file,
                'log_to_console': logging_config.log_to_console,
                'log_file_name': logging_config.log_file_name,
            },
            
            # Конфигурация GigaChat (если используется)
            'gigachat_config': {
                'enabled': gigachat_config.enabled,
                'model': gigachat_config.model,
                'scope': gigachat_config.scope,
                'verify_ssl_certs': gigachat_config.verify_ssl_certs,
                'base_url': gigachat_config.base_url,
                'timeout': gigachat_config.timeout,
                'max_retries': gigachat_config.max_retries,
                # API ключ не сохраняем в явном виде по соображениям безопасности
                'api_key_configured': bool(gigachat_config.api_key),
            },
        }
        
        # Добавляем параметры из pipeline, если он передан
        if pipeline:
            full_config['pipeline_config'] = {
                'chunk_size': pipeline.chunk_size,
                'chunk_overlap': pipeline.chunk_overlap,
                'vector_store_initialized': pipeline.vector_store_initialized,
                'embedding_type': pipeline.vector_store_manager.embedding_type if pipeline.vector_store_manager else None,
                'llm_type': pipeline.llm_manager.model_type if pipeline.llm_manager else None,
            }
        
        return full_config
    
    def _save_full_config(self, experiment_id: str, pipeline: RAGPipeline = None):
        """
        Сохранение полной конфигурации в отдельный файл
        
        Args:
            experiment_id: ID эксперимента
            pipeline: RAGPipeline для получения дополнительных параметров
        """
        config_file = self.results_dir / f"{experiment_id}_config.json"
        
        full_config = self._get_full_config(pipeline)
        full_config['experiment_id'] = experiment_id
        full_config['saved_at'] = datetime.now().isoformat()
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📝 Конфигурация сохранена: {config_file}")
        return config_file

    def run_experiment(self,
                      pipeline: RAGPipeline,
                      test_samples: Dict[str, TestSample],
                      experiment_name: str,
                      retrieval_type: RetrievalType,
                      k_retrieve: int = 5,
                      save_results: bool = True,
                      detailed_logging: bool = True) -> ExperimentResult:
        """
        Запуск эксперимента
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"=" * 60)
        logger.info(f"Запуск эксперимента: {experiment_id}")
        logger.info(f"=" * 60)
        logger.info(f"Метод поиска: {retrieval_type.value}")
        logger.info(f"K для поиска: {k_retrieve}")
        logger.info(f"Количество тестов: {len(test_samples)}")

        start_time = time.time()

        # Сохраняем полную конфигурацию
        if save_results:
            self._save_full_config(experiment_id, pipeline)

        # Результаты
        predictions = {}          # {query_id: [retrieved_doc_ids]}
        detailed_results = {}     # {query_id: detailed_info}
        generated_answers = {}    # {query_id: generated_answer}

        # Статистика выполнения
        retrieval_times = []
        generation_times = []
        errors = []

        # Запуск для каждого тестового примера
        for idx, (qid, sample) in enumerate(test_samples.items(), 1):
            if detailed_logging:
                logger.info(f"\n[{idx}/{len(test_samples)}] Обработка {qid}: {sample.query[:50]}...")

            try:
                # Вызов основного метода RAG
                result = pipeline.process_query(sample.query, k=k_retrieve)

                # Извлекаем результаты поиска
                retrieval_result = result.retrieval_results
                retrieved_docs_ids = [doc['doc_id'] for doc in retrieval_result.retrieved_docs]
                retrieval_time = retrieval_result.retrieval_time
                retrieval_times.append(retrieval_time)

                predictions[qid] = retrieved_docs_ids

                # Генерация ответа
                generation_time = result.generation_time
                if sample.reference_answer:
                    generation_times.append(generation_time)
                    generated_answers[qid] = result.answer

                # Сохраняем детали
                detailed_results[qid] = {
                    'query': sample.query,
                    'query_id': qid,
                    'retrieved_docs': retrieval_result.retrieved_docs,
                    'retrieved_scores': retrieval_result.scores,
                    'retrieved_docs_ids': retrieved_docs_ids,
                    'relevant_docs': sample.relevant_docs,
                    'generated_answer': result.answer if sample.reference_answer else None,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'success': True
                }

                if detailed_logging:
                    logger.info(f"  ✓ Поиск: {retrieval_time:.3f} сек, найдено: {len(retrieved_docs_ids)} док.")
                    if sample.reference_answer:
                        logger.info(f"  ✓ Генерация: {generation_time:.3f} сек")

            except Exception as e:
                error_msg = f"Ошибка при обработке {qid}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

                detailed_results[qid] = {
                    'query': sample.query,
                    'query_id': qid,
                    'success': False,
                    'error': error_msg
                }
                predictions[qid] = []

        # Оценка качества поиска
        logger.info("\n" + "=" * 60)
        logger.info("Оценка качества поиска...")

        ground_truth = {qid: sample.relevant_docs for qid, sample in test_samples.items()}

        # Создаем базовый evaluator для метрик поиска и генерации
        evaluator = RAGEvaluator()  # Без LLM для базовых метрик

        retrieval_metrics = evaluator.evaluate_retrieval(
            predictions, 
            ground_truth,
            ks=[1, 3, 5, 10]
        )

        # Оценка качества генерации
        generation_metrics = {}
        if any(sample.reference_answer for sample in test_samples.values()):
            logger.info("Оценка качества генерации...")

            references = {
                qid: sample.reference_answer 
                for qid, sample in test_samples.items() 
                if sample.reference_answer and qid in generated_answers
            }

            if references and generated_answers:
                filtered_generated = {
                    qid: generated_answers[qid] 
                    for qid in references.keys() 
                    if qid in generated_answers
                }

                if filtered_generated:
                    gen_metrics_dict = evaluator.evaluate_generation(
                        filtered_generated,
                        references
                    )
                    generation_metrics = evaluator.calculate_average_metrics(gen_metrics_dict)

        execution_time = time.time() - start_time

        # Получаем полную конфигурацию для сохранения в результате
        full_system_config = self._get_full_config(pipeline)

        token_stats = pipeline.token_counter.get_stats_for_json() if hasattr(pipeline, 'token_counter') else {}

        # Подготовка результата
        experiment_result = ExperimentResult(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            config={
                'experiment_name': experiment_name,
                'retrieval_type': retrieval_type.value,
                'k_retrieve': k_retrieve,
                'num_test_samples': len(test_samples),
                'num_errors': len(errors),
                'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
                'avg_generation_time': sum(generation_times) / len(generation_times) if generation_times else 0,
                'llm_model': model_config.llm_model_name,
                'embedding_model': model_config.embedding_model_name,
                'chunk_size': data_config.chunk_size,
                'chunk_overlap': data_config.chunk_overlap,
                'temperature': model_config.temperature,
                'max_tokens': model_config.max_new_tokens,
                'device': model_config.device,
            },
            retrieval_metrics=retrieval_metrics.to_dict(),
            generation_metrics=generation_metrics,
            detailed_predictions=detailed_results,
            execution_time=execution_time,
            status=ExperimentStatus.COMPLETED if not errors else ExperimentStatus.FAILED,
            error_message="; ".join(errors[:5]) if errors else None,
            advanced_metrics={},  # Инициализируем пустым словарем
            token_stats=token_stats   # Добавляем токены в результат
        )

        # ==================== RAGAS МЕТРИКИ ====================
        # Собираем данные для продвинутых метрик
        questions = []
        answers = []
        contexts = []

        for qid, sample in test_samples.items():
            if qid in generated_answers and qid in detailed_results:
                questions.append(sample.query)
                answers.append(generated_answers[qid])

                # Получаем детальные результаты
                details = detailed_results[qid]
                retrieved_docs = details.get('retrieved_docs', [])

                # Извлекаем контексты
                context_texts = []
                for doc in retrieved_docs[:3]:
                    # Пробуем разные ключи для текста
                    text = None
                    if isinstance(doc, dict):
                        text = doc.get('content') or doc.get('text') or doc.get('page_content')
                    if text:
                        context_texts.append(text[:500])
                    else:
                        context_texts.append(str(doc)[:500])

                contexts.append(context_texts)

        advanced_metrics = None  # Инициализируем переменную
        # Вычисляем RAGAS метрики если есть данные
        if questions and answers and contexts:
            try:
                # Создаем LLM и эмбеддинги для RAGAS
                from langchain_gigachat import GigaChat
                from langchain_gigachat.embeddings import GigaChatEmbeddings
                from config import gigachat_config

                ragas_llm = GigaChat(
                    credentials=gigachat_config.api_key,
                    scope='GIGACHAT_API_B2B',
                    verify_ssl_certs=False,
                    model=gigachat_config.model,
                    temperature=model_config.ragas_temperature,
                    max_tokens=model_config.ragas_max_tokens
                )

                ragas_embeddings = GigaChatEmbeddings(
                    credentials=gigachat_config.api_key,
                    scope='GIGACHAT_API_B2B',
                    verify_ssl_certs=False
                )

                # Создаем evaluator с LLM для RAGAS
                ragas_evaluator = RAGEvaluator(llm=ragas_llm, embeddings=ragas_embeddings)

                # Вычисляем RAGAS метрики
                ragas_metrics = ragas_evaluator.evaluate_ragas_metrics(
                    questions=questions,
                    answers=answers,
                    contexts=contexts
                )

                # Сохраняем результаты
                #experiment_result.advanced_metrics = ragas_metrics.to_dict()
                advanced_metrics = ragas_metrics

                logger.info(f"📊 RAGAS метрики:")
                logger.info(f"  Faithfulness: {ragas_metrics.faithfulness:.3f}")
                logger.info(f"  Answer Relevancy: {ragas_metrics.answer_relevancy:.3f}")
                logger.info(f"  Context Relevancy: {ragas_metrics.context_relevancy:.3f}")

            except Exception as e:
                logger.warning(f"Не удалось вычислить RAGAS метрики: {e}")
                import traceback
                traceback.print_exc()
                advanced_metrics = None

        # Добавляем продвинутые метрики в результат (ВАЖНО!)
        if advanced_metrics is not None:
            experiment_result.advanced_metrics = advanced_metrics.to_dict()
        else:
            experiment_result.advanced_metrics = {}

        # Сохранение результатов
        if save_results:
            self.save_results(experiment_result, full_system_config)

        # Вывод результатов
        self.print_results(experiment_result)

        # Вывод RAGAS метрик если они есть
        if experiment_result.advanced_metrics:
            print(f"\n📊 RAGAS МЕТРИКИ:")
            print(f"  Faithfulness: {experiment_result.advanced_metrics.get('faithfulness', 0):.3f}")
            print(f"  Answer Relevancy: {experiment_result.advanced_metrics.get('answer_relevancy', 0):.3f}")
            print(f"  Context Relevancy: {experiment_result.advanced_metrics.get('context_relevancy', 0):.3f}")

        return experiment_result    
    
    def save_results(self, result: ExperimentResult, full_system_config: Dict[str, Any] = None):
        """
        Сохранение результатов эксперимента с полной конфигурацией
        
        Args:
            result: Результат эксперимента
            full_system_config: Полная конфигурация системы
        """
        # Получаем словарь с результатом
        result_dict = result.to_dict()
        
        # Добавляем полную конфигурацию системы
        if full_system_config:
            result_dict['system_configuration'] = full_system_config
        else:
            # Если конфигурация не передана, пробуем получить ее
            result_dict['system_configuration'] = self._get_full_config()
        
        # Добавляем метаинформацию
        result_dict['meta'] = {
            'exported_at': datetime.now().isoformat(),
            'exporter_version': '1.0',
            'file_format': 'experiment_result'
        }
        
        # Сохраняем полный результат в JSON
        result_file = self.results_dir / f"{result.experiment_id}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Результат сохранен: {result_file}")
        
        # Сохраняем упрощенную версию для быстрого просмотра
        summary_file = self.results_dir / f"{result.experiment_id}_summary.json"
        summary_dict = {
            'experiment_id': result.experiment_id,
            'timestamp': result.timestamp,
            'experiment_name': result.config.get('experiment_name'),
            'retrieval_type': result.config.get('retrieval_type'),
            'k_retrieve': result.config.get('k_retrieve'),
            'status': result.status.value,
            'map_score': result.retrieval_metrics.get('map', 0),
            'mrr_score': result.retrieval_metrics.get('mrr', 0),
            'execution_time': result.execution_time,
            'num_queries': result.config.get('num_test_samples'),
            'num_errors': result.config.get('num_errors'),
            'key_config': {
                'llm_model': result.config.get('llm_model'),
                'embedding_model': result.config.get('embedding_model'),
                'chunk_size': result.config.get('chunk_size'),
                'temperature': result.config.get('temperature'),
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Краткая сводка сохранена: {summary_file}")
        
        # Обновляем CSV сводку
        self._update_summary_csv(result)
        
        # Генерируем отчет
        self._generate_report(result)
    
    def _update_summary_csv(self, result: ExperimentResult):
        """
        Обновление CSV файла со сводкой всех экспериментов
        
        Args:
            result: Новый результат эксперимента
        """
        csv_file = self.experiments_dir / "experiments_summary.csv"
        
        # Создаем запись для CSV с расширенными данными
        record = {
            'experiment_id': result.experiment_id,
            'timestamp': result.timestamp,
            'experiment_name': result.config.get('experiment_name', ''),
            'retrieval_type': result.config.get('retrieval_type', ''),
            'k_retrieve': result.config.get('k_retrieve', 0),
            'num_queries': result.config.get('num_test_samples', 0),
            'num_errors': result.config.get('num_errors', 0),
            'execution_time': result.execution_time,
            'avg_retrieval_time': result.config.get('avg_retrieval_time', 0),
            'avg_generation_time': result.config.get('avg_generation_time', 0),
            'map_score': result.retrieval_metrics.get('map', 0),
            'mrr_score': result.retrieval_metrics.get('mrr', 0),
            # Конфигурационные параметры
            'llm_model': result.config.get('llm_model', ''),
            'embedding_model': result.config.get('embedding_model', ''),
            'chunk_size': result.config.get('chunk_size', 0),
            'chunk_overlap': result.config.get('chunk_overlap', 0),
            'temperature': result.config.get('temperature', 0),
            'max_tokens': result.config.get('max_tokens', 0),
            'device': result.config.get('device', ''),
        }
        
        # Добавляем precision@k
        precisions = result.retrieval_metrics.get('precision_at_k', {})
        for k, v in precisions.items():
            record[f'precision@{k}'] = v
        
        # Добавляем recall@k
        recalls = result.retrieval_metrics.get('recall_at_k', {})
        for k, v in recalls.items():
            record[f'recall@{k}'] = v
        
        # Добавляем ndcg@k
        ndcgs = result.retrieval_metrics.get('ndcg_at_k', {})
        for k, v in ndcgs.items():
            record[f'ndcg@{k}'] = v

        # Добавляем продвинутые метрики
        if result.advanced_metrics:
            record['faithfulness'] = result.advanced_metrics.get('faithfulness', 0)
            record['answer_relevancy'] = result.advanced_metrics.get('answer_relevancy', 0)
            record['context_relevancy'] = result.advanced_metrics.get('context_relevancy', 0)

        # Добавляем метрики генерации
        if result.generation_metrics:
            for metric, value in result.generation_metrics.items():
                record[metric] = value
        
        # Обновляем CSV
        df = pd.DataFrame([record])
        if csv_file.exists():
            existing_df = pd.read_csv(csv_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(csv_file, index=False)
        logger.info(f"📊 CSV обновлен: {csv_file}")
    
    def _generate_report(self, result: ExperimentResult):
        """
        Генерация подробного отчета по эксперименту
        
        Args:
            result: Результат эксперимента
        """
        report_file = self.reports_dir / f"{result.experiment_id}_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ОТЧЕТ ПО ЭКСПЕРИМЕНТУ\n")
            f.write(f"ID: {result.experiment_id}\n")
            f.write(f"Дата: {result.timestamp}\n")
            f.write("=" * 80 + "\n\n")
            
            # Конфигурация
            f.write("КОНФИГУРАЦИЯ:\n")
            f.write("-" * 40 + "\n")
            for key, value in result.config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Метрики поиска
            f.write("МЕТРИКИ ПОИСКА:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  MAP: {result.retrieval_metrics.get('map', 0):.4f}\n")
            f.write(f"  MRR: {result.retrieval_metrics.get('mrr', 0):.4f}\n")
            f.write("\n  Precision@k:\n")
            for k, v in result.retrieval_metrics.get('precision_at_k', {}).items():
                f.write(f"    P@{k}: {v:.4f}\n")
            f.write("\n  Recall@k:\n")
            for k, v in result.retrieval_metrics.get('recall_at_k', {}).items():
                f.write(f"    R@{k}: {v:.4f}\n")
            f.write("\n  NDCG@k:\n")
            for k, v in result.retrieval_metrics.get('ndcg_at_k', {}).items():
                f.write(f"    NDCG@{k}: {v:.4f}\n")
            f.write("\n")
            
            # Метрики генерации
            if result.generation_metrics:
                f.write("МЕТРИКИ ГЕНЕРАЦИИ:\n")
                f.write("-" * 40 + "\n")
                for metric, value in result.generation_metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
            
            # Детальные результаты по запросам
            f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ЗАПРОСАМ:\n")
            f.write("-" * 40 + "\n")
            
            for qid, details in result.detailed_predictions.items():
                if details.get('success', False):
                    f.write(f"\n{qid}: {details['query'][:80]}...\n")
                    f.write(f"  Релевантные документы: {len(details.get('relevant_docs', []))}\n")
                    f.write(f"  Найденные документы: {len(details.get('retrieved_docs_ids', []))}\n")
                    f.write(f"  Время поиска: {details.get('retrieval_time', 0):.3f} сек\n")
                    
                    # Показываем топ-3 найденных документа
                    retrieved = details.get('retrieved_docs', [])[:3]
                    if retrieved:
                        f.write(f"  Топ-3 найденных документов:\n")
                        for i, doc in enumerate(retrieved, 1):
                            f.write(f"    {i}. {doc['doc_id']} (score: {doc['score']:.3f})\n")
                else:
                    f.write(f"\n{qid}: ОШИБКА - {details.get('error', 'Unknown error')}\n")
        
        logger.info(f"📄 Отчет сохранен: {report_file}")
    
    def print_results(self, result: ExperimentResult):
        """
        Вывод результатов эксперимента в консоль
        
        Args:
            result: Результат эксперимента
        """
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА: {result.experiment_id}")
        print(f"{'='*60}")
        
        print(f"\n📊 МЕТРИКИ ПОИСКА:")
        print(f"  MAP: {result.retrieval_metrics.get('map', 0):.4f}")
        print(f"  MRR: {result.retrieval_metrics.get('mrr', 0):.4f}")
        
        print(f"\n  Precision@k:")
        for k, v in result.retrieval_metrics.get('precision_at_k', {}).items():
            print(f"    P@{k}: {v:.4f}")
        
        print(f"\n  Recall@k:")
        for k, v in result.retrieval_metrics.get('recall_at_k', {}).items():
            print(f"    R@{k}: {v:.4f}")
        
        print(f"\n  NDCG@k:")
        for k, v in result.retrieval_metrics.get('ndcg_at_k', {}).items():
            print(f"    NDCG@{k}: {v:.4f}")
        
        # Вывод RAGAS метрик (добавлено)
        if result.advanced_metrics:
            print(f"\n🤖 RAGAS МЕТРИКИ:")
            print(f"  Faithfulness: {result.advanced_metrics.get('faithfulness', 0):.4f}")
            print(f"  Answer Relevancy: {result.advanced_metrics.get('answer_relevancy', 0):.4f}")
            print(f"  Context Relevancy: {result.advanced_metrics.get('context_relevancy', 0):.4f}")

        if result.generation_metrics:
            print(f"\n🤖 МЕТРИКИ ГЕНЕРАЦИИ:")
            for metric, value in result.generation_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        print(f"\n⚙️ КОНФИГУРАЦИЯ:")
        print(f"  LLM модель: {result.config.get('llm_model', 'N/A')}")
        print(f"  Эмбеддинги: {result.config.get('embedding_model', 'N/A')}")
        print(f"  Chunk size: {result.config.get('chunk_size', 'N/A')}")
        print(f"  Temperature: {result.config.get('temperature', 'N/A')}")
        
        print(f"\n⏱️  Время выполнения: {result.execution_time:.2f} сек")
        print(f"📊 Количество тестов: {result.config.get('num_test_samples', 0)}")
        print(f"❌ Ошибок: {result.config.get('num_errors', 0)}")
        
        if result.status == ExperimentStatus.FAILED:
            print(f"\n⚠️ Эксперимент завершен с ошибками")
            if result.error_message:
                print(f"  Ошибки: {result.error_message}")
    
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """
        Загрузка сохраненного эксперимента

        Args:
            experiment_id: ID эксперимента

        Returns:
            ExperimentResult или None
        """
        result_file = self.results_dir / f"{experiment_id}.json"

        if not result_file.exists():
            logger.error(f"Эксперимент {experiment_id} не найден")
            return None

        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Восстанавливаем объект
        return ExperimentResult(
            experiment_id=data['experiment_id'],
            timestamp=data['timestamp'],
            config=data['config'],
            retrieval_metrics=data['retrieval_metrics'],
            generation_metrics=data['generation_metrics'],
            detailed_predictions=data['detailed_predictions'],
            execution_time=data['execution_time'],
            status=ExperimentStatus(data.get('status', 'completed')),
            error_message=data.get('error_message')
        )

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        Сравнение нескольких экспериментов

        Args:
            experiment_ids: Список ID экспериментов для сравнения

        Returns:
            DataFrame с результатами сравнения
        """
        comparisons = []

        for exp_id in experiment_ids:
            result = self.load_experiment(exp_id)
            if result:
                comparisons.append({
                    'experiment_id': exp_id,
                    'retrieval_type': result.config['retrieval_type'],
                    'k_retrieve': result.config['k_retrieve'],
                    'MAP': result.retrieval_metrics['map'],
                    'MRR': result.retrieval_metrics['mrr'],
                    'P@1': result.retrieval_metrics['precision_at_k'].get(1, 0),
                    'P@5': result.retrieval_metrics['precision_at_k'].get(5, 0),
                    'P@10': result.retrieval_metrics['precision_at_k'].get(10, 0),
                    'R@5': result.retrieval_metrics['recall_at_k'].get(5, 0),
                    'R@10': result.retrieval_metrics['recall_at_k'].get(10, 0),
                    'NDCG@5': result.retrieval_metrics['ndcg_at_k'].get(5, 0),
                    'execution_time': result.execution_time,
                    'num_queries': result.config['num_test_samples']
                })

                # Добавляем метрики генерации если есть
                if result.generation_metrics:
                    comparisons[-1]['rouge1'] = result.generation_metrics.get('rouge1', 0)
                    comparisons[-1]['rouge2'] = result.generation_metrics.get('rouge2', 0)
                    comparisons[-1]['bert_score'] = result.generation_metrics.get('bert_score', 0)

        return pd.DataFrame(comparisons)

    def get_best_experiment(self, metric: str = "MAP") -> Optional[ExperimentResult]:
        """
        Находит лучший эксперимент по указанной метрике

        Args:
            metric: Название метрики для сравнения

        Returns:
            Лучший результат эксперимента
        """
        csv_file = self.experiments_dir / "experiments_summary.csv"

        if not csv_file.exists():
            logger.error("Файл с результатами не найден")
            return None

        df = pd.read_csv(csv_file)

        if metric not in df.columns:
            logger.error(f"Метрика {metric} не найдена")
            return None

        best_row = df.loc[df[metric].idxmax()]
        best_id = best_row['experiment_id']

        return self.load_experiment(best_id)

    def export_to_latex(self, experiment_ids: List[str], output_file: Path) -> None:
        """
        Экспорт результатов в LaTeX таблицу

        Args:
            experiment_ids: Список ID экспериментов
            output_file: Путь для сохранения
        """
        df = self.compare_experiments(experiment_ids)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{|l|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Эксперимент & MAP & MRR & P@5 & R@5 & NDCG@5 \\\\\n")
            f.write("\\hline\n")

            for _, row in df.iterrows():
                f.write(f"{row['experiment_id']} & ")
                f.write(f"{row['MAP']:.4f} & ")
                f.write(f"{row['MRR']:.4f} & ")
                f.write(f"{row['P@5']:.4f} & ")
                f.write(f"{row['R@5']:.4f} & ")
                f.write(f"{row['NDCG@5']:.4f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write(f"\\caption{{Сравнение экспериментов}}\n")
            f.write("\\end{table}\n")

        logger.info(f"✅ LaTeX таблица сохранена: {output_file}")

def generate_report_after_experiment():
    """Генерация Excel отчета после завершения эксперимента"""
    from excel_reporter import ExcelReporter
    
    print("\n" + "="*60)
    print("📊 Генерация Excel отчета по всем экспериментам")
    print("="*60)
    
    reporter = ExcelReporter()
    output_file = reporter.generate_report()
    
    if output_file and output_file.exists():
        print(f"\n✅ Excel отчет создан: {output_file}")
    else:
        print("\n❌ Не удалось создать Excel отчет")

def generate_excel_report():
    """Генерация Excel отчета по всем экспериментам"""
    try:
        from excel_reporter import generate_experiments_summary
        generate_experiments_summary()
    except ImportError as e:
        logger.warning(f"Не удалось импортировать excel_reporter: {e}")
    except Exception as e:
        logger.error(f"Ошибка при генерации Excel отчета: {e}")

def main(args=None):
    import argparse
    from config import data_config, model_config


    parser = argparse.ArgumentParser(description="Запуск экспериментов RAG")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Директория с PDF документами")
    parser.add_argument("--testset", type=str, required=True, help="Путь к JSON файлу с тестовыми запросами")
    parser.add_argument("--name", type=str, required=True, help="Название эксперимента")
    parser.add_argument("--k", type=int, default=5, help="K для поиска")
    parser.add_argument("--chunk_size", type=int, default=None, help="Размер чанка")
    parser.add_argument("--chunk_overlap", type=int, default=None, help="Перекрытие чанков")
    parser.add_argument("--embedding_type", type=str, default="gigachat", choices=["huggingface", "gigachat"],
                        help="Тип эмбеддингов")
    parser.add_argument("--llm_type", type=str, default="gigachat", choices=["local", "gigachat", "openai"],
                        help="Тип LLM")
    parser.add_argument("--force_reload", action="store_true", help="Принудительно пересоздать FAISS индекс")


    # Если передан список аргументов, используем его, иначе берем из sys.argv
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()    

    # Загрузка тестовых данных
    loader = TestDataLoader()
    test_samples = loader.load_from_json(Path(args.testset))


    # Создание пайплайна с параметрами из аргументов или конфига
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size or data_config.chunk_size,
        chunk_overlap=args.chunk_overlap or data_config.chunk_overlap,
        embedding_type=args.embedding_type,
        llm_type=args.llm_type
    )
    token_counter = pipeline.token_counter


    # Получаем баланс до и после
    if pipeline.gigachat_client:
        initial_balance = pipeline.get_balance_info()
        logger.info(f"Initial balance: {initial_balance}")
    else:
        initial_balance = None
        logger.warning("No GigaChat client available, skipping balance tracking")

    # Загрузка документов
    pdf_dir = Path(args.pdf_dir)
    if pdf_dir.exists():
        pipeline.load_from_pdf_directory_with_metadata(
            pdf_dir,
            recursive=True,
            force_reload=args.force_reload
        )
    else:
        logger.error(f"Директория {pdf_dir} не найдена")
        exit(1)


    # Запуск эксперимента
    runner = ExperimentRunner()
    result = runner.run_experiment(
        pipeline=pipeline,
        test_samples=test_samples,
        experiment_name=args.name,
        retrieval_type=RetrievalType.DENSE,  # для совместимости
        k_retrieve=args.k,
        detailed_logging=True
    )


    # Вывод итогов
    runner.print_results(result)

    # Выводим статистику токенов
    token_counter.print_summary()

    # Сохраняем в файл
    token_counter.save_to_file("token_stats.json")

    # Получаем конечный баланс





    if pipeline.gigachat_client:
        final_balance = pipeline.get_balance_info()
        logger.info(f"Final balance: {final_balance}")
    else:
        final_balance = None
    
    # Расчет дельты с проверкой
    if initial_balance is not None and final_balance is not None:
        delta = token_counter.calculate_balance_delta(initial_balance, final_balance)
        
        if delta.get('has_data'):
            logger.info(f"Balance delta calculated: {delta}")

            # Статистика
            stats = token_counter.get_stats_for_json()
            print(f"Статистика: {stats}")

        else:
            logger.warning(f"Could not calculate balance delta: {delta.get('error')}")
    else:
        logger.warning("Skipping balance delta calculation due to missing balance data")

    #generate_report_after_experiment()
    generate_excel_report()
if __name__ == "__main__":
    #main()
    #main(["--pdf_dir", "data/domain_3_WikiEval_1row/books", "--testset", "data/domain_3_WikiEval_1row/testset.json", "--name", "WikiEval_1row_experiment", "--force_reload"])
    main(["--pdf_dir", "data/domain_4_WikiEval_2row/books", "--testset", "data/domain_4_WikiEval_2row/testset.json", "--name", "domain_4_WikiEval_2row_experiment", "--force_reload"])
    #main(["--pdf_dir", "data/domain_4_WikiEval_2row/books", "--testset", "data/domain_4_WikiEval_2row/testset.json", "--name", "wikieval_experiment", "--force_reload"])
    #main(["--pdf_dir", "data/domain_5_WikiEval_full/books", "--testset", "data/domain_5_WikiEval_full/testset.json", "--name", "wikieval_experiment_full", "--force_reload"])
    #main(["--pdf_dir", "data/domain_6_WikiEval_5row/books", "--testset", "data/domain_6_WikiEval_5row/testset.json", "--name", "wikieval_experiment_WikiEval_5row", "--force_reload"])

#& d:/study/2026-03_RAG/.venv/Scripts/python.exe d:/study/2026-03_RAG/experiment.py --pdf_dir "data/domain_2_Debug/books" --testset "testset.json" --name "my_experiment" --force_reload
#& d:/study/2026-03_RAG/.venv/Scripts/python.exe d:/study/2026-03_RAG/experiment.py --pdf_dir data/domain_3_WikiEval_1row/books  --testset data/domain_3_WikiEval_1row/testset.json --name wikieval_experiment  --force_reload