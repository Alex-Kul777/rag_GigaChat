"""
app.py RAG Application - Единая точка входа
Поддерживает три режима работы:
1. UI режим (по умолчанию) - интерактивный Streamlit интерфейс
2. Single Query режим - ответ на один вопрос
3. Experiment режим - запуск серии экспериментов
git commit -m "Initial commit"
 git add .  

 git remote add origin https://git.standard-data.ru/course_ragsystem_20260323/rag_akulikov.git
"""
import sys
import argparse
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

#from models import RetrievalType
#from rag_core import RAGPipeline
#from data_loader import TestDataLoader
#from experiment import ExperimentRunner
#from config import config
from config import model_config, data_config, vectorstore_config, experiment_config, logging_config, gigachat_config


def run_streamlit_ui():
    """Запуск Streamlit UI с ui_components.py"""
    print("🚀 Запуск Streamlit UI...")
    
    # Получаем путь к текущей директории
    current_dir = Path(__file__).parent
    ui_file = current_dir / "ui_streamlit.py"
    
    # Проверяем существование файла
    if not ui_file.exists():
        print(f"❌ Ошибка: Файл {ui_file} не найден!")
        print("Убедитесь, что ui_components.py существует в той же директории")
        return False
    
    # Формируем команду для запуска Streamlit
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ui_file),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    print(f"📁 Запуск: {' '.join(streamlit_cmd)}")
    print("🌐 Streamlit UI будет доступен по адресу: http://localhost:8501")
    print("⏹️  Для остановки нажмите Ctrl+C")
    print("-" * 60)
    
    try:
        # Запускаем Streamlit как дочерний процесс
        process = subprocess.Popen(
            streamlit_cmd,
            cwd=str(current_dir),
            env=os.environ.copy()
        )
        
        # Ожидаем завершения процесса
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Остановка Streamlit...")
        process.terminate()
        process.wait()
        print("✅ Streamlit остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска Streamlit: {e}")
        return False
    
    return True



def run_query_mode(args):
    """Режим 2: Ответ на один вопрос"""
    print("🎯 Запуск режима Single Query...")
    
    # Инициализация
    app = RAGApp()
    
    print(f"📝 Вопрос: {args.query}")
    print(f"🔧 Метод поиска: {args.retrieval_type}")
    print(f"🔍 K для поиска: {args.k}")
    print("-" * 60)
    
    # Инициализация с указанными параметрами
    retrieval_type = RetrievalType(args.retrieval_type)
    
    success = app.initialize(
        retrieval_type=retrieval_type,
        documents_path=Path(args.documents) if args.documents else None,
        dense_weight=args.dense_weight,
        sparse_weight=1.0 - args.dense_weight if args.retrieval_type == "hybrid" else 0.0
    )
    
    if not success:
        print("❌ Ошибка инициализации системы")
        return
    
    # Обработка запроса
    result = app.process_query(args.query, k=args.k)
    
    if "error" in result:
        print(f"❌ Ошибка: {result['error']}")
        return
    
    # Вывод результата
    print(f"\n🤖 Ответ:\n{result['answer']}")
    print(f"\n📚 Использованные документы:")
    for i, doc in enumerate(result['retrieved_docs'], 1):
        print(f"  {i}. {doc['doc_id']} (score: {doc['score']:.3f})")
        print(f"     {doc['text_preview']}")
    
    # Сохранение результата
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результат сохранен в: {output_path}")


def run_experiment_mode(args):
    """Режим 3: Запуск экспериментов"""
    print("🔬 Запуск режима экспериментов...")
    
    # Инициализация
    app = RAGApp()
    
    print(f"📁 Тестовый набор: {args.testset}")
    print(f"📝 Название эксперимента: {args.experiment_name}")
    print(f"🔧 Метод поиска: {args.retrieval_type}")
    print(f"🔍 K для поиска: {args.k}")
    print("-" * 60)
    
    # Инициализация
    retrieval_type = RetrievalType(args.retrieval_type)
    
    success = app.initialize(
        retrieval_type=retrieval_type,
        documents_path=Path(args.documents) if args.documents else None,
        dense_weight=args.dense_weight,
        sparse_weight=1.0 - args.dense_weight if args.retrieval_type == "hybrid" else 0.0
    )
    
    if not success:
        print("❌ Ошибка инициализации системы")
        return
    
    # Запуск эксперимента
    result = app.run_experiment(
        testset_path=Path(args.testset),
        experiment_name=args.experiment_name,
        retrieval_type=retrieval_type,
        k_retrieve=args.k,
        save_results=not args.no_save
    )
    
    if "error" in result:
        print(f"❌ Ошибка: {result['error']}")
        return
    
    # Вывод результатов
    print(f"\n📊 Результаты эксперимента:")
    print(f"  Эксперимент ID: {result['experiment_id']}")
    print(f"  MAP: {result['metrics']['map']:.4f}")
    print(f"  MRR: {result['metrics']['mrr']:.4f}")
    print(f"  Precision@5: {result['metrics']['precision_at_5']:.4f}")
    print(f"  Recall@5: {result['metrics']['recall_at_5']:.4f}")
    
    if result['generation_metrics']:
        print(f"\n🤖 Метрики генерации:")
        for metric, value in result['generation_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\n⏱️  Время выполнения: {result['execution_time']:.2f} сек")


def main():
    """Основная функция с парсингом аргументов"""
    parser = argparse.ArgumentParser(
        description="RAG Application - единый вход для UI, запросов и экспериментов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # UI режим (по умолчанию)
  python app.py
  
  # Режим одного запроса
  python app.py --mode query --query "Что такое нейросети?" --retrieval_type dense
  
  # Режим экспериментов
  python app.py --mode experiment --testset test_queries.json --experiment_name exp_1 --retrieval_type hybrid
  
  # Сохранение результата запроса в файл
  python app.py --mode query --query "Что такое RAG?" --output result.json
        """
    )
    
    # Режим работы
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["ui", "query", "experiment"],
        default="ui",
        help="Режим работы: ui (интерфейс), query (один запрос), experiment (эксперименты)"
    )
    
                                                      
    # Параметры для режима query
    parser.add_argument(
        "--query",
        type=str,
        help="Текст запроса (для режима query)"
    )
    
    args = parser.parse_args()
    
    # Определяем устройство
  
    print(f"💻 Используется устройство: {model_config.device}")
    
    # Запуск соответствующего режима
    if args.mode == "ui":
        success = run_streamlit_ui()
        if not success:
            sys.exit(1)

    elif args.mode == "query":
        if not args.query:
            print("❌ Для режима query необходимо указать --query")
            return
        run_query_mode(args)
    elif args.mode == "experiment":
        if not args.testset:
            print("❌ Для режима experiment необходимо указать --testset")
            return
        run_experiment_mode(args)


if __name__ == "__main__":
    main()