"""
app.py RAG Application - Единая точка входа
Поддерживает три режима работы:
1. UI режим (по умолчанию) - интерактивный Streamlit интерфейс
2. Single Query режим - ответ на один вопрос
3. Experiment режим - запуск серии экспериментов
"""
import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

# Подавляем шумные предупреждения от transformers и HuggingFace
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from rag_gigachat.models import RetrievalType
from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.config import model_config, data_config, vectorstore_config, experiment_config, logging_config, gigachat_config

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, logging_config.log_level),
    format=logging_config.log_format,
)
logger = logging.getLogger(__name__)


class RAGApp:
    """Основной класс приложения RAG"""

    def __init__(self):
        """Инициализация приложения"""
        self.pipeline = None
        logger.info("RAGApp инициализирован")

    def initialize(self,
                   retrieval_type: RetrievalType = RetrievalType.DENSE,
                   documents_path: Optional[Path] = None,
                   dense_weight: float = 1.0,
                   sparse_weight: float = 0.0) -> bool:
        """
        Инициализация pipeline с документами

        Args:
            retrieval_type: Тип поиска
            documents_path: Путь к папке с документами
            dense_weight: Вес плотного поиска
            sparse_weight: Вес разреженного поиска

        Returns:
            Успешность инициализации
        """
        try:
            logger.info(f"Инициализация RAGPipeline: retrieval_type={retrieval_type}")

            self.pipeline = RAGPipeline(
                retrieval_type=retrieval_type,
                embedding_type="huggingface",
                llm_type="local"
            )

            if documents_path and documents_path.exists():
                logger.info(f"Загрузка документов из {documents_path}")
                self.pipeline.load_from_pdf_directory_with_metadata(
                    documents_path,
                    recursive=True,
                    force_reload=False
                )

            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}", exc_info=True)
            return False

    def process_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Обработка одного запроса

        Args:
            query: Текст запроса
            k: Количество документов для поиска

        Returns:
            Словарь с результатами
        """
        if self.pipeline is None:
            return {"error": "Pipeline не инициализирован"}

        if not self.pipeline.vector_store_initialized:
            return {"error": "Vector store не инициализирован. Загрузите документы."}

        try:
            result = self.pipeline.process_query(query, k=k)

            return {
                "query": query,
                "answer": result.answer,
                "generation_time": result.generation_time,
                "tokens_generated": result.tokens_generated,
                "retrieved_docs": [
                    {
                        "doc_id": doc["doc_id"],
                        "score": doc.get("score", 1.0),
                        "text_preview": doc["text"][:200] if doc.get("text") else ""
                    }
                    for doc in result.retrieval_results.retrieved_docs
                ]
            }
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
            return {"error": str(e)}


def run_streamlit_ui():
    """Запуск Streamlit UI с ui_components.py"""
    print("🚀 Запуск Streamlit UI...")

    current_dir = Path(__file__).parent
    ui_file = current_dir / "src" / "rag_gigachat" / "ui" / "streamlit_app.py"

    if not ui_file.exists():
        print(f"❌ Ошибка: Файл {ui_file} не найден!")
        return False

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
        process = subprocess.Popen(
            streamlit_cmd,
            cwd=str(current_dir),
            env=os.environ.copy()
        )
        # Ждем завершения процесса пользователем (Ctrl+C)
        # Используем poll() чтобы проверять статус периодически
        while True:
            if process.poll() is not None:
                # Процесс неожиданно завершился, перезапустим его
                print("⚠️  Streamlit процесс завершился, перезапускаю...")
                process = subprocess.Popen(
                    streamlit_cmd,
                    cwd=str(current_dir),
                    env=os.environ.copy()
                )
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Остановка Streamlit...")
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("✅ Streamlit остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска Streamlit: {e}")
        return False

    return True


def run_query_mode(args):
    """Режим 2: Ответ на один вопрос"""
    print("🎯 Запуск режима Single Query...")

    app = RAGApp()

    print(f"📝 Вопрос: {args.query}")
    print(f"📁 Директория документов: {args.documents}")
    print(f"🔧 Метод поиска: {args.retrieval_type}")
    print(f"🔍 K для поиска: {args.k}")
    print("-" * 60)

    retrieval_type = RetrievalType(args.retrieval_type)
    documents_path = Path(args.documents) if args.documents else None

    success = app.initialize(
        retrieval_type=retrieval_type,
        documents_path=documents_path
    )

    if not success:
        print("❌ Ошибка инициализации системы")
        return

    result = app.process_query(args.query, k=args.k)

    if "error" in result:
        print(f"❌ Ошибка: {result['error']}")
        return

    # Вывод результата
    print(f"\n{'='*60}")
    print(f"🤖 ОТВЕТ:")
    print(f"{'='*60}")
    print(f"{result['answer']}\n")

    print(f"{'='*60}")
    print(f"📚 НАЙДЕННЫЕ ДОКУМЕНТЫ ({len(result['retrieved_docs'])}):")
    print(f"{'='*60}")
    for i, doc in enumerate(result['retrieved_docs'], 1):
        print(f"\n[{i}] {doc['doc_id']}")
        print(f"    Score: {doc['score']:.3f}")
        print(f"    Preview: {doc['text_preview']}...")

    print(f"\n{'='*60}")
    print(f"⏱️  Время обработки: {result['generation_time']:.2f} сек")
    print(f"🔢 Токенов в ответе: {result['tokens_generated']}")
    print(f"{'='*60}\n")

    # Сохранение результата если указан выход
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"💾 Результат сохранен в: {output_path}")


def main():
    """Основная функция с парсингом аргументов"""
    parser = argparse.ArgumentParser(
        description="RAG Application - единый вход для UI и запросов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # UI режим (по умолчанию)
  python app.py

  # Режим одного запроса с документами
  python app.py --mode query --query "Что такое RAG?" --documents data/domain_2_Debug/books

  # Сохранение результата
  python app.py --mode query --query "Вопрос?" --documents data/domain_2_Debug/books --output result.json
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["ui", "query"],
        default="ui",
        help="Режим работы: ui (интерфейс), query (один запрос)"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Текст запроса (для режима query)"
    )

    parser.add_argument(
        "--documents",
        type=str,
        default="data/domain_2_Debug/books",
        help="Путь к директории с PDF документами"
    )

    parser.add_argument(
        "--retrieval_type",
        type=str,
        choices=["dense", "sparse", "hybrid"],
        default="dense",
        help="Тип поиска документов"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Количество документов для поиска"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Путь для сохранения результата (JSON)"
    )

    parser.add_argument(
        "--test-question",
        type=str,
        help="Автоматически отправить тестовый вопрос при запуске UI (debug режим)"
    )

    args = parser.parse_args()

    print(f"💻 Используется устройство: {model_config.device}")

    if args.mode == "ui":
        # 🧪 Передать тестовый вопрос в session_state через env var
        if args.test_question:
            os.environ["RAG_TEST_QUESTION"] = args.test_question
            logger.info(f"🧪 Тестовый вопрос установлен: {args.test_question}")
        success = run_streamlit_ui()
        if not success:
            sys.exit(1)
    elif args.mode == "query":
        if not args.query:
            print("❌ Для режима query необходимо указать --query")
            parser.print_help()
            sys.exit(1)
        run_query_mode(args)


if __name__ == "__main__":
    main()
