"""Entry point for python -m rag_gigachat"""

import sys
from pathlib import Path

# Добавить root в sys.path для доступа к data/, logs/ и т.д.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Импортировать и запустить основное приложение
if __name__ == "__main__":
    # Используем app.py из root как entry point
    from app import main
    sys.exit(main(sys.argv[1:]))
