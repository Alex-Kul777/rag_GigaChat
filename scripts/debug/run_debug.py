#!/usr/bin/env python3
"""
run_debug.py - Debug Runner для запуска Streamlit с усиленным логированием
"""
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import signal

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Формат логов для сессии
now = datetime.now()
session_timestamp = now.strftime("%Y%m%d_%H%M%S")
session_log = LOGS_DIR / f"session_{session_timestamp}.log"


def run_streamlit():
    """Запуск Streamlit с логированием в файл"""
    env = os.environ.copy()
    env["RAG_DEBUG"] = "true"
    env["RAG_LOG_LEVEL"] = "DEBUG"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "src" / "rag_gigachat" / "ui" / "streamlit_app.py"),
        "--logger.level=debug",
    ]

    print(f"🚀 Starting Streamlit debug session...")
    print(f"📝 Log file: {session_log}")
    print(f"💡 Press Ctrl+C to stop and run: python scripts/debug/mine_process.py")
    print("-" * 60)

    with open(session_log, "w") as log_file:
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            proc.wait()
        except KeyboardInterrupt:
            print("\n⏹️  Stopping Streamlit...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"✅ Session ended. Logs saved to {session_log}")
            print(f"📊 Next: python scripts/debug/mine_process.py")


if __name__ == "__main__":
    run_streamlit()
