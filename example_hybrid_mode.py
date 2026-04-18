#!/usr/bin/env python3
"""
example_hybrid_mode.py - Примеры использования гибридного режима offline/online

Показывает:
1. Проверку наличия модели в кэше
2. Автоматическое скачивание модели
3. Работу LLMManager с автоматическим переключением режимов
"""

import os
import logging
from pathlib import Path

# Добавляем src в путь
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Настройка логирования для просмотра деталей
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

from rag_gigachat.core.model_downloader import (
    is_model_cached,
    is_offline_mode_enabled,
    check_and_download_model,
    get_hf_cache_dir,
    set_offline_mode
)


def example_1_check_cache():
    """Пример 1: Проверка наличия модели в кэше"""
    print("\n" + "="*60)
    print("ПРИМЕР 1: Проверка наличия модели в кэше")
    print("="*60)

    model_name = "gpt2"
    print(f"\n📌 Проверяем модель: {model_name}")
    print(f"   Директория кэша: {get_hf_cache_dir()}")

    if is_model_cached(model_name):
        print(f"   ✅ Модель найдена в кэше")
    else:
        print(f"   ❌ Модель отсутствует в кэше")


def example_2_offline_mode_status():
    """Пример 2: Проверка статуса оффлайн-режима"""
    print("\n" + "="*60)
    print("ПРИМЕР 2: Статус оффлайн-режима")
    print("="*60)

    print(f"\n📌 Текущий статус оффлайн-режима:")
    if is_offline_mode_enabled():
        print(f"   🔵 Режим: ОФФЛАЙН (HF_HUB_OFFLINE=1)")
    else:
        print(f"   🟢 Режим: ОНЛАЙН (HF_HUB_OFFLINE=0)")

    print(f"\n📌 Переключение режимов:")
    print(f"   Включаем онлайн-режим...")
    was_set, old_value = set_offline_mode(False)
    print(f"   → HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE')}")

    print(f"\n   Включаем оффлайн-режим...")
    set_offline_mode(True)
    print(f"   → HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE')}")


def example_3_download_model():
    """Пример 3: Проверка и скачивание модели"""
    print("\n" + "="*60)
    print("ПРИМЕР 3: Проверка и скачивание модели")
    print("="*60)

    model_name = "gpt2"
    print(f"\n📌 Работаем с моделью: {model_name}")
    print(f"   Текущий режим: {'ОФФЛАЙН' if is_offline_mode_enabled() else 'ОНЛАЙН'}")

    success = check_and_download_model(model_name)
    if success:
        print(f"\n   ✅ Модель готова к использованию!")
        if is_model_cached(model_name):
            print(f"   ✓ Модель теперь в кэше")
    else:
        print(f"\n   ❌ Не удалось загрузить модель")


def example_4_context_managers():
    """Пример 4: Использование как контекст-менеджера (продвинуто)"""
    print("\n" + "="*60)
    print("ПРИМЕР 4: Управление состоянием оффлайн-режима")
    print("="*60)

    print(f"\n📌 Исходное состояние: HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE')}")

    # Сохраняем текущее состояние
    was_set, old_value = set_offline_mode(False)
    print(f"   Переключили на онлайн: HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE')}")

    # Выполняем операцию
    print(f"   [Выполняем операцию в онлайн-режиме...]")

    # Восстанавливаем состояние
    if was_set:
        set_offline_mode(old_value == "1")
    else:
        set_offline_mode(True)
    print(f"   Восстановили состояние: HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE')}")


if __name__ == "__main__":
    print("\n🚀 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ГИБРИДНОГО РЕЖИМА OFFLINE/ONLINE\n")

    try:
        # Убеждаемся, что оффлайн-режим включен
        set_offline_mode(True)

        # Примеры
        example_1_check_cache()
        example_2_offline_mode_status()
        # example_3_download_model()  # Раскомментируй для скачивания моделей
        example_4_context_managers()

        print("\n" + "="*60)
        print("✅ ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ")
        print("="*60)
        print(f"\n💡 Для автоматического использования в приложении:")
        print(f"   • LLMManager автоматически вызывает check_and_download_model()")
        print(f"   • VectorStoreManager автоматически вызывает check_and_download_model()")
        print(f"   • Не требует ручного управления в коде приложения\n")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
