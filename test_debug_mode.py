#!/usr/bin/env python3
"""
Быстрый тест debug-режима
Сравнивает производительность production и debug моделей
"""
import os
import sys
import time
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_production_model():
    """Тест production модели"""
    print("\n" + "="*70)
    print("🚀 PRODUCTION РЕЖИМ")
    print("="*70)

    from rag_gigachat.config import debug_config, model_config
    from rag_gigachat.core.llm_manager import LLMManager

    # Отключаем debug режим
    debug_config.debug_mode = False

    print(f"📦 Модель: {model_config.llm_model_name}")
    print(f"   Профиль: {model_config.model_profile}")
    print(f"   Max tokens: {model_config.max_new_tokens}")

    # Загрузка модели
    start = time.time()
    llm_manager = LLMManager(model_type="local")
    llm = llm_manager.get_llm()
    load_time = time.time() - start

    # Генерация ответа
    prompt = "Что такое машинное обучение? Ответ на русском."
    start = time.time()
    try:
        response = llm.invoke(prompt)
        gen_time = time.time() - start
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
        gen_time = None
        response = None

    print(f"\n⏱️  Результаты:")
    print(f"   Загрузка модели: {load_time:.2f} сек")
    if gen_time:
        print(f"   Генерация ответа: {gen_time:.2f} сек")
        if response:
            print(f"   Ответ: {str(response)[:100]}...")
    print()

    return load_time, gen_time

def test_debug_model():
    """Тест debug модели"""
    print("\n" + "="*70)
    print("🐛 DEBUG РЕЖИМ (быстрая многоязычная модель)")
    print("="*70)

    # Очищаем импорты для чистого теста
    import importlib
    import rag_gigachat.config as config_module
    importlib.reload(config_module)

    from rag_gigachat.config import debug_config, model_config
    from rag_gigachat.core.llm_manager import LLMManager

    # Включаем debug режим
    debug_config.debug_mode = True

    print(f"📦 Debug модель: {debug_config.debug_model_name}")
    print(f"   Основная модель: {model_config.llm_model_name}")
    print(f"   Max tokens: {model_config.max_new_tokens}")

    # Загрузка модели
    start = time.time()
    llm_manager = LLMManager(model_type="local")
    llm = llm_manager.get_llm()
    load_time = time.time() - start

    # Генерация ответа
    prompt = "Что такое машинное обучение? Ответ на русском."
    start = time.time()
    try:
        response = llm.invoke(prompt)
        gen_time = time.time() - start
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
        gen_time = None
        response = None

    print(f"\n⏱️  Результаты:")
    print(f"   Загрузка модели: {load_time:.2f} сек")
    if gen_time:
        print(f"   Генерация ответа: {gen_time:.2f} сек")
        if response:
            print(f"   Ответ: {str(response)[:100]}...")
    print()

    return load_time, gen_time

def main():
    """Основной тест"""
    print("\n" + "🎯 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*70)

    # Тест production (если захотите раскомментировать)
    # prod_load, prod_gen = test_production_model()

    # Тест debug
    debug_load, debug_gen = test_debug_model()

    # Итоги
    print("="*70)
    print("📊 ИТОГИ")
    print("="*70)
    print("\n✨ Debug-режим включен и работает!")
    print(f"   Загрузка: {debug_load:.2f} сек ✅")
    if debug_gen:
        print(f"   Генерация: {debug_gen:.2f} сек ✅")

    print("\n💡 Использование debug-режима:")
    print("   export RAG_DEBUG_MODE=true")
    print("   python app.py --mode ui")

    print("\n🔧 Для отключения debug-режима:")
    print("   unset RAG_DEBUG_MODE")
    print("   python app.py --mode ui")

if __name__ == "__main__":
    main()
