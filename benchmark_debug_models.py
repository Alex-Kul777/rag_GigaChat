#!/usr/bin/env python3
"""
Быстрый бенчмарк моделей для debug-режима
Критерии: время загрузки < 5 сек, генерация < 2 сек, память < 1 GB
"""
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List
import torch

# Русский тестовый текст
TEST_PROMPT_RU = "Ответьте кратко: Что такое искусственный интеллект?"
TEST_PROMPT_EN = "Answer briefly: What is artificial intelligence?"

#候选модели для тестирования
MODELS_TO_TEST = [
    ("distilgpt2", "text-generation", "eng", 82),  # MB
    ("facebook/opt-125m", "text-generation", "eng", 250),
    ("google/flan-t5-small", "text2text-generation", "multi", 300),
    ("cointegrated/rubert-tiny2", "feature-extraction", "rus", 30),
]

def get_memory_usage() -> float:
    """Получить использование памяти в GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9

def benchmark_model(model_name: str, task: str, lang: str, estimated_size_mb: int):
    """Бенчмарк одной модели"""
    print(f"\n{'='*70}")
    print(f"📊 Тестирование: {model_name}")
    print(f"   Lang: {lang}, Est. size: {estimated_size_mb}MB")
    print('='*70)

    try:
        from transformers import pipeline, AutoTokenizer, AutoModel

        # 1. Память ДО загрузки
        mem_before = get_memory_usage()
        print(f"🔍 Память до загрузки: {mem_before:.2f} GB")

        # 2. Загрузка модели
        load_start = time.time()

        try:
            if task == "feature-extraction":
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            else:
                pipe = pipeline(task, model=model_name, device=-1)  # CPU only

            load_time = time.time() - load_start
            print(f"✅ Загрузка за {load_time:.2f} сек")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return None

        # 3. Память ПОСЛЕ загрузки
        mem_after = get_memory_usage()
        mem_used = mem_after - mem_before
        print(f"💾 Память после загрузки: {mem_after:.2f} GB (использовано: +{mem_used:.2f} GB)")

        # 4. Генерация текста
        prompt = TEST_PROMPT_RU if lang == "rus" else TEST_PROMPT_EN

        gen_start = time.time()
        try:
            if task == "feature-extraction":
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                response = f"[Embeddings generated, shape: {outputs.last_hidden_state.shape}]"
            else:
                output = pipe(prompt, max_new_tokens=20, do_sample=False)
                response = output[0]['generated_text'] if isinstance(output, list) else str(output)

            gen_time = time.time() - gen_start
            print(f"⏱️  Генерация за {gen_time:.2f} сек")
            print(f"📝 Ответ: {response[:80]}...")
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            gen_time = None

        # 5. Итоги
        print(f"\n✨ ИТОГИ:")
        print(f"   Загрузка: {load_time:.2f}s {'✅' if load_time < 5 else '❌'}")
        print(f"   Генерация: {gen_time:.2f}s {'✅' if gen_time and gen_time < 2 else '⚠️ '}")
        print(f"   Память: {mem_used:.2f}GB {'✅' if mem_used < 1 else '⚠️ '}")

        return {
            "model": model_name,
            "lang": lang,
            "task": task,
            "load_time": load_time,
            "gen_time": gen_time,
            "mem_used": mem_used,
            "estimated_size_mb": estimated_size_mb,
        }

    except Exception as e:
        print(f"❌ Критическая ошибка при тестировании {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n🚀 БЕНЧМАРК МОДЕЛЕЙ ДЛЯ DEBUG-РЕЖИМА")
    print("Критерии: загрузка < 5с, генерация < 2с, память < 1GB\n")

    results = []

    for model_name, task, lang, est_size in MODELS_TO_TEST:
        result = benchmark_model(model_name, task, lang, est_size)
        if result:
            results.append(result)
        # Очистка памяти между тестами
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Таблица результатов
    print(f"\n\n{'='*70}")
    print("📊 СВОДКА РЕЗУЛЬТАТОВ")
    print('='*70)
    print(f"{'Model':<30} {'Load':<8} {'Gen':<8} {'Mem':<8} {'✨ Score'}")
    print('-'*70)

    for r in results:
        load_ok = "✅" if r["load_time"] < 5 else "❌"
        gen_ok = "✅" if r["gen_time"] < 2 else "⚠️ "
        mem_ok = "✅" if r["mem_used"] < 1 else "⚠️ "
        score = f"{load_ok}{gen_ok}{mem_ok}"

        print(f"{r['model']:<30} {r['load_time']:<7.2f}s {r['gen_time']:<7.2f}s {r['mem_used']:<7.2f}GB {score}")

    # Рекомендация
    print(f"\n{'='*70}")
    print("🎯 РЕКОМЕНДАЦИЯ:")

    # Выбираем по критериям
    candidates = [
        r for r in results
        if r["load_time"] < 5 and r["gen_time"] < 2 and r["mem_used"] < 1
    ]

    if candidates:
        best = sorted(candidates, key=lambda x: x["load_time"] + x["gen_time"])[0]
        print(f"✅ Лучшая модель: {best['model']}")
        print(f"   - Язык: {best['lang']}")
        print(f"   - Загрузка: {best['load_time']:.2f}s")
        print(f"   - Генерация: {best['gen_time']:.2f}s")
        print(f"   - Память: {best['mem_used']:.2f}GB")
    else:
        print("❌ Нет моделей, удовлетворяющих всем критериям")
        print("⚠️  Используйте ближайшую по характеристикам")

if __name__ == "__main__":
    main()
