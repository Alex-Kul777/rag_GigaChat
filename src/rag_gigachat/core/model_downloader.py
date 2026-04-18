"""
model_downloader.py - Гибридный режим работы с оффлайн/онлайн переключением для Hugging Face моделей
"""
import logging
import os
from typing import Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def get_hf_cache_dir() -> Path:
    """Получение директории кэша Hugging Face"""
    hf_cache = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_cache)


def is_model_cached(model_name: str) -> bool:
    """
    Проверка наличия модели в кэше Hugging Face

    Args:
        model_name: Название модели (e.g., "gpt2", "sentence-transformers/all-MiniLM-L6-v2")

    Returns:
        True если модель найдена в кэше, False иначе
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        cache_path = try_to_load_from_cache(
            repo_id=model_name,
            filename="config.json"
        )
        is_cached = cache_path is not None
        logger.debug(f"Проверка кэша: {model_name} = {is_cached}")
        return is_cached
    except Exception as e:
        logger.warning(f"Ошибка проверки кэша модели {model_name}: {e}")
        return False


def is_offline_mode_enabled() -> bool:
    """Проверка, включен ли режим оффлайн"""
    return os.getenv("HF_HUB_OFFLINE", "0") == "1"


def set_offline_mode(offline: bool) -> Tuple[bool, bool]:
    """
    Включение/отключение режима оффлайн

    Args:
        offline: True для включения оффлайна, False для отключения

    Returns:
        Кортеж (была_ли_переменная_установлена_раньше, старое_значение)
        Это нужно для восстановления предыдущего состояния
    """
    old_value = os.getenv("HF_HUB_OFFLINE", "0")
    was_set = "HF_HUB_OFFLINE" in os.environ

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.debug("Оффлайн режим включен")
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        logger.debug("Оффлайн режим отключен")

    return was_set, old_value


def check_and_download_model(model_name: str) -> bool:
    """
    Проверить наличие модели в кэше. Если отсутствует → скачать онлайн

    Args:
        model_name: Название модели на Hugging Face

    Returns:
        True если модель успешно загружена (либо была в кэше), False если ошибка
    """
    # Проверяем кэш
    if is_model_cached(model_name):
        logger.info(f"✅ Модель найдена в кэше: {model_name}")
        return True

    logger.info(f"📥 Модель отсутствует в кэше, начинаем скачивание: {model_name}")

    # Сохраняем текущее состояние оффлайн-режима
    was_offline_set, old_offline_value = set_offline_mode(False)

    try:
        # Скачиваем модель
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_name,
            cache_dir=str(get_hf_cache_dir() / "hub"),
            resume_download=True,
            local_files_only=False
        )
        logger.info(f"✅ Модель успешно скачана: {model_name}")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка скачивания модели {model_name}: {e}")
        return False

    finally:
        # Восстанавливаем предыдущее состояние оффлайн-режима
        if was_offline_set:
            set_offline_mode(old_offline_value == "1")
        else:
            set_offline_mode(True)  # По умолчанию включаем оффлайн
        logger.debug("Восстановлено предыдущее состояние оффлайн-режима")


def ensure_model_available(model_name: str) -> bool:
    """
    Убедиться, что модель доступна. Вспомогательная функция.

    Args:
        model_name: Название модели

    Returns:
        True если модель доступна, False иначе
    """
    if is_offline_mode_enabled():
        logger.info(f"Оффлайн режим включен, проверяем кэш: {model_name}")
        if not is_model_cached(model_name):
            logger.error(f"❌ Модель отсутствует в кэше (оффлайн режим): {model_name}")
            return False

    return check_and_download_model(model_name)
