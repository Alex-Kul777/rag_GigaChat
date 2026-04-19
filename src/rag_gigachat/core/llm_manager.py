"""
llm_manager.py - Менеджер языковых моделей
"""
import logging
import time
import random
from typing import Optional

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models import BaseLLM

try:
    from langchain_gigachat.chat_models import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False
    torch = None  # Placeholder если torch недоступен

from rag_gigachat.config import model_config, gigachat_config, debug_config
from rag_gigachat.core.model_downloader import check_and_download_model

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Менеджер языковых моделей.
    Поддерживает локальные модели (HuggingFace), GigaChat и OpenAI.
    """

    def __init__(self, model_name: str = None, model_type: str = "local"):
        """
        Инициализация менеджера LLM

        Args:
            model_name: Название модели
            model_type: Тип модели ("local", "openai", "gigachat")
        """
        self.model_type = model_type
        self.model_name = model_name or model_config.llm_model_name
        self.llm = None
        self.is_initialized = False
        self.is_offline = model_type == "local"

        # 🐛 DEBUG MODE: Логируем статус debug-режима
        import os
        # Проверяем ТЕКУЩЕЕ значение переменной окружения (не только при инициализации config)
        env_debug_mode = os.getenv("RAG_DEBUG_MODE", "false").lower() == "true"

        if model_type == "local":
            # Используем env_debug_mode, т.к. это актуальное значение в текущем процессе
            if env_debug_mode:
                # ✅ DEBUG режим ВКЛЮЧЕН
                original_model = self.model_name
                self.model_name = debug_config.debug_model_name
                logger.info(f"🐛 DEBUG MODE ENABLED: Using fast model {self.model_name} (125M) instead of {original_model} (500M)")
                logger.info(f"⏱️  Expected: Load ~2-3sec, Generate ~1-2sec, Memory ~400MB")
                print(f"🐛 DEBUG MODE: {self.model_name} (fast, 125M params)")
            else:
                # ❌ DEBUG режим ОТКЛЮЧЕН
                logger.info(f"📦 PRODUCTION MODE: Using {self.model_name} (500M, high quality)")
                logger.info(f"💡 To use debug mode with fast model: export RAG_DEBUG_MODE=true")
                print(f"📦 Production mode: {self.model_name}")

        offline_status = "OFFLINE" if self.is_offline else "ONLINE"
        logger.info(f"LLMManager: model_type={model_type}, model={self.model_name}, mode={offline_status}")
        print(f"📦 LLMManager: {offline_status} mode, model_type={model_type}")

    def load_gigachat_model(self) -> BaseLLM:
        """
        Загрузка GigaChat модели

        Returns:
            LangChain LLM объект
        """
        if not GIGACHAT_AVAILABLE:
            raise ImportError(
                "langchain-gigachat не установлен. Установите: pip install langchain-gigachat"
            )

        if not gigachat_config.api_key:
            raise ValueError("GigaChat API ключ не настроен в конфигурации")

        logger.info(f"Загрузка GigaChat модели: {gigachat_config.model}")

        try:
            self.llm = GigaChat(
                credentials=gigachat_config.api_key,
                verify_ssl_certs=gigachat_config.verify_ssl_certs,
                scope=gigachat_config.scope,
                model=gigachat_config.model,
                timeout=gigachat_config.timeout,
                max_retries=gigachat_config.max_retries
            )
            self.is_initialized = True
            logger.info("GigaChat модель загружена")
            return self.llm

        except Exception as e:
            logger.error(f"Ошибка загрузки GigaChat модели: {e}")
            raise

    def load_local_model(self) -> BaseLLM:
        """
        Загрузка локальной модели через HuggingFace (text-generation task)
        """
        if not _torch_available:
            raise RuntimeError(
                "PyTorch не установлен. Установите: pip install torch torchvision torchaudio"
            )

        import torch  # Явный импорт в функцию
        from transformers import AutoTokenizer, pipeline as hf_pipeline

        logger.info(f"Загрузка локальной модели: {self.model_name}")
        print(f"🔍 DEBUG: Загрузка модели {self.model_name}")

        # Гибридный режим: проверяем/скачиваем модель если нужно
        if not check_and_download_model(self.model_name):
            raise RuntimeError(
                f"Не удалось загрузить модель {self.model_name}. "
                f"Проверьте интернет-соединение или скачайте модель вручную."
            )

        try:
            if not _torch_available or torch is None:
                raise RuntimeError("PyTorch недоступен")

            # 🐛 DEBUG MODE: Использовать fp16 для экономии GPU памяти при debug моделях
            import os
            is_debug_mode = os.getenv("RAG_DEBUG_MODE", "false").lower() == "true"
            torch_dtype = torch.float16 if is_debug_mode else torch.float32
            print(f"🔍 DEBUG: dtype: {torch_dtype} ({('fp16 - быстро!' if is_debug_mode else 'fp32 - качество')}), device: {model_config.device}")

            print("🔍 DEBUG: Создаем pipeline...")
            # Использовать device=-1 (CPU) для экономии памяти, или cuda если есть место
            use_device = 0 if torch.cuda.is_available() else -1  # 0=GPU, -1=CPU
            print(f"🔍 DEBUG: Используем device: {'GPU (cuda:0)' if use_device == 0 else 'CPU'}")

            text_gen_pipeline = hf_pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch_dtype,
                device=use_device,
                device_map="auto" if use_device == 0 else None,  # Auto split между GPU/CPU если нужно
                max_new_tokens=model_config.max_new_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            print("🔍 DEBUG: Pipeline создан")

            self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
            self.is_initialized = True

            # Диагностика успешной загрузки
            print(f"✅ DEBUG: Модель успешно загружена (text-generation)")
            if _torch_available and torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1e9
                print(f"🔍 GPU память используется: {gpu_mem:.2f} GB")
            return self.llm

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            print(f"❌ DEBUG: Ошибка загрузки: {e}")
            import traceback
            traceback.print_exc()
            raise

    def load_openai_model(self, api_key: str, model_name: str = "gpt-3.5-turbo") -> BaseLLM:
        """
        Загрузка OpenAI модели

        Args:
            api_key: API ключ OpenAI
            model_name: Название модели

        Returns:
            LangChain LLM объект
        """
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=model_config.temperature,
            max_tokens=model_config.max_new_tokens
        )
        self.is_initialized = True
        logger.info(f"OpenAI модель загружена: {model_name}")
        return self.llm

    def get_llm(self) -> BaseLLM:
        """Получение LLM (загружает если нужно)"""
        if not self.is_initialized:
            if self.model_type == "gigachat":
                self.load_gigachat_model()
            elif self.model_type == "openai":
                self.load_openai_model(model_config.openai_api_key)
            else:
                self.load_local_model()
        return self.llm

    def invoke_with_retry(self, prompt, max_retries: int = 3, timeout: float = 60.0):
        """
        Invoke LLM with exponential backoff retry logic.
        Handles timeouts and transient errors gracefully.

        Args:
            prompt: Input prompt to the LLM
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Timeout per attempt in seconds (default: 60.0)

        Returns:
            LLM response

        Raises:
            TimeoutError: If call exceeds timeout on all attempts
            Exception: If all retries fail
        """
        llm = self.get_llm()
        last_error = None
        start = time.time()
        cuda_error_encountered = False

        for attempt in range(max_retries):
            try:
                attempt_start = time.time()
                logger.debug(f"⏱️ LLM invoke attempt {attempt + 1}/{max_retries}, timeout={timeout}s")

                # Диагностика перед CUDA операциями
                if _torch_available:
                    import torch
                    if torch.cuda.is_available():
                        # Логируем статус GPU
                        gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1e9
                        gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9
                        logger.debug(f"🔍 GPU память: allocated={gpu_mem_alloc:.2f}GB, reserved={gpu_mem_reserved:.2f}GB")

                        # Очищаем GPU кэш
                        torch.cuda.empty_cache()
                        logger.debug(f"🔍 GPU кэш очищен")

                response = llm.invoke(prompt)
                elapsed = time.time() - attempt_start
                logger.info(f"✅ LLM ответ получен за {elapsed:.1f} сек")
                return response

            except RuntimeError as e:
                elapsed = time.time() - attempt_start
                error_msg = str(e)

                # Специальная обработка CUDA ошибок
                if "cuda" in error_msg.lower() or "device-side assert" in error_msg.lower():
                    cuda_error_encountered = True
                    logger.error(f"❌ CUDA ошибка: {error_msg}")

                    # Fallback на CPU
                    if self.model_type == "local" and _torch_available:
                        logger.warning(f"⚠️  Переключение локальной модели на CPU")
                        try:
                            import torch
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()

                            # Переместим модель на CPU
                            if hasattr(self.llm, 'model'):
                                self.llm.model.to('cpu')
                                logger.info("✓ Модель перемещена на CPU")

                            # Повторим попытку на CPU
                            response = llm.invoke(prompt)
                            logger.info(f"✅ Ответ получен на CPU за {elapsed:.1f} сек")
                            return response
                        except Exception as cpu_error:
                            logger.error(f"❌ Ошибка даже на CPU: {cpu_error}")
                            last_error = e
                    else:
                        last_error = e
                else:
                    last_error = e

                if attempt < max_retries - 1 and not cuda_error_encountered:
                    wait_time = (2 ** attempt) * 1.0 + random.uniform(0, 1)
                    logger.warning(
                        f"❌ LLM call failed after {elapsed:.1f}s (attempt {attempt + 1}): {type(e).__name__}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                elif cuda_error_encountered:
                    logger.error(f"❌ CUDA ошибка не решена на CPU")
                    raise
                else:
                    total_elapsed = time.time() - start
                    logger.error(
                        f"❌ LLM call failed after {max_retries} attempts ({total_elapsed:.1f}s total)"
                    )
                    raise

            except (TimeoutError, Exception) as e:
                elapsed = time.time() - attempt_start
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0 + random.uniform(0, 1)
                    logger.warning(
                        f"❌ LLM call failed after {elapsed:.1f}s (attempt {attempt + 1}): {type(e).__name__}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    total_elapsed = time.time() - start
                    logger.error(
                        f"❌ LLM call failed after {max_retries} attempts ({total_elapsed:.1f}s total)"
                    )
                    raise

        raise last_error
