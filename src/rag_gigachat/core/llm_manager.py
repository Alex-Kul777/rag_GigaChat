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

from rag_gigachat.config import model_config, gigachat_config
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

        logger.info(f"LLMManager инициализирован. Модель: {model_name}")

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
        Загрузка локальной модели через HuggingFace

        Returns:
            LangChain LLM объект
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

        logger.info(f"Загрузка локальной модели: {self.model_name}")
        print(f"🔍 DEBUG: Загрузка модели {self.model_name}")

        # Гибридный режим: проверяем/скачиваем модель если нужно
        if not check_and_download_model(self.model_name):
            raise RuntimeError(
                f"Не удалось загрузить модель {self.model_name}. "
                f"Проверьте интернет-соединение или скачайте модель вручную."
            )

        try:
            torch_dtype = torch.float16 if model_config.device == "cuda" else torch.float32
            print(f"🔍 DEBUG: Используем dtype: {torch_dtype}, device: {model_config.device}")

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                dtype=torch_dtype,
                device_map="auto" if model_config.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("🔍 DEBUG: Модель загружена")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            print("🔍 DEBUG: Токенизатор загружен")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print("🔍 DEBUG: Создаем pipeline...")
            text_gen_pipeline = hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=model_config.max_new_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                do_sample=True,
                device=0 if model_config.device == "cuda" else -1
            )
            print("🔍 DEBUG: Pipeline создан")

            self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
            self.is_initialized = True

            print("✅ DEBUG: Модель успешно загружена")
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

        for attempt in range(max_retries):
            try:
                attempt_start = time.time()
                logger.debug(f"⏱️ LLM invoke attempt {attempt + 1}/{max_retries}, timeout={timeout}s")
                response = llm.invoke(prompt)
                elapsed = time.time() - attempt_start
                logger.info(f"✅ LLM ответ получен за {elapsed:.1f} сек")
                return response
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
