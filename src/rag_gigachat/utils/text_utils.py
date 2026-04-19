"""Утилиты для анализа и обработки текста"""

import re
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# spaCy и langdetect импорты с fallback
try:
    import spacy
    from langdetect import detect, LangDetectException
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy или langdetect не установлены. pip install spacy langdetect")


def analyze_text_quality(text: str) -> Dict[str, Any]:
    """Анализ качества текста - выявляет артефакты

    Args:
        text: Входной текст

    Returns:
        Словарь с результатами анализа
    """
    if not text:
        return {
            'size': 0,
            'char_count': 0,
            'word_count': 0,
            'line_count': 0,
            'issues': {
                'multiple_spaces': 0,
                'multiple_newlines': 0,
                'tabs': 0,
                'no_break_spaces': 0,
                'broken_words': 0,
            },
            'total_issues': 0,
            'waste_percent': 0.0,
            'samples': {}
        }

    analysis = {
        'size': len(text),
        'char_count': len(text),
        'word_count': len(text.split()),
        'line_count': len(text.split('\n')),
        'issues': {
            'multiple_spaces': len(re.findall(r' {2,}', text)),
            'multiple_newlines': len(re.findall(r'\n{3,}', text)),
            'tabs': text.count('\t'),
            'no_break_spaces': text.count('\u00A0'),
            'broken_words': len(re.findall(r'[а-яёa-z]\n[а-яёa-z]', text, re.IGNORECASE)),
        },
        'samples': {
            'first_100_chars': text[:100],
            'with_multiple_spaces': None,
            'with_broken_word': None,
        }
    }

    # Извлекаем примеры проблем
    match = re.search(r' {2,}[^ ]+', text)
    if match:
        analysis['samples']['with_multiple_spaces'] = match.group(0)

    match = re.search(r'[а-яёa-z]\n[а-яёa-z]', text, re.IGNORECASE)
    if match:
        analysis['samples']['with_broken_word'] = match.group(0)

    # Подсчет процентов потерь
    total_issues = sum(analysis['issues'].values())
    analysis['total_issues'] = total_issues
    analysis['waste_percent'] = round((total_issues / max(len(text), 1)) * 100, 2)

    return analysis


def normalize_text(text: str) -> str:
    """Нормализация текста для улучшения качества эмбеддингов.

    Действия:
    - Удаляет множественные пробелы (2+) → один пробел
    - Удаляет множественные переносы строк (3+) → два переноса (абзац)
    - Убирает табуляции и странные whitespace
    - Чистит разрывы внутри слов (буква+перенос→буква)
    - Удаляет пространство в начале/конце строк

    Args:
        text: Сырой текст из PDF

    Returns:
        Нормализованный текст

    Example:
        >>> normalize_text("Текст    с     пробелами.\\n\\n\\n")
        'Текст с пробелами.'
    """
    if not text:
        return ""

    # 1. Заменяем табуляции на пробелы
    text = text.replace('\t', ' ')

    # 2. Убираем no-break space (U+00A0)
    text = text.replace('\u00A0', ' ')

    # 3. Удаляем множественные пробелы (2+) → один пробел
    text = re.sub(r' {2,}', ' ', text)

    # 4. Нормализуем переносы строк (CRLF → LF)
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 5. Чистим разрывы внутри слов
    # Паттерн: строчная буква (кириллица или латиница) + перевод строки + строчная буква
    text = re.sub(r'([а-яёa-z])\n([а-яёa-z])', r'\1\2', text, flags=re.IGNORECASE)

    # 6. Удаляем множественные переносы строк (3+) → два переноса (абзац)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 7. Убираем пробелы в начале/конце каждой строки
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # 8. Финальный trim
    text = text.strip()

    return text


class SpacySmartSplitter:
    """Разбиение текста на предложения с автоопределением языка (RU/EN).

    Использует spaCy для интеллектуального разбиения на предложения.
    Поддерживает русский и английский языки с автоопределением.

    Singleton паттерн: модели загружаются один раз при первом использовании.

    Attributes:
        nlp_en: spaCy модель для английского языка
        nlp_ru: spaCy модель для русского языка
        _initialized: флаг инициализации
    """

    _instance: Optional["SpacySmartSplitter"] = None

    def __new__(cls) -> "SpacySmartSplitter":
        """Singleton: только один экземпляр класса"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Инициализация spaCy моделей (только если не инициализировано)"""
        if self._initialized:
            return

        self.nlp_en: Optional[Any] = None
        self.nlp_ru: Optional[Any] = None

        if not SPACY_AVAILABLE:
            logger.warning("spaCy не установлена. Разбиение на предложения недоступно.")
            self._initialized = True
            return

        try:
            logger.debug("⏳ Загрузка spaCy моделей для EN и RU...")
            self.nlp_en = spacy.load("en_core_web_sm")
            self.nlp_ru = spacy.load("ru_core_news_sm")
            self._initialized = True
            logger.info("✅ spaCy модели загружены (EN + RU)")
        except OSError as e:
            logger.error(f"❌ Ошибка загрузки spaCy моделей: {e}")
            self._initialized = True

    def detect_language(self, text: str) -> str:
        """Определение языка текста (RU/EN/MIX).

        Args:
            text: Входной текст для определения языка

        Returns:
            'ru' - русский, 'en' - английский (или другой)

        Note:
            Для коротких текстов (<100 символов) используется fallback на EN
        """
        if not SPACY_AVAILABLE or not text or len(text) < 100:
            return 'en'  # Fallback для коротких текстов

        try:
            lang = detect(text)
            return 'ru' if lang == 'ru' else 'en'
        except (LangDetectException, Exception):
            return 'en'  # Fallback при ошибке

    def split_into_sentences(
        self,
        text: str,
        language: Optional[str] = None
    ) -> List[str]:
        """Разбиение текста на предложения с помощью spaCy.

        Args:
            text: Входной текст для разбиения
            language: 'ru', 'en', или None (для автоопределения)

        Returns:
            Список предложений

        Example:
            >>> splitter = SpacySmartSplitter()
            >>> sentences = splitter.split_into_sentences("Mr. Smith works here. He is a doctor.")
            >>> len(sentences)
            2
        """
        if not text or not text.strip():
            return []

        if not SPACY_AVAILABLE or not self.nlp_en or not self.nlp_ru:
            # Fallback: разбиение по точкам, если spaCy недоступна
            logger.debug("spaCy недоступна, используем fallback разбиение")
            return [s.strip() for s in text.split('. ') if s.strip()]

        # Автоопределение языка если не указан
        if language is None:
            language = self.detect_language(text)

        # Выбираем правильную модель
        nlp = self.nlp_ru if language == 'ru' else self.nlp_en

        if nlp is None:
            logger.warning(f"spaCy модель для {language} не загружена")
            return [text]

        # Обрабатываем текст через spaCy
        try:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

            logger.debug(
                f"spaCy разбиение ({language}): {len(sentences)} предложений "
                f"из {len(text)} символов"
            )
            return sentences
        except Exception as e:
            logger.error(f"Ошибка при разбиении spaCy: {e}")
            return [text]


def estimate_token_count(text: str, language: str = 'ru') -> int:
    """Оценка количества токенов в тексте.

    Использует простую эвристику:
    - Русский текст: 1 токен на 4 символа (русский текст более плотный)
    - Английский текст: 1 токен на 3.5 символа

    Args:
        text: Входной текст
        language: Язык ('ru', 'en' или другое)

    Returns:
        Приблизительное количество токенов
    """
    if not text:
        return 0

    # Простая эвристика для оценки токенов
    # GigaChat использует более детальную токенизацию, но это приемлемая оценка
    if language == 'ru':
        chars_per_token = 4.0  # Русский текст более плотный
    else:
        chars_per_token = 3.5  # Английский текст менее плотный

    # Оценка на основе символов
    estimated = max(1, int(len(text) / chars_per_token))
    return estimated


def estimate_language(text: str) -> str:
    """Оценка языка текста для подсчёта токенов.

    Args:
        text: Входной текст

    Returns:
        'ru', 'en' или другой языковой код
    """
    if not SPACY_AVAILABLE or len(text) < 100:
        return 'ru'  # По умолчанию русский

    try:
        from langdetect import detect
        lang = detect(text)
        return lang
    except Exception:
        return 'ru'  # Fallback на русский


def filter_chunks_by_token_count(
    chunks: List[str],
    min_tokens: int = 30,
    max_tokens: Optional[int] = None,
    language: Optional[str] = None
) -> List[str]:
    """Фильтрация чанков по минимальному количеству токенов.

    Удаляет слишком короткие чанки, которые будут низкого качества
    для эмбеддингов. Минимум ~30 токенов соответствует ~1-2 предложениям.

    Args:
        chunks: Список текстовых чанков
        min_tokens: Минимальное количество токенов (по умолчанию 30)
        max_tokens: Максимальное количество токенов (опционально)
        language: Язык для расчёта (ru/en). Если None, определяется автоматически

    Returns:
        Отфильтрованный список чанков
    """
    if not chunks:
        return []

    filtered = []
    removed_count = 0

    for chunk in chunks:
        # Определяем язык если нужно
        if language is None:
            chunk_lang = estimate_language(chunk)
        else:
            chunk_lang = language

        # Считаем токены
        token_count = estimate_token_count(chunk, chunk_lang)

        # Проверяем минимум
        if token_count < min_tokens:
            removed_count += 1
            logger.debug(f"Отфильтрован чанк: {token_count} токенов < {min_tokens} (минимум)")
            continue

        # Проверяем максимум если установлен
        if max_tokens is not None and token_count > max_tokens:
            removed_count += 1
            logger.debug(f"Отфильтрован чанк: {token_count} токенов > {max_tokens} (максимум)")
            continue

        # Чанк прошёл фильтр
        filtered.append(chunk)

    if removed_count > 0:
        logger.info(
            f"Фильтрация чанков: {len(chunks)} → {len(filtered)} чанков "
            f"({removed_count} удалено, min={min_tokens})"
        )

    return filtered


def filter_documents_by_token_count(
    documents: List,
    min_tokens: int = 30,
    max_tokens: Optional[int] = None,
    language: Optional[str] = None
) -> List:
    """Фильтрация LangChain документов по количеству токенов.

    Args:
        documents: Список LangChainDocument объектов
        min_tokens: Минимальное количество токенов
        max_tokens: Максимальное количество токенов
        language: Язык для расчёта

    Returns:
        Отфильтрованный список документов
    """
    if not documents:
        return []

    filtered = []
    removed_count = 0

    for doc in documents:
        # Определяем язык если нужно
        if language is None:
            doc_lang = estimate_language(doc.page_content)
        else:
            doc_lang = language

        # Считаем токены
        token_count = estimate_token_count(doc.page_content, doc_lang)

        # Проверяем минимум
        if token_count < min_tokens:
            removed_count += 1
            logger.debug(f"Отфильтрован документ: {token_count} токенов < {min_tokens}")
            continue

        # Проверяем максимум если установлен
        if max_tokens is not None and token_count > max_tokens:
            removed_count += 1
            logger.debug(f"Отфильтрован документ: {token_count} токенов > {max_tokens}")
            continue

        # Добавляем информацию о токенах в метаданные
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            doc.metadata['token_count'] = token_count
            doc.metadata['language'] = doc_lang

        filtered.append(doc)

    if removed_count > 0:
        logger.info(
            f"Фильтрация документов: {len(documents)} → {len(filtered)} документов "
            f"({removed_count} удалено, min={min_tokens} токенов)"
        )

    return filtered
