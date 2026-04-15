import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_documents():
    """Тестовые документы из fixtures/sample_docs.json"""
    with open(FIXTURES_DIR / "sample_docs.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_embeddings():
    """
    Мок GigaChatEmbeddings и FAISS — позволяет тестировать load_documents_from_dict
    без реального API и без установки faiss-cpu.
    """
    fake_emb = MagicMock()
    fake_emb.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]] * 10
    fake_emb.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]

    fake_doc = MagicMock()
    fake_doc.page_content = "Нейросети — математические модели."
    fake_doc.metadata = {"source": "doc_1"}

    fake_store = MagicMock()
    fake_store.similarity_search.return_value = [fake_doc]
    fake_store.similarity_search_with_score.return_value = [(fake_doc, 0.9)]

    with patch("vector_store.GigaChatEmbeddings", return_value=fake_emb), \
         patch("vector_store.FAISS") as mock_faiss_cls:
        mock_faiss_cls.from_documents.return_value = fake_store
        mock_faiss_cls.load_local.return_value = fake_store
        yield fake_emb


@pytest.fixture
def mock_gigachat():
    """
    Мок GigaChat LLM — возвращает предопределённый ответ без обращения к API.
    """
    mock_response = MagicMock()
    mock_response.content = "Нейросети — это математические модели, вдохновлённые мозгом."

    with patch("llm_manager.GigaChat") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_cls.return_value = mock_llm
        yield mock_llm
