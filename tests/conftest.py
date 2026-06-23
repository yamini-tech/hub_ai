import sys
import os
import json
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from typing import Any
from collections import defaultdict

os.environ.setdefault("SMARTHUB_API_KEY", "")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("SMARTHUB_ALLOWED_EXTRACT_DIR", "/")

sys.modules.pop("sentence_transformers", None)
sys.modules.pop("chromadb", None)

st_mock = MagicMock()
st_encode = MagicMock()
st_encode.tolist.return_value = [0.1] * 384
st_mock.SentenceTransformer.return_value.encode.return_value = st_encode
sys.modules["sentence_transformers"] = st_mock

chroma_collection = MagicMock()
chroma_collection.add.return_value = None
chroma_collection.query.return_value = {"documents": [["mock result"]]}
chroma_collection.get.return_value = {"documents": ["mock doc"]}
chroma_client = MagicMock()
chroma_client.get_or_create_collection.return_value = chroma_collection
chromadb_mock = MagicMock()
chromadb_mock.PersistentClient.return_value = chroma_client
sys.modules["chromadb"] = chromadb_mock


@pytest.fixture(autouse=True)
def clear_throttling():
    from app.services.throttling import _rate_windows
    _rate_windows.clear()


@pytest.fixture(autouse=True)
def clear_chat_histories():
    from app.routers.chat import chat_histories
    chat_histories.clear()


@pytest.fixture(autouse=True)
def clear_job_store():
    from app.services.job_manager import _memory_store
    _memory_store.clear()


@pytest.fixture(autouse=True)
def clear_llm_cache():
    from app.services.cache import clear_cache
    clear_cache()


@pytest.fixture
def mock_tiktoken():
    with patch("app.services.throttling.tiktoken") as mock_tk:
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 10
        mock_tk.encoding_for_model.return_value = mock_enc
        yield mock_tk


@pytest.fixture
def mock_litellm_acompletion():
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
        mock_message = MagicMock()
        mock_message.content = "test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_delta = MagicMock()
        mock_delta.content = "test chunk"
        mock_choice.delta = mock_delta
        mock.return_value = MagicMock(choices=[mock_choice])
        yield mock


@pytest.fixture
def mock_litellm_completion():
    with patch("litellm.completion") as mock:
        mock_message = MagicMock()
        mock_message.content = "test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock.return_value = MagicMock(choices=[mock_choice])
        yield mock


@pytest.fixture
def mock_chroma_collection():
    return chroma_collection


@pytest.fixture
def app():
    from app.main import app as _app
    return _app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
