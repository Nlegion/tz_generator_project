"""Тесты для RAG сервиса."""
import pytest
from unittest.mock import MagicMock, patch, Mock
import os

from app.services.rag_service import RAGService, _get_embedding_model, _get_chroma_client
from app.core.config import settings


@pytest.fixture
def mock_embedding_model():
    """Мок модели embeddings."""
    mock = MagicMock()
    mock.encode.return_value = [[0.1, 0.2, 0.3] * 100]  # Mock embedding vector
    return mock


@pytest.fixture
def mock_chroma_collection():
    """Мок коллекции Chroma."""
    mock = MagicMock()
    mock.count.return_value = 0
    mock.query.return_value = {
        'documents': [['chunk1', 'chunk2']],
        'metadatas': [[{'source': 'test'}, {'source': 'test'}]],
        'distances': [[0.1, 0.2]]
    }
    return mock


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    """Мок клиента Chroma."""
    mock = MagicMock()
    mock.get_or_create_collection.return_value = mock_chroma_collection
    return mock


def test_rag_service_init_disabled():
    """Тест инициализации RAG при отключенном RAG."""
    with patch('app.services.rag_service.settings.RAG_ENABLED', False):
        service = RAGService()
        assert service.is_available is False


def test_rag_service_init_with_mocks(mock_embedding_model, mock_chroma_client, mock_chroma_collection):
    """Тест инициализации RAG с моками."""
    with patch('app.services.rag_service._get_embedding_model', return_value=mock_embedding_model):
        with patch('app.services.rag_service._get_chroma_client', return_value=mock_chroma_client):
            with patch('app.services.rag_service._get_chroma_collection', return_value=mock_chroma_collection):
                service = RAGService()
                assert service.is_available is True


def test_rag_service_index_examples_not_available():
    """Тест индексации при недоступном RAG."""
    service = RAGService()
    service.is_available = False
    result = service.index_examples()
    assert result["status"] == "error"


def test_rag_service_retrieve_not_available():
    """Тест поиска при недоступном RAG."""
    service = RAGService()
    service.is_available = False
    chunks = service.retrieve_relevant_chunks("test query")
    assert chunks == []


def test_rag_service_get_stats_not_available():
    """Тест статистики при недоступном RAG."""
    service = RAGService()
    service.is_available = False
    stats = service.get_stats()
    assert stats["status"] == "unavailable"


def test_rag_service_get_stats_available(mock_chroma_collection):
    """Тест статистики при доступном RAG."""
    service = RAGService()
    service.is_available = True
    service.collection = mock_chroma_collection
    
    with patch('app.services.rag_service.settings.RAG_ENABLED', True):
        with patch('app.services.rag_service.settings.RAG_EMBEDDINGS_MODEL', 'test-model'):
            with patch('app.services.rag_service.settings.RAG_LANGUAGE', 'ru'):
                with patch('app.services.rag_service._get_chroma_collection', return_value=mock_chroma_collection):
                    stats = service.get_stats()
                    assert stats["status"] == "available"
                    assert stats["enabled"] is True
                    assert "chunks_count" in stats


def test_rag_service_retrieve_chunks(mock_embedding_model, mock_chroma_collection):
    """Тест поиска релевантных чанков."""
    service = RAGService()
    service.is_available = True
    service.embedding_model = mock_embedding_model
    service.collection = mock_chroma_collection
    
    # Настраиваем мок для encode (возвращает numpy array-like объект)
    import numpy as np
    mock_embedding_array = np.array([[0.1, 0.2, 0.3] * 100])
    mock_embedding_model.encode.return_value = mock_embedding_array
    
    # Мокируем глобальные функции
    with patch('app.services.rag_service._get_embedding_model', return_value=mock_embedding_model):
        with patch('app.services.rag_service._get_chroma_collection', return_value=mock_chroma_collection):
            chunks = service.retrieve_relevant_chunks("test query", top_k=2)
            assert len(chunks) == 2
            assert "text" in chunks[0]
            assert "metadata" in chunks[0]
            assert "relevance_score" in chunks[0]


def test_rag_service_index_examples_empty(mock_chroma_collection):
    """Тест индексации при отсутствии примеров."""
    service = RAGService()
    service.is_available = True
    service.collection = mock_chroma_collection
    
    with patch('app.services.rag_service.load_examples', return_value=[]):
        with patch('app.services.rag_service._get_chroma_collection', return_value=mock_chroma_collection):
            result = service.index_examples()
            assert result["status"] == "error"
            assert "Нет примеров" in result["message"] or "примеров" in result["message"].lower()


def test_get_embedding_model_import_error():
    """Тест обработки ошибки импорта sentence-transformers."""
    with patch('builtins.__import__', side_effect=ImportError("No module")):
        with pytest.raises(ImportError):
            _get_embedding_model()


def test_get_chroma_client_import_error():
    """Тест обработки ошибки импорта chromadb."""
    with patch('builtins.__import__', side_effect=ImportError("No module")):
        with pytest.raises(ImportError):
            _get_chroma_client()
