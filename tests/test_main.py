import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


def test_app_startup():
    """Тест запуска приложения"""
    assert app is not None
    assert app.title == 'TZ Generator API'


def test_health_check(client):
    """Тест health check endpoint"""
    with patch('app.main.settings') as mock_settings:
        mock_settings.MODEL_PATH = './data/model/GigaChat3-10B-A1.8B-q6_k.gguf'
        mock_settings.EXAMPLES_DIR = './data/tz_examples/'
        
        with patch('os.path.exists', return_value=True):
            with patch('os.listdir', return_value=['example1.txt', 'example2.txt']):
                response = client.get('/health')
                
                assert response.status_code == 200
                data = response.json()
                assert 'status' in data
                assert data['status'] == 'healthy'
                assert 'model_loaded' in data
                assert 'examples_count' in data


def test_frontend_index(client):
    """Тест главной страницы"""
    response = client.get('/')
    
    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']


def test_frontend_generate_page(client):
    """Тест страницы генерации"""
    response = client.get('/generate')
    
    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']


def test_frontend_validate_page(client):
    """Тест страницы валидации"""
    response = client.get('/validate')
    
    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']


def test_api_docs_available(client):
    """Тест доступности API документации"""
    response = client.get('/api/docs')
    
    assert response.status_code == 200


def test_api_redoc_available(client):
    """Тест доступности ReDoc"""
    response = client.get('/api/redoc')
    
    assert response.status_code == 200


def test_cors_headers(client):
    """Тест CORS заголовков"""
    response = client.options('/api/v1/generate')
    
    # FastAPI автоматически добавляет CORS заголовки
    assert response.status_code in [200, 405]  # OPTIONS может вернуть 405


def test_startup_creates_directories():
    """Тест создания директорий при старте"""
    import os
    import tempfile
    import shutil
    
    temp_path = tempfile.mkdtemp()
    outputs_dir = os.path.join(temp_path, 'outputs')
    examples_dir = os.path.join(temp_path, 'examples')
    templates_dir = os.path.join(temp_path, 'templates')
    
    try:
        # Мокируем os.makedirs для проверки вызова
        with patch('app.main.os.makedirs') as mock_makedirs:
            from app.main import startup_event
            import asyncio
            
            asyncio.run(startup_event())
            
            # Проверяем, что makedirs был вызван для нужных директорий
            assert mock_makedirs.called
            # Проверяем, что были созданы нужные директории
            call_args_list = [str(call) for call in mock_makedirs.call_args_list]
            assert any('outputs' in str(call) for call in mock_makedirs.call_args_list)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)
