import pytest
import os
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def mock_generator():
    """Мок для TZGenerator"""
    mock = MagicMock()
    mock.generate.return_value = {
        'text': '# Техническое задание\n\n## 1. Введение\nТестовое ТЗ',
        'metadata': {
            'project_type': 'web',
            'style': 'formal',
            'length_chars': 50,
            'example_count': 0
        }
    }
    return mock


@pytest.fixture
def mock_validator():
    """Мок для TZValidator"""
    mock = MagicMock()
    mock.validate.return_value = {
        'completeness_score': 85,
        'issues': [],
        'suggestions': ['Предложение'],
        'overall_feedback': 'ТЗ валидно'
    }
    return mock


@pytest.fixture
def mock_docx_generator():
    """Мок для DocxGenerator"""
    mock = MagicMock()
    mock.create_document.return_value = '/path/to/file.docx'
    return mock


def test_generate_tz_success(client, mock_generator, mock_docx_generator, temp_output_dir):
    """Тест успешной генерации ТЗ"""
    with patch('app.api.routes.tz_generator', mock_generator):
        with patch('app.api.routes.docx_generator', mock_docx_generator):
            with patch('app.core.config.settings') as mock_settings:
                mock_settings.OUTPUT_DIR = temp_output_dir
                
                response = client.post(
                    '/api/v1/generate',
                    json={
                        'description': 'Веб-приложение',
                        'project_type': 'web',
                        'style': 'formal',
                        'use_examples': False
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert 'task_id' in data
                assert 'text' in data
                assert 'download_url' in data
                assert data['metadata']['project_type'] == 'web'


def test_generate_tz_error(client, mock_generator):
    """Тест ошибки при генерации ТЗ"""
    mock_generator.generate.side_effect = RuntimeError('Ошибка генерации')
    
    with patch('app.api.routes.tz_generator', mock_generator):
        response = client.post(
            '/api/v1/generate',
            json={
                'description': 'Проект',
                'project_type': 'web'
            }
        )
        
        assert response.status_code == 500


def test_validate_tz_text(client, mock_validator):
    """Тест валидации ТЗ по тексту"""
    with patch('app.api.routes.tz_validator', mock_validator):
        response = client.post(
            '/api/v1/validate',
            data={'text': 'Тестовое ТЗ для проверки'}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'is_valid' in data
        assert 'score' in data
        assert data['score'] == 85


def test_validate_tz_file(client, mock_validator, sample_docx_file):
    """Тест валидации ТЗ из .docx файла"""
    with patch('app.api.routes.tz_validator', mock_validator):
        with open(sample_docx_file, 'rb') as f:
            response = client.post(
                '/api/v1/validate',
                files={'file': ('test.docx', f, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert 'is_valid' in data


def test_validate_tz_no_input(client):
    """Тест валидации без входных данных"""
    response = client.post('/api/v1/validate')
        
    assert response.status_code == 400


def test_validate_tz_empty_text(client):
    """Тест валидации пустого текста"""
    response = client.post(
        '/api/v1/validate',
        data={'text': ''}
    )
    
    assert response.status_code == 400


def test_validate_tz_file_too_large(client, temp_dir):
    """Тест валидации слишком большого файла"""
    # Создаем большой файл
    large_file = os.path.join(temp_dir, 'large.txt')
    with open(large_file, 'wb') as f:
        f.write(b'x' * (11 * 1024 * 1024))  # 11 МБ
    
    with open(large_file, 'rb') as f:
        response = client.post(
            '/api/v1/validate',
            files={'file': ('large.txt', f, 'text/plain')}
        )
    
    assert response.status_code == 400


def test_chat_endpoint(client, mock_llm_instance):
    """Тест chat endpoint"""
    # Мокируем модель в routes
    with patch('app.api.routes.model') as mock_model:
        mock_model.generate.return_value = 'Ответ модели'
        mock_model.is_None = False
        
        response = client.post(
            '/api/v1/chat',
            json={
                'message': 'Привет',
                'conversation_id': None,
                'context': None
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'response' in data
        assert 'conversation_id' in data


def test_chat_endpoint_with_context(client, mock_llm_instance):
    """Тест chat endpoint с контекстом"""
    with patch('app.api.routes.model') as mock_model:
        mock_model.generate.return_value = 'Ответ модели'
        mock_model.is_None = False
        
        response = client.post(
            '/api/v1/chat',
            json={
                'message': 'Опиши проект',
                'conversation_id': 'conv123',
                'context': {'description': 'Веб-приложение'}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'response' in data
        assert data['conversation_id'] == 'conv123'


def test_download_tz_success(client):
    """Тест скачивания сгенерированного ТЗ"""
    import uuid
    task_id = str(uuid.uuid4())
    
    # Создаем файл в стандартной директории outputs
    standard_output_dir = './data/outputs'
    os.makedirs(standard_output_dir, exist_ok=True)
    file_path = os.path.join(standard_output_dir, f'{task_id}.docx')
    
    # Создаем тестовый файл ДО запроса
    from docx import Document
    doc = Document()
    doc.add_paragraph('Тестовое ТЗ')
    doc.save(file_path)
    
    try:
        # Убеждаемся, что файл существует
        assert os.path.exists(file_path)
        
        response = client.get(f'/api/v1/download/{task_id}')
        
        assert response.status_code == 200
        assert 'application' in response.headers.get('content-type', '')
    finally:
        # Удаляем тестовый файл
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except OSError:
                pass


def test_download_tz_not_found(client):
    """Тест скачивания несуществующего ТЗ"""
    response = client.get('/api/v1/download/nonexistent-id')
    
    assert response.status_code == 404


def test_get_examples(client, temp_examples_dir, sample_tz_text):
    """Тест получения списка примеров"""
    # Создаем тестовый файл
    file_path = os.path.join(temp_examples_dir, 'example.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_tz_text)
    
    # Мокируем напрямую в routes
    with patch('app.api.routes.os.path.exists') as mock_exists:
        mock_exists.return_value = True
        with patch('app.api.routes.os.listdir') as mock_listdir:
            mock_listdir.return_value = ['example.txt']
            with patch('app.api.routes.os.path.isfile') as mock_isfile:
                mock_isfile.return_value = True
                with patch('app.api.routes.os.path.join') as mock_join:
                    mock_join.side_effect = lambda *args: '/'.join(args)
                    with patch('app.api.routes.os.path.getsize') as mock_getsize:
                        mock_getsize.return_value = 100
                        
                        response = client.get('/api/v1/examples')
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert 'examples' in data
                        assert len(data['examples']) > 0


def test_get_examples_empty(client):
    """Тест получения списка примеров из пустой директории"""
    with patch('app.core.config.settings') as mock_settings:
        mock_settings.EXAMPLES_DIR = '/nonexistent/dir'
        
        response = client.get('/api/v1/examples')
        
        assert response.status_code == 200
        data = response.json()
        assert 'examples' in data
        assert len(data['examples']) == 0
