import pytest
from pydantic import ValidationError

from app.models.tz_schema import (
    TZRequest, TZResponse, ValidationRequest, ValidationResponse,
    ChatRequest, ChatResponse
)


def test_tz_request_valid():
    """Тест валидации валидного TZRequest"""
    request = TZRequest(
        description='Веб-приложение для управления задачами',
        project_type='web',
        style='formal',
        use_examples=True
    )
    
    assert request.description == 'Веб-приложение для управления задачами'
    assert request.project_type == 'web'
    assert request.style == 'formal'
    assert request.use_examples is True


def test_tz_request_defaults():
    """Тест значений по умолчанию для TZRequest"""
    request = TZRequest(description='Проект')
    
    assert request.project_type == 'general'
    assert request.style == 'formal'
    assert request.use_examples is True


def test_tz_request_missing_description():
    """Тест ошибки при отсутствии описания"""
    with pytest.raises(ValidationError):
        TZRequest()


def test_tz_response_valid():
    """Тест валидации валидного TZResponse"""
    response = TZResponse(
        task_id='12345',
        text='Текст ТЗ',
        metadata={'project_type': 'web'},
        download_url='/api/v1/download/12345'
    )
    
    assert response.task_id == '12345'
    assert response.text == 'Текст ТЗ'
    assert response.metadata == {'project_type': 'web'}
    assert response.download_url == '/api/v1/download/12345'


def test_validation_request_text():
    """Тест ValidationRequest с текстом"""
    request = ValidationRequest(text='Текст ТЗ для проверки')
    
    assert request.text == 'Текст ТЗ для проверки'


def test_validation_request_empty():
    """Тест ValidationRequest без текста и файла"""
    request = ValidationRequest()
    
    assert request.text is None


def test_validation_response_valid():
    """Тест валидации валидного ValidationResponse"""
    response = ValidationResponse(
        is_valid=True,
        score=85,
        issues=[],
        suggestions=['Предложение 1'],
        feedback='ТЗ валидно'
    )
    
    assert response.is_valid is True
    assert response.score == 85
    assert len(response.issues) == 0
    assert len(response.suggestions) == 1
    assert response.feedback == 'ТЗ валидно'


def test_validation_response_score_range():
    """Тест валидации диапазона score"""
    # Валидный score
    response = ValidationResponse(
        is_valid=True,
        score=75,
        issues=[],
        suggestions=[],
        feedback='OK'
    )
    assert response.score == 75
    
    # Score вне диапазона должен вызвать ошибку
    with pytest.raises(ValidationError):
        ValidationResponse(
            is_valid=True,
            score=150,  # Вне диапазона 0-100
            issues=[],
            suggestions=[],
            feedback='OK'
        )


def test_chat_request_valid():
    """Тест валидации валидного ChatRequest"""
    request = ChatRequest(
        message='Привет',
        conversation_id='conv123',
        context={'description': 'Проект'}
    )
    
    assert request.message == 'Привет'
    assert request.conversation_id == 'conv123'
    assert request.context == {'description': 'Проект'}


def test_chat_request_defaults():
    """Тест значений по умолчанию для ChatRequest"""
    request = ChatRequest(message='Сообщение')
    
    assert request.conversation_id is None
    assert request.context is None


def test_chat_request_missing_message():
    """Тест ошибки при отсутствии сообщения"""
    with pytest.raises(ValidationError):
        ChatRequest()


def test_chat_response_valid():
    """Тест валидации валидного ChatResponse"""
    response = ChatResponse(
        response='Ответ модели',
        conversation_id='conv123',
        context={'description': 'Проект'}
    )
    
    assert response.response == 'Ответ модели'
    assert response.conversation_id == 'conv123'
    assert response.context == {'description': 'Проект'}


def test_chat_response_required_fields():
    """Тест обязательных полей ChatResponse"""
    with pytest.raises(ValidationError):
        ChatResponse(response='Ответ')  # Отсутствует conversation_id
    
    with pytest.raises(ValidationError):
        ChatResponse(conversation_id='conv123')  # Отсутствует response


def test_validation_response_with_issues():
    """Тест ValidationResponse с проблемами"""
    response = ValidationResponse(
        is_valid=False,
        score=50,
        issues=[
            {
                'type': 'missing_section',
                'description': 'Отсутствует раздел',
                'severity': 'high'
            }
        ],
        suggestions=[],
        feedback='Требуется доработка'
    )
    
    assert response.is_valid is False
    assert len(response.issues) == 1
    assert response.issues[0]['type'] == 'missing_section'
