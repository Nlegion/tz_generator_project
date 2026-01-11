import pytest
import json
from unittest.mock import patch, MagicMock

from app.services.tz_validator import TZValidator


@pytest.fixture
def validator_with_mock(mock_llm_instance):
    """Валидатор с моком LLM"""
    return TZValidator()


def test_validate_valid_tz(validator_with_mock, sample_tz_text):
    """Тест валидации валидного ТЗ"""
    result = validator_with_mock.validate(sample_tz_text)
    
    assert 'completeness_score' in result
    assert 'issues' in result
    assert 'suggestions' in result
    assert 'overall_feedback' in result
    assert isinstance(result['completeness_score'], int)
    assert 0 <= result['completeness_score'] <= 100


def test_validate_empty_text(validator_with_mock):
    """Тест валидации пустого текста"""
    result = validator_with_mock.validate('')
    
    assert result['completeness_score'] == 0
    assert len(result['issues']) > 0
    assert result['issues'][0]['type'] == 'empty_document'


def test_validate_whitespace_only(validator_with_mock):
    """Тест валидации текста только с пробелами"""
    result = validator_with_mock.validate('   \n\n   ')
    
    assert result['completeness_score'] == 0
    assert len(result['issues']) > 0


def test_validate_json_parsing(validator_with_mock):
    """Тест парсинга JSON ответа от LLM"""
    # Мокаем модель для возврата валидного JSON
    with patch.object(validator_with_mock.model, 'generate') as mock_generate:
        mock_generate.return_value = json.dumps({
            'completeness_score': 90,
            'issues': [],
            'suggestions': ['Отличное ТЗ'],
            'overall_feedback': 'Валидно'
        })
        
        result = validator_with_mock.validate('Тестовое ТЗ')
        
        assert result['completeness_score'] == 90
        assert len(result['issues']) == 0


def test_validate_json_parse_error(validator_with_mock):
    """Тест обработки ошибки парсинга JSON"""
    # Мокаем модель для возврата невалидного JSON
    with patch.object(validator_with_mock.model, 'generate') as mock_generate:
        mock_generate.return_value = 'Не JSON текст'
        
        result = validator_with_mock.validate('Тестовое ТЗ')
        
        # Должен вернуть fallback результат
        assert result['completeness_score'] == 50
        assert len(result['issues']) > 0
        assert any(issue['type'] == 'parsing_error' for issue in result['issues'])


def test_validate_extract_json_from_text(validator_with_mock):
    """Тест извлечения JSON из текста"""
    # Мокаем модель для возврата текста с JSON
    with patch.object(validator_with_mock.model, 'generate') as mock_generate:
        mock_generate.return_value = 'Некоторый текст {"completeness_score": 75, "issues": []} еще текст'
        
        result = validator_with_mock.validate('Тестовое ТЗ')
        
        assert result['completeness_score'] == 75


def test_validate_result_structure_normalization(validator_with_mock):
    """Тест нормализации структуры результата"""
    # Мокаем модель для возврата результата с неполной структурой
    with patch.object(validator_with_mock.model, 'generate') as mock_generate:
        mock_generate.return_value = json.dumps({
            'completeness_score': 150,  # Вне диапазона
            'issues': 'not a list',  # Не список
        })
        
        result = validator_with_mock.validate('Тестовое ТЗ')
        
        # Должен нормализовать
        assert 0 <= result['completeness_score'] <= 100
        assert isinstance(result['issues'], list)
        assert isinstance(result['suggestions'], list)


def test_validate_error_handling(validator_with_mock):
    """Тест обработки ошибок валидации"""
    # Мокаем модель для выброса ошибки
    with patch.object(validator_with_mock.model, 'generate') as mock_generate:
        mock_generate.side_effect = RuntimeError('Ошибка модели')
        
        result = validator_with_mock.validate('Тестовое ТЗ')
        
        # Должен вернуть fallback результат с ошибкой
        assert 'completeness_score' in result
        assert len(result['issues']) > 0
        # Проверяем, что есть информация об ошибке
        assert any('error' in str(issue).lower() or issue.get('type') == 'validation_error' 
                   for issue in result['issues'])


def test_validate_long_text(validator_with_mock):
    """Тест валидации длинного текста (должен обрезаться)"""
    long_text = 'Текст ' * 1000  # Очень длинный текст
    
    with patch.object(validator_with_mock.model, 'generate') as mock_generate:
        mock_generate.return_value = json.dumps({
            'completeness_score': 80,
            'issues': [],
            'suggestions': [],
            'overall_feedback': 'OK'
        })
        
        result = validator_with_mock.validate(long_text)
        
        # Проверяем, что модель получила обрезанный текст
        call_args = mock_generate.call_args[0][0]
        assert len(call_args) < len(long_text)


def test_validate_fallback_result(validator_with_mock):
    """Тест создания fallback результата"""
    result = validator_with_mock._create_fallback_result('test_error', 'Тестовая ошибка')
    
    assert result['completeness_score'] == 50
    assert len(result['issues']) == 1
    assert result['issues'][0]['type'] == 'test_error'
    assert result['issues'][0]['description'] == 'Тестовая ошибка'
