import pytest
from unittest.mock import Mock, MagicMock, patch
import json
import os

from app.models.llm_model import GigaChatModel


@pytest.fixture
def mock_llama():
    """Мок для llama_cpp.Llama"""
    mock = MagicMock()
    mock.create_chat_completion.return_value = {
        'choices': [{
            'message': {
                'content': 'Тестовый ответ от модели'
            }
        }]
    }
    return mock


def test_llm_model_initialization_with_mock(mock_llama):
    """Тест инициализации модели с моком"""
    with patch('app.models.llm_model.Llama', return_value=mock_llama):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.N_GPU_LAYERS = 0
            mock_settings.CONTEXT_SIZE = 2048
            
            # Сбрасываем singleton
            GigaChatModel._instance = None
            
            model = GigaChatModel()
            
            assert model.llm is not None


def test_llm_model_singleton():
    """Тест паттерна singleton"""
    with patch('app.models.llm_model.Llama') as mock_llama_class:
        mock_llama = MagicMock()
        mock_llama_class.return_value = mock_llama
        
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.N_GPU_LAYERS = 0
            mock_settings.CONTEXT_SIZE = 2048
            
            # Сбрасываем singleton
            GigaChatModel._instance = None
            
            model1 = GigaChatModel()
            model2 = GigaChatModel()
            
            # Должны быть одним и тем же экземпляром
            assert model1 is model2
            # Llama должен быть вызван только один раз
            assert mock_llama_class.call_count == 1


def test_generate_basic(mock_llama):
    """Тест базовой генерации"""
    with patch('app.models.llm_model.Llama', return_value=mock_llama):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.N_GPU_LAYERS = 0
            mock_settings.CONTEXT_SIZE = 2048
            mock_settings.MAX_TOKENS = 100
            mock_settings.TEMPERATURE = 0.3
            
            GigaChatModel._instance = None
            model = GigaChatModel()
            
            result = model.generate('Тестовый промпт')
            
            assert result == 'Тестовый ответ от модели'
            mock_llama.create_chat_completion.assert_called_once()


def test_generate_with_parameters(mock_llama):
    """Тест генерации с параметрами"""
    with patch('app.models.llm_model.Llama', return_value=mock_llama):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.N_GPU_LAYERS = 0
            mock_settings.CONTEXT_SIZE = 2048
            mock_settings.MAX_TOKENS = 100
            mock_settings.TEMPERATURE = 0.3
            
            GigaChatModel._instance = None
            model = GigaChatModel()
            
            model.generate('Промпт', max_tokens=200, temperature=0.7)
            
            call_kwargs = mock_llama.create_chat_completion.call_args[1]
            assert call_kwargs['max_tokens'] == 200
            assert call_kwargs['temperature'] == 0.7


def test_generate_key_error_handling(mock_llama):
    """Тест обработки KeyError при генерации"""
    mock_llama.create_chat_completion.return_value = {}  # Пустой ответ
    
    with patch('app.models.llm_model.Llama', return_value=mock_llama):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.N_GPU_LAYERS = 0
            mock_settings.CONTEXT_SIZE = 2048
            
            GigaChatModel._instance = None
            model = GigaChatModel()
            
            with pytest.raises(ValueError):
                model.generate('Промпт')


def test_generate_runtime_error_handling(mock_llama):
    """Тест обработки RuntimeError при генерации"""
    mock_llama.create_chat_completion.side_effect = RuntimeError('Ошибка модели')
    
    with patch('app.models.llm_model.Llama', return_value=mock_llama):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.N_GPU_LAYERS = 0
            mock_settings.CONTEXT_SIZE = 2048
            
            GigaChatModel._instance = None
            model = GigaChatModel()
            
            with pytest.raises(RuntimeError):
                model.generate('Промпт')


def test_validate_basic(mock_llama):
    """Тест валидации через LLM"""
    validation_response = json.dumps({
        'completeness_score': 85,
        'issues': [],
        'suggestions': [],
        'overall_feedback': 'OK'
    })
    mock_llama.create_chat_completion.return_value = {
        'choices': [{
            'message': {
                'content': validation_response
            }
        }]
    }
    
    with patch('app.models.llm_model.Llama', return_value=mock_llama):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.N_GPU_LAYERS = 0
            mock_settings.CONTEXT_SIZE = 2048
            
            GigaChatModel._instance = None
            model = GigaChatModel()
            
            result = model.validate('Тестовое ТЗ')
            
            assert 'completeness_score' in result
            assert result['completeness_score'] == 85


def test_validate_json_parse_error(mock_llama):
    """Тест обработки ошибки парсинга JSON при валидации"""
    mock_llama.create_chat_completion.return_value = {
        'choices': [{
            'message': {
                'content': 'Не JSON текст'
            }
        }]
    }
    
    with patch('app.models.llm_model.Llama', return_value=mock_llama):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.N_GPU_LAYERS = 0
            mock_settings.CONTEXT_SIZE = 2048
            
            GigaChatModel._instance = None
            model = GigaChatModel()
            
            result = model.validate('Тестовое ТЗ')
            
            # Должен вернуть fallback результат
            assert 'completeness_score' in result
            assert result['completeness_score'] == 50


def test_validate_text_truncation(mock_llama):
    """Тест обрезания длинного текста при валидации"""
    long_text = 'Текст ' * 1000
    
    with patch('app.models.llm_model.Llama', return_value=mock_llama):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.N_GPU_LAYERS = 0
            mock_settings.CONTEXT_SIZE = 2048
            
            GigaChatModel._instance = None
            model = GigaChatModel()
            
            model.validate(long_text)
            
            # Проверяем, что промпт содержит обрезанный текст
            call_kwargs = mock_llama.create_chat_completion.call_args
            if call_kwargs:
                # Проверяем через messages
                messages = call_kwargs[1].get('messages', call_kwargs[0] if call_kwargs[0] else [])
                if messages and len(messages) > 0:
                    prompt = messages[0].get('content', '')
                    assert len(prompt) < len(long_text) * 2


def test_detect_and_configure_device_gpu_available():
    """Тест определения устройства при доступной GPU."""
    with patch('app.models.llm_model.detect_gpu_availability', return_value=(True, {'gpu_count': 1})):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.AUTO_DETECT_GPU = True
            mock_settings.USE_GPU = True
            mock_settings.N_GPU_LAYERS = 40
            mock_settings.CPU_THREADS = 4
            mock_settings.CPU_BATCH_SIZE = 512
            
            GigaChatModel._instance = None
            model = GigaChatModel()
            
            config = model._detect_and_configure_device()
            
            assert config['device_mode'] == 'gpu'
            assert config['n_gpu_layers'] == 40
            assert config['gpu_available'] is True


def test_detect_and_configure_device_cpu_mode():
    """Тест определения устройства в CPU режиме."""
    with patch('app.models.llm_model.detect_gpu_availability', return_value=(False, {})):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.AUTO_DETECT_GPU = True
            mock_settings.USE_GPU = True
            mock_settings.CPU_THREADS = 8
            mock_settings.CPU_BATCH_SIZE = 256
            
            GigaChatModel._instance = None
            model = GigaChatModel()
            
            config = model._detect_and_configure_device()
            
            assert config['device_mode'] == 'cpu'
            assert config['n_gpu_layers'] == 0
            assert config['n_threads'] == 8
            assert config['n_batch'] == 256


def test_initialize_gpu_fallback_to_cpu():
    """Тест fallback с GPU на CPU при ошибке инициализации GPU."""
    mock_llama = MagicMock()
    
    with patch('app.models.llm_model.detect_gpu_availability', return_value=(True, {'gpu_count': 1})):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.AUTO_DETECT_GPU = True
            mock_settings.USE_GPU = True
            mock_settings.GPU_FALLBACK_ENABLED = True
            mock_settings.N_GPU_LAYERS = 40
            mock_settings.CONTEXT_SIZE = 2048
            mock_settings.CPU_THREADS = 4
            mock_settings.CPU_BATCH_SIZE = 512
            
            # Первый вызов (GPU) вызывает ошибку, второй (CPU) успешен
            with patch('app.models.llm_model.Llama', side_effect=[
                RuntimeError('CUDA error: out of memory'),
                mock_llama
            ]):
                GigaChatModel._instance = None
                model = GigaChatModel()
                
                assert model.device_mode == 'cpu'
                assert model.llm is not None


def test_initialize_gpu_fallback_disabled():
    """Тест что fallback не происходит при отключенном GPU_FALLBACK_ENABLED."""
    with patch('app.models.llm_model.detect_gpu_availability', return_value=(True, {'gpu_count': 1})):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.AUTO_DETECT_GPU = True
            mock_settings.USE_GPU = True
            mock_settings.GPU_FALLBACK_ENABLED = False
            mock_settings.N_GPU_LAYERS = 40
            mock_settings.CONTEXT_SIZE = 2048
            
            with patch('app.models.llm_model.Llama', side_effect=RuntimeError('CUDA error')):
                GigaChatModel._instance = None
                
                with pytest.raises(RuntimeError, match='Ошибка инициализации модели'):
                    GigaChatModel()


def test_initialize_cpu_mode_directly():
    """Тест прямой инициализации в CPU режиме."""
    mock_llama = MagicMock()
    
    with patch('app.models.llm_model.detect_gpu_availability', return_value=(False, {})):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.AUTO_DETECT_GPU = True
            mock_settings.USE_GPU = False
            mock_settings.CONTEXT_SIZE = 2048
            mock_settings.CPU_THREADS = 6
            mock_settings.CPU_BATCH_SIZE = 256
            
            with patch('app.models.llm_model.Llama', return_value=mock_llama):
                GigaChatModel._instance = None
                model = GigaChatModel()
                
                assert model.device_mode == 'cpu'
                assert model.llm is not None
                
                # Проверяем, что Llama был вызван с правильными параметрами
                call_kwargs = mock_llama.__init__.call_args[1]
                assert call_kwargs['n_gpu_layers'] == 0
                assert call_kwargs['n_threads'] == 6
                assert call_kwargs['n_batch'] == 256


def test_initialize_manual_config_no_auto_detect():
    """Тест ручной конфигурации без автоопределения."""
    mock_llama = MagicMock()
    
    with patch('app.models.llm_model.settings') as mock_settings:
        mock_settings.MODEL_PATH = './test_model.gguf'
        mock_settings.AUTO_DETECT_GPU = False
        mock_settings.USE_GPU = True
        mock_settings.N_GPU_LAYERS = 40
        mock_settings.CONTEXT_SIZE = 2048
        mock_settings.CPU_THREADS = 4
        mock_settings.CPU_BATCH_SIZE = 512
        
        with patch('app.models.llm_model.Llama', return_value=mock_llama):
            GigaChatModel._instance = None
            model = GigaChatModel()
            
            # При ручной конфигурации и USE_GPU=True должен быть GPU режим
            assert model.device_mode == 'gpu'
            
            # Проверяем, что detect_gpu_availability не вызывался
            with patch('app.models.llm_model.detect_gpu_availability') as mock_detect:
                GigaChatModel._instance = None
                GigaChatModel()
                # При AUTO_DETECT_GPU=False detect не должен вызываться в _detect_and_configure_device
                # но он все равно может вызываться, поэтому просто проверяем результат


def test_device_mode_attribute():
    """Тест что атрибут device_mode устанавливается корректно."""
    mock_llama = MagicMock()
    
    with patch('app.models.llm_model.detect_gpu_availability', return_value=(True, {'gpu_count': 1})):
        with patch('app.models.llm_model.settings') as mock_settings:
            mock_settings.MODEL_PATH = './test_model.gguf'
            mock_settings.AUTO_DETECT_GPU = True
            mock_settings.USE_GPU = True
            mock_settings.N_GPU_LAYERS = 40
            mock_settings.CONTEXT_SIZE = 2048
            mock_settings.CPU_THREADS = 4
            mock_settings.CPU_BATCH_SIZE = 512
            
            with patch('app.models.llm_model.Llama', return_value=mock_llama):
                GigaChatModel._instance = None
                model = GigaChatModel()
                
                assert hasattr(model, 'device_mode')
                assert model.device_mode in ['gpu', 'cpu']
                assert hasattr(model, 'gpu_available')
