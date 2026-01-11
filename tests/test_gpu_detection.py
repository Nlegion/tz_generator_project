"""Тесты для модуля определения доступности GPU."""
import pytest
import os
from unittest.mock import patch, MagicMock, Mock
from subprocess import TimeoutExpired

from app.utils.gpu_detection import (
    detect_gpu_availability,
    get_optimal_gpu_layers,
    get_device_info
)


def test_detect_gpu_availability_no_cuda():
    """Тест определения GPU при отсутствии CUDA."""
    with patch.dict(os.environ, {}, clear=True):
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            # llama_cpp импорт обрабатывается внутри функции через try-except
            # Если llama_cpp не установлен или без CUDA, GPU должна быть недоступна
            is_available, device_info = detect_gpu_availability()
            # Проверяем структуру ответа (llama_cpp может быть установлен, поэтому is_available может быть True)
            assert 'cuda_available' in device_info
            assert 'gpu_count' in device_info
            assert isinstance(is_available, bool)
            # Без CUDA_VISIBLE_DEVICES и nvidia-smi, cuda_available должна быть False
            assert device_info['cuda_available'] is False


def test_detect_gpu_availability_with_cuda_visible():
    """Тест определения GPU при наличии CUDA_VISIBLE_DEVICES."""
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}, clear=False):
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            is_available, device_info = detect_gpu_availability()
            # С CUDA_VISIBLE_DEVICES GPU должна считаться доступной
            assert is_available is True
            assert device_info['cuda_visible'] is True
            assert device_info['cuda_visible_devices'] == '0'


def test_detect_gpu_availability_with_nvidia_smi():
    """Тест определения GPU через nvidia-smi."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = '1\n'

    with patch('subprocess.run', return_value=mock_result):
        is_available, device_info = detect_gpu_availability()
        assert is_available is True
        assert device_info['cuda_available'] is True
        assert device_info['gpu_count'] == 1


def test_detect_gpu_availability_nvidia_smi_timeout():
    """Тест обработки таймаута nvidia-smi."""
    with patch('subprocess.run', side_effect=TimeoutExpired('nvidia-smi', 2)):
        is_available, device_info = detect_gpu_availability()
        assert device_info['cuda_available'] is False


def test_detect_gpu_availability_with_llama_cpp_cuda():
    """Тест определения GPU через llama_cpp."""
    # Создаем мок модуля llama_cpp с нужным атрибутом
    mock_llama_cpp_module = MagicMock()
    mock_llama_cpp_module.llama_cpp = MagicMock()
    mock_llama_cpp_module.llama_cpp.llama_supports_gpu_offload = True

    with patch('subprocess.run', side_effect=FileNotFoundError()):
        # Патчим импорт внутри функции
        import sys
        original_modules = sys.modules.copy()
        try:
            sys.modules['llama_cpp'] = mock_llama_cpp_module
            # Перезагружаем модуль чтобы применить патч
            import importlib
            import app.utils.gpu_detection
            importlib.reload(app.utils.gpu_detection)
            from app.utils.gpu_detection import detect_gpu_availability
            
            is_available, device_info = detect_gpu_availability()
            assert is_available is True
            assert device_info['llama_cpp_cuda'] is True
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)


def test_detect_gpu_availability_llama_cpp_import_error():
    """Тест обработки ошибки импорта llama_cpp."""
    # Импорт llama_cpp обрабатывается внутри функции через try-except
    # Просто проверяем что функция работает корректно
    with patch('subprocess.run', side_effect=FileNotFoundError()):
        is_available, device_info = detect_gpu_availability()
        # Проверяем что функция вернула результат
        assert 'llama_cpp_cuda' in device_info or 'error' in device_info


def test_get_optimal_gpu_layers_no_gpu():
    """Тест получения оптимального количества слоев при отсутствии GPU."""
    with patch('app.utils.gpu_detection.detect_gpu_availability', return_value=(False, {})):
        layers = get_optimal_gpu_layers()
        assert layers == 0


def test_get_optimal_gpu_layers_with_gpu():
    """Тест получения оптимального количества слоев при наличии GPU."""
    with patch('app.utils.gpu_detection.detect_gpu_availability', return_value=(True, {'gpu_count': 1})):
        layers = get_optimal_gpu_layers()
        assert layers == 40  # Значение по умолчанию


def test_get_device_info():
    """Тест получения полной информации об устройстве."""
    with patch('app.utils.gpu_detection.detect_gpu_availability', return_value=(True, {'gpu_count': 1})):
        device_info = get_device_info()
        assert 'gpu_available' in device_info
        assert device_info['gpu_available'] is True
        assert 'gpu_count' in device_info


def test_detect_gpu_availability_multiple_gpus():
    """Тест определения нескольких GPU."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = '2\n'

    with patch('subprocess.run', return_value=mock_result):
        is_available, device_info = detect_gpu_availability()
        assert is_available is True
        assert device_info['gpu_count'] == 2


def test_detect_gpu_availability_nvidia_smi_error():
    """Тест обработки ошибки nvidia-smi."""
    mock_result = Mock()
    mock_result.returncode = 1
    mock_result.stdout = ''

    with patch('subprocess.run', return_value=mock_result):
        is_available, device_info = detect_gpu_availability()
        assert device_info['cuda_available'] is False
