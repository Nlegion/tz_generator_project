"""Утилиты для определения доступности GPU и CUDA."""
import os
import structlog
from typing import Tuple, Dict, Any, Optional

logger = structlog.get_logger(__name__)


def detect_gpu_availability() -> Tuple[bool, Dict[str, Any]]:
    """
    Определяет доступность GPU/CUDA для использования с llama-cpp-python.

    Returns:
        Tuple[bool, Dict[str, Any]]: (is_available, device_info)
            - is_available: True если GPU доступна
            - device_info: словарь с информацией об устройстве
    """
    device_info: Dict[str, Any] = {
        'cuda_visible': False,
        'cuda_available': False,
        'llama_cpp_cuda': False,
        'gpu_count': 0,
        'error': None
    }

    # Проверка переменной окружения CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible is not None:
        device_info['cuda_visible'] = True
        device_info['cuda_visible_devices'] = cuda_visible
        logger.debug('cuda_visible_devices_set', devices=cuda_visible)

    # Попытка проверить доступность CUDA через llama_cpp
    try:
        from llama_cpp import llama_cpp
        # Проверяем, есть ли CUDA поддержка в скомпилированной версии
        # llama-cpp-python с CUDA обычно имеет атрибуты для работы с GPU
        if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
            # Проверяем, поддерживает ли библиотека GPU offload
            device_info['llama_cpp_cuda'] = True
            logger.debug('llama_cpp_cuda_support_detected')
    except ImportError:
        logger.debug('llama_cpp_import_failed', message='llama_cpp module not available')
        device_info['error'] = 'llama_cpp module not available'
    except Exception as e:
        logger.warning('llama_cpp_cuda_check_error', error=str(e))
        device_info['error'] = str(e)

    # Попытка проверить доступность CUDA через системные вызовы
    try:
        # Проверяем наличие nvidia-smi (если доступен)
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            try:
                gpu_count = int(result.stdout.strip().split('\n')[0])
                device_info['gpu_count'] = gpu_count
                device_info['cuda_available'] = True
                logger.info('nvidia_smi_detected', gpu_count=gpu_count)
            except (ValueError, IndexError):
                logger.debug('nvidia_smi_parse_error', output=result.stdout)
    except FileNotFoundError:
        logger.debug('nvidia_smi_not_found', message='nvidia-smi command not available')
    except subprocess.TimeoutExpired:
        logger.debug('nvidia_smi_timeout', message='nvidia-smi command timeout')
    except Exception as e:
        logger.debug('nvidia_smi_check_error', error=str(e))

    # Определяем итоговую доступность GPU
    # GPU считается доступной, если:
    # 1. Есть CUDA_VISIBLE_DEVICES (даже если пусто, это означает что CUDA может быть доступна)
    # 2. ИЛИ llama_cpp поддерживает CUDA
    # 3. ИЛИ nvidia-smi показывает GPU
    is_available = (
        device_info['cuda_visible'] or
        device_info['llama_cpp_cuda'] or
        device_info['cuda_available']
    )

    if is_available:
        logger.info('gpu_detection_success', device_info=device_info)
    else:
        logger.info('gpu_detection_no_gpu', device_info=device_info)

    return is_available, device_info


def get_optimal_gpu_layers() -> int:
    """
    Определяет оптимальное количество слоев для загрузки на GPU.

    Returns:
        int: Количество слоев (0 если GPU недоступна)
    """
    is_available, device_info = detect_gpu_availability()

    if not is_available:
        return 0

    # Если есть информация о количестве GPU, можно оптимизировать
    # Для начала используем максимальное значение
    # В будущем можно добавить логику определения на основе памяти GPU
    return 40  # Значение по умолчанию из настроек


def get_device_info() -> Dict[str, Any]:
    """
    Возвращает полную информацию об устройстве для логирования.

    Returns:
        Dict[str, Any]: Информация об устройстве
    """
    is_available, device_info = detect_gpu_availability()
    return {
        'gpu_available': is_available,
        **device_info
    }
