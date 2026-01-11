from typing import Optional, Dict, Any
from llama_cpp import Llama
import structlog
import json
import os

from app.core.config import settings
from app.utils.gpu_detection import detect_gpu_availability, get_device_info

logger = structlog.get_logger(__name__)


class GigaChatModel:
    _instance: Optional['GigaChatModel'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _detect_and_configure_device(self) -> Dict[str, Any]:
        """
        Определяет доступность GPU и конфигурирует параметры инициализации.

        Returns:
            Dict[str, Any]: Конфигурация для инициализации модели
        """
        device_config = {
            'n_gpu_layers': 0,
            'n_threads': settings.CPU_THREADS,
            'n_batch': settings.CPU_BATCH_SIZE,
            'device_mode': 'cpu',
            'gpu_available': False
        }

        # Автоматическое определение GPU, если включено
        if settings.AUTO_DETECT_GPU:
            gpu_available, device_info = detect_gpu_availability()
            device_config['gpu_available'] = gpu_available
            device_config['device_info'] = device_info

            if gpu_available and settings.USE_GPU:
                # Пытаемся использовать GPU
                device_config['n_gpu_layers'] = settings.N_GPU_LAYERS
                device_config['device_mode'] = 'gpu'
                device_config['n_threads'] = 4  # Для GPU можно использовать меньше потоков
                device_config['n_batch'] = 512
                logger.info(
                    'gpu_config_selected',
                    n_gpu_layers=settings.N_GPU_LAYERS,
                    device_info=device_info
                )
            else:
                logger.info(
                    'cpu_config_selected',
                    reason='GPU not available or USE_GPU=False' if not gpu_available else 'USE_GPU=False',
                    n_threads=settings.CPU_THREADS
                )
        else:
            # Ручная конфигурация без автоопределения
            if settings.USE_GPU:
                device_config['n_gpu_layers'] = settings.N_GPU_LAYERS
                device_config['device_mode'] = 'gpu'
            logger.info('manual_config_selected', device_mode=device_config['device_mode'])

        return device_config

    def _initialize(self):
        """Инициализация модели с поддержкой GPU и fallback на CPU"""
        # Определяем конфигурацию устройства
        device_config = self._detect_and_configure_device()
        self.device_mode = device_config['device_mode']
        self.gpu_available = device_config.get('gpu_available', False)

        # Попытка инициализации с GPU (если настроено)
        if device_config['device_mode'] == 'gpu' and device_config['n_gpu_layers'] > 0:
            try:
                logger.info(
                    'model_loading_gpu',
                    model_path=settings.MODEL_PATH,
                    n_gpu_layers=device_config['n_gpu_layers'],
                    context_size=settings.CONTEXT_SIZE
                )

                self.llm = Llama(
                    model_path=settings.MODEL_PATH,
                    n_gpu_layers=device_config['n_gpu_layers'],
                    n_ctx=settings.CONTEXT_SIZE,
                    n_batch=device_config['n_batch'],
                    n_threads=device_config['n_threads'],
                    verbose=False,
                    seed=42,
                    use_mlock=False,
                )

                logger.info(
                    'model_loaded_successfully_gpu',
                    model_path=settings.MODEL_PATH,
                    device_mode=self.device_mode
                )
                return

            except RuntimeError as e:
                error_msg = str(e)
                # Проверяем, связана ли ошибка с GPU
                gpu_related_errors = ['cuda', 'gpu', 'cublas', 'cuda error']
                is_gpu_error = any(keyword in error_msg.lower() for keyword in gpu_related_errors)

                if is_gpu_error and settings.GPU_FALLBACK_ENABLED:
                    logger.warning(
                        'gpu_init_failed_fallback',
                        error=error_msg,
                        falling_back_to='cpu'
                    )
                    # Продолжаем к CPU инициализации
                else:
                    msg = f'Ошибка инициализации модели: {e}'
                    logger.error('model_init_error', error=error_msg, model_path=settings.MODEL_PATH)
                    raise RuntimeError(msg) from e

        # Инициализация CPU режима (fallback или основной режим)
        try:
            logger.info(
                'model_loading_cpu',
                model_path=settings.MODEL_PATH,
                n_threads=device_config['n_threads'],
                context_size=settings.CONTEXT_SIZE
            )

            self.llm = Llama(
                model_path=settings.MODEL_PATH,
                n_gpu_layers=0,  # Явно указываем CPU режим
                n_ctx=settings.CONTEXT_SIZE,
                n_batch=device_config['n_batch'],
                n_threads=device_config['n_threads'],
                verbose=False,
                seed=42,
                use_mlock=False,
            )

            self.device_mode = 'cpu'
            logger.info(
                'model_loaded_successfully_cpu',
                model_path=settings.MODEL_PATH,
                device_mode=self.device_mode,
                n_threads=device_config['n_threads']
            )

        except (FileNotFoundError, OSError) as e:
            msg = f'Модель не найдена или ошибка доступа: {e}'
            logger.error('model_load_error', error=str(e), model_path=settings.MODEL_PATH)
            raise FileNotFoundError(msg) from e
        except RuntimeError as e:
            msg = f'Ошибка инициализации модели: {e}'
            logger.error('model_init_error', error=str(e), model_path=settings.MODEL_PATH)
            raise RuntimeError(msg) from e

    def generate(self, prompt: str, **kwargs) -> str:
        """Генерация текста по промпту"""
        max_tokens = kwargs.get('max_tokens', settings.MAX_TOKENS)
        temperature = kwargs.get('temperature', settings.TEMPERATURE)
        
        try:
            logger.debug(
                'llm_generation_start',
                prompt_length=len(prompt),
                max_tokens=max_tokens,
                temperature=temperature
            )

            response = self.llm.create_chat_completion(
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get('top_p', 0.9),
                repeat_penalty=kwargs.get('repeat_penalty', 1.1),
                stream=False,
            )

            result = response['choices'][0]['message']['content']
            logger.debug('llm_generation_complete', result_length=len(result))
            return result

        except (KeyError, IndexError) as e:
            msg = f'Неверный формат ответа от модели: {e}'
            logger.error('llm_response_format_error', error=str(e))
            raise ValueError(msg) from e
        except RuntimeError as e:
            msg = f'Ошибка выполнения модели: {e}'
            logger.error('llm_runtime_error', error=str(e), prompt_length=len(prompt))
            raise RuntimeError(msg) from e
        except TimeoutError as e:
            msg = f'Таймаут генерации: {e}'
            logger.error('llm_timeout_error', error=str(e))
            raise TimeoutError(msg) from e

    def validate(self, text: str) -> Dict[str, Any]:
        """Проверка ТЗ на качество"""
        validation_prompt = f"""
        Анализируй техническое задание ниже. Сформируй ответ в формате JSON:
        {{
            "completeness_score": 0-100,
            "issues": [
                {{"type": "missing_section", "description": "описание", "severity": "low/medium/high"}},
                {{"type": "ambiguity", "description": "описание", "severity": "low/medium/high"}},
                {{"type": "contradiction", "description": "описание", "severity": "low/medium/high"}}
            ],
            "suggestions": ["предложение 1", "предложение 2"],
            "overall_feedback": "общая оценка"
        }}

        Текст ТЗ для анализа:
        {text[:3000]}  # Ограничиваем длину для анализа
        """

        try:
            result = self.generate(validation_prompt, max_tokens=1000)
            # Парсинг JSON из ответа модели
            parsed_result = json.loads(result)
            logger.debug('validation_result_parsed', result_keys=list(parsed_result.keys()))
            return parsed_result
        except json.JSONDecodeError as e:
            msg = f'Ошибка парсинга JSON ответа от модели: {e}'
            logger.warning('validation_json_parse_error', error=str(e), text_preview=text[:200])
            # Fallback если не удалось распарсить JSON
            return {
                'completeness_score': 50,
                'issues': [
                    {'type': 'parsing_error', 'description': 'Не удалось проанализировать ТЗ', 'severity': 'medium'}
                ],
                'suggestions': ['Проверьте форматирование документа'],
                'overall_feedback': 'Требуется ручная проверка'
            }
        except (ValueError, RuntimeError, TimeoutError) as e:
            msg = f'Ошибка валидации ТЗ: {e}'
            logger.error('validation_error', error=str(e), text_length=len(text))
            return {
                'completeness_score': 0,
                'issues': [
                    {'type': 'validation_error', 'description': msg, 'severity': 'high'}
                ],
                'suggestions': ['Попробуйте позже или проверьте текст ТЗ'],
                'overall_feedback': 'Ошибка при анализе ТЗ'
            }


# Глобальный экземпляр модели
# В тестах будет заменен через фикстуру
try:
    if not os.environ.get('PYTEST_CURRENT_TEST'):
        model = GigaChatModel()
    else:
        # В тестах создаем заглушку, которая будет заменена фикстурой
        model = None
except (FileNotFoundError, ValueError, RuntimeError):
    # Если модель не найдена (например, в тестах), создаем None
    model = None