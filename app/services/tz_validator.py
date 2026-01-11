from typing import Dict, Any
from app.models.llm_model import model
from app.utils.prompt_templates import VALIDATION_TEMPLATE
import structlog
import json

logger = structlog.get_logger(__name__)


class TZValidator:
    def __init__(self):
        # В тестах model может быть None, будет заменено фикстурой
        self.model = model

    def validate(self, text: str) -> Dict[str, Any]:
        """
        Проверка ТЗ на качество с использованием LLM.

        Args:
            text: Текст ТЗ для проверки

        Returns:
            Словарь с результатами проверки:
            - completeness_score: оценка полноты (0-100)
            - issues: список найденных проблем
            - suggestions: предложения по улучшению
            - overall_feedback: общая обратная связь
        """
        if not text or not text.strip():
            logger.warning('empty_tz_text')
            return {
                'completeness_score': 0,
                'issues': [
                    {
                        'type': 'empty_document',
                        'description': 'Документ пуст',
                        'severity': 'high'
                    }
                ],
                'suggestions': ['Добавьте содержание в техническое задание'],
                'overall_feedback': 'Документ не содержит текста'
            }

        # Ограничиваем длину текста для анализа (первые 3000 символов)
        text_preview = text[:3000] if len(text) > 3000 else text
        
        logger.info('validation_start', text_length=len(text), preview_length=len(text_preview))

        # Формируем промпт для валидации
        validation_prompt = VALIDATION_TEMPLATE.format(tz_text=text_preview)
        
        # Добавляем инструкцию по формату ответа
        validation_prompt += """

Сформируй ответ в формате JSON:
{
    "completeness_score": 0-100,
    "issues": [
        {"type": "missing_section", "description": "описание", "severity": "low/medium/high"},
        {"type": "ambiguity", "description": "описание", "severity": "low/medium/high"},
        {"type": "contradiction", "description": "описание", "severity": "low/medium/high"}
    ],
    "suggestions": ["предложение 1", "предложение 2"],
    "overall_feedback": "общая оценка"
}
"""

        try:
            # Генерируем ответ от модели
            result_text = self.model.generate(validation_prompt, max_tokens=1000)
            
            # Пытаемся извлечь JSON из ответа
            json_text = self._extract_json_from_text(result_text)
            parsed_result = json.loads(json_text)
            
            # Валидация структуры ответа
            validated_result = self._validate_result_structure(parsed_result)
            
            logger.info(
                'validation_complete',
                completeness_score=validated_result.get('completeness_score', 0),
                issues_count=len(validated_result.get('issues', []))
            )
            
            return validated_result

        except json.JSONDecodeError as e:
            msg = f'Ошибка парсинга JSON ответа от модели: {e}'
            logger.warning('validation_json_parse_error', error=str(e), result_preview=result_text[:200])
            return self._create_fallback_result('parsing_error', msg)
        
        except (ValueError, RuntimeError, TimeoutError) as e:
            msg = f'Ошибка валидации ТЗ: {e}'
            logger.error('validation_error', error=str(e), text_length=len(text))
            return self._create_fallback_result('validation_error', msg)

    def _extract_json_from_text(self, text: str) -> str:
        """
        Извлечение JSON из текста ответа модели.

        Args:
            text: Текст ответа модели

        Returns:
            JSON строка
        """
        # Ищем JSON блок между { и }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1]
        
        # Если не нашли, возвращаем весь текст
        return text.strip()

    def _validate_result_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Валидация и нормализация структуры результата.

        Args:
            result: Результат парсинга JSON

        Returns:
            Валидированный результат
        """
        # Убеждаемся, что все обязательные поля присутствуют
        validated = {
            'completeness_score': result.get('completeness_score', 50),
            'issues': result.get('issues', []),
            'suggestions': result.get('suggestions', []),
            'overall_feedback': result.get('overall_feedback', 'Требуется ручная проверка')
        }

        # Нормализуем score в диапазон 0-100
        validated['completeness_score'] = max(0, min(100, int(validated['completeness_score'])))

        # Убеждаемся, что issues - это список словарей
        if not isinstance(validated['issues'], list):
            validated['issues'] = []

        # Убеждаемся, что suggestions - это список строк
        if not isinstance(validated['suggestions'], list):
            validated['suggestions'] = []

        return validated

    def _create_fallback_result(self, error_type: str, error_message: str) -> Dict[str, Any]:
        """
        Создание fallback результата при ошибке.

        Args:
            error_type: Тип ошибки
            error_message: Сообщение об ошибке

        Returns:
            Fallback результат
        """
        return {
            'completeness_score': 50,
            'issues': [
                {
                    'type': error_type,
                    'description': error_message,
                    'severity': 'medium'
                }
            ],
            'suggestions': ['Попробуйте позже или проверьте текст ТЗ'],
            'overall_feedback': 'Требуется ручная проверка'
        }
