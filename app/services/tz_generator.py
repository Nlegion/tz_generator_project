from typing import Dict, Any, Optional
import structlog

from app.models.llm_model import model
from app.utils.file_handlers import load_examples, get_relevant_examples
from app.utils.prompt_templates import GENERATION_TEMPLATES
from app.core.config import settings

logger = structlog.get_logger(__name__)

# Глобальный экземпляр RAGService (инициализируется в main.py)
rag_service: Optional['RAGService'] = None


class TZGenerator:
    def __init__(self):
        try:
            self.examples = load_examples(settings.EXAMPLES_DIR)
            logger.info('tz_generator_initialized', examples_count=len(self.examples))
        except OSError as e:
            logger.warning('examples_load_failed', error=str(e), examples_dir=settings.EXAMPLES_DIR)
            self.examples = []

    def generate(self, project_description: str, project_type: str = "general",
                 style: str = "formal", use_examples: bool = True) -> Dict[str, Any]:
        """
        Генерация ТЗ на основе описания проекта

        Args:
            project_description: Описание проекта
            project_type: Тип проекта (web, mobile, desktop, etc.)
            style: Стиль документа (formal, concise, detailed)
            use_examples: Использовать ли примеры для контекста

        Returns:
            Dict с сгенерированным текстом и метаданными
        """
        # Выбор шаблона промпта
        template = GENERATION_TEMPLATES.get(
            project_type,
            GENERATION_TEMPLATES["general"]
        )

        # Подготовка контекста
        context_examples = ""
        relevant_examples = []
        
        if use_examples:
            # Пробуем использовать RAG, если доступен
            if settings.RAG_ENABLED and rag_service is not None and rag_service.is_available:
                try:
                    chunks = rag_service.retrieve_relevant_chunks(
                        query=project_description,
                        top_k=settings.RAG_TOP_K
                    )
                    
                    if chunks:
                        # Объединяем чанки с ограничением по длине
                        context_parts = []
                        total_length = 0
                        
                        for chunk in chunks:
                            chunk_text = chunk['text']
                            if total_length + len(chunk_text) <= settings.RAG_MAX_CONTEXT_LENGTH:
                                context_parts.append(chunk_text)
                                total_length += len(chunk_text)
                            else:
                                # Добавляем частично, если есть место
                                remaining = settings.RAG_MAX_CONTEXT_LENGTH - total_length
                                if remaining > 100:  # Минимум 100 символов
                                    context_parts.append(chunk_text[:remaining])
                                break
                        
                        if context_parts:
                            context_examples = "\n\n--- Релевантные фрагменты ТЗ ---\n" + "\n\n---\n\n".join(context_parts)
                            relevant_examples = [chunk['text'] for chunk in chunks[:len(context_parts)]]
                            logger.debug(
                                'rag_context_used',
                                chunks_count=len(context_parts),
                                total_length=total_length
                            )
                except Exception as e:
                    logger.warning('rag_retrieval_failed', error=str(e), falling_back='old_method')
                    # Fallback на старый метод
                    if self.examples:
                        relevant_examples = get_relevant_examples(
                            project_description,
                            self.examples,
                            n_examples=2
                        )
                        context_examples = "\n\n--- Примеры ТЗ ---\n" + "\n---\n".join(relevant_examples)
            else:
                # Используем старый метод (RAG недоступен или отключен)
                if self.examples:
                    relevant_examples = get_relevant_examples(
                        project_description,
                        self.examples,
                        n_examples=2
                    )
                    context_examples = "\n\n--- Примеры ТЗ ---\n" + "\n---\n".join(relevant_examples)

        # Формирование промпта
        prompt = template.format(
            description=project_description,
            examples=context_examples,
            style=style
        )

        logger.info(
            'generation_start',
            project_type=project_type,
            style=style,
            use_examples=use_examples,
            description_length=len(project_description)
        )

        # Генерация
        try:
            # В тестах model может быть None, используем фикстуру
            from app.models.llm_model import model as llm_model
            if llm_model is None:
                # Fallback для тестов - будет заменено фикстурой
                raise RuntimeError('Model not initialized')
            generated_text = llm_model.generate(prompt)
        except (ValueError, RuntimeError, TimeoutError) as e:
            msg = f'Ошибка генерации текста: {e}'
            logger.error('generation_error', error=str(e), project_type=project_type)
            raise RuntimeError(msg) from e

        # Форматирование результата
        formatted_text = self._format_output(generated_text)

        logger.info(
            'generation_complete',
            project_type=project_type,
            text_length=len(formatted_text)
        )

        return {
            'text': formatted_text,
            'metadata': {
                'project_type': project_type,
                'style': style,
                'length_chars': len(formatted_text),
                'example_count': len(relevant_examples) if use_examples and relevant_examples else 0,
                'rag_used': (settings.RAG_ENABLED and rag_service is not None and 
                            rag_service.is_available and use_examples and context_examples)
            }
        }

    def _format_output(self, text: str) -> str:
        """Базовая очистка и форматирование текста"""
        # Удаление лишних переносов, форматирование заголовков
        lines = text.strip().split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith('## '):
                formatted_lines.append(f"# {line[3:]}")
            elif line.startswith('### '):
                formatted_lines.append(f"## {line[4:]}")
            elif line and not line.isspace():
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)