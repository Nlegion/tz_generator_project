import os
from typing import List, Dict
from docx import Document
import structlog

logger = structlog.get_logger(__name__)


def load_examples(examples_dir: str = './data/tz_examples/') -> List[str]:
    """
    Загрузка примеров ТЗ из директории.

    Args:
        examples_dir: Путь к директории с примерами

    Returns:
        Список текстов примеров ТЗ
    """
    examples = []
    
    if not os.path.exists(examples_dir):
        logger.warning('examples_dir_not_found', examples_dir=examples_dir)
        return examples

    try:
        for filename in os.listdir(examples_dir):
            file_path = os.path.join(examples_dir, filename)
            
            if not os.path.isfile(file_path):
                continue

            try:
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            examples.append(content)
                            logger.debug('example_loaded', filename=filename, length=len(content))
                
                elif filename.endswith('.docx'):
                    doc = Document(file_path)
                    content = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
                    if content:
                        examples.append(content)
                        logger.debug('example_loaded', filename=filename, length=len(content), format='docx')
            
            except (UnicodeDecodeError, OSError) as e:
                logger.warning('example_load_error', filename=filename, error=str(e))
                continue

        logger.info('examples_loaded', count=len(examples), examples_dir=examples_dir)
        return examples

    except OSError as e:
        msg = f'Ошибка доступа к директории примеров: {e}'
        logger.error('examples_dir_access_error', examples_dir=examples_dir, error=str(e))
        raise OSError(msg) from e


def get_relevant_examples(project_description: str, examples: List[str], n_examples: int = 2) -> List[str]:
    """
    Выбор релевантных примеров на основе описания проекта.

    Args:
        project_description: Описание проекта
        examples: Список всех примеров
        n_examples: Количество примеров для возврата

    Returns:
        Список релевантных примеров
    """
    if not examples:
        logger.debug('no_examples_available')
        return []

    if len(examples) <= n_examples:
        logger.debug('returning_all_examples', count=len(examples))
        return examples

    # Простая эвристика: выбираем примеры, которые содержат похожие ключевые слова
    description_lower = project_description.lower()
    description_words = set(description_lower.split())

    scored_examples = []
    for example in examples:
        example_lower = example.lower()
        example_words = set(example_lower.split())
        
        # Подсчет пересечения слов
        common_words = description_words.intersection(example_words)
        score = len(common_words)
        
        scored_examples.append((score, example))

    # Сортируем по релевантности и берем топ-N
    scored_examples.sort(key=lambda x: x[0], reverse=True)
    relevant = [ex[1] for ex in scored_examples[:n_examples]]

    logger.debug(
        'relevant_examples_selected',
        total_examples=len(examples),
        selected_count=len(relevant),
        top_scores=[ex[0] for ex in scored_examples[:n_examples]]
    )

    return relevant
