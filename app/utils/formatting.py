from typing import List, Dict, Any
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class BlockType(str, Enum):
    HEADING1 = 'heading1'
    HEADING2 = 'heading2'
    HEADING3 = 'heading3'
    PARAGRAPH = 'paragraph'
    LIST_ITEM = 'list_item'
    NUMBERED_ITEM = 'numbered_item'
    EMPTY = 'empty'


class DocxBlock:
    def __init__(self, block_type: BlockType, content: str, level: int = 0):
        self.block_type = block_type
        self.content = content
        self.level = level

    def __repr__(self):
        return f'DocxBlock(type={self.block_type}, content={self.content[:50]}..., level={self.level})'


def parse_markdown_to_docx_structure(text: str) -> List[DocxBlock]:
    """
    Парсинг markdown текста в структуру для создания .docx документа.

    Args:
        text: Текст в формате markdown

    Returns:
        Список блоков DocxBlock для обработки в DocxGenerator
    """
    blocks = []
    lines = text.split('\n')

    for line in lines:
        line = line.rstrip()
        
        if not line or line.isspace():
            blocks.append(DocxBlock(BlockType.EMPTY, ''))
            continue

        # Заголовки
        if line.startswith('# '):
            blocks.append(DocxBlock(BlockType.HEADING1, line[2:].strip()))
        elif line.startswith('## '):
            blocks.append(DocxBlock(BlockType.HEADING2, line[3:].strip()))
        elif line.startswith('### '):
            blocks.append(DocxBlock(BlockType.HEADING3, line[4:].strip()))
        
        # Маркированные списки
        elif line.startswith('- ') or line.startswith('* '):
            content = line[2:].strip()
            blocks.append(DocxBlock(BlockType.LIST_ITEM, content))
        
        # Нумерованные списки (формат: "1. ", "2. " и т.д.)
        elif len(line) >= 3 and line[0].isdigit() and line[1:3] == '. ':
            content = line[3:].strip()
            blocks.append(DocxBlock(BlockType.NUMBERED_ITEM, content))
        
        # Обычный параграф
        else:
            blocks.append(DocxBlock(BlockType.PARAGRAPH, line.strip()))

    logger.debug('markdown_parsed', total_blocks=len(blocks), block_types=[b.block_type.value for b in blocks[:10]])
    return blocks


def format_text_for_docx(text: str) -> str:
    """
    Базовая очистка и форматирование текста перед парсингом.

    Args:
        text: Исходный текст

    Returns:
        Отформатированный текст
    """
    # Удаление лишних пробелов и переносов
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.rstrip()
        if line:
            formatted_lines.append(line)
        elif formatted_lines and formatted_lines[-1]:  # Добавляем пустую строку только если предыдущая не пустая
            formatted_lines.append('')

    result = '\n'.join(formatted_lines)
    logger.debug('text_formatted', original_length=len(text), formatted_length=len(result))
    return result
