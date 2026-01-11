import pytest
from app.utils.formatting import (
    parse_markdown_to_docx_structure,
    format_text_for_docx,
    DocxBlock,
    BlockType
)


def test_parse_heading1():
    """Тест парсинга заголовка уровня 1"""
    text = '# Заголовок 1'
    blocks = parse_markdown_to_docx_structure(text)
    
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.HEADING1
    assert blocks[0].content == 'Заголовок 1'


def test_parse_heading2():
    """Тест парсинга заголовка уровня 2"""
    text = '## Заголовок 2'
    blocks = parse_markdown_to_docx_structure(text)
    
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.HEADING2
    assert blocks[0].content == 'Заголовок 2'


def test_parse_heading3():
    """Тест парсинга заголовка уровня 3"""
    text = '### Заголовок 3'
    blocks = parse_markdown_to_docx_structure(text)
    
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.HEADING3
    assert blocks[0].content == 'Заголовок 3'


def test_parse_list_item():
    """Тест парсинга маркированного списка"""
    text = '- Элемент списка'
    blocks = parse_markdown_to_docx_structure(text)
    
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.LIST_ITEM
    assert blocks[0].content == 'Элемент списка'


def test_parse_list_item_asterisk():
    """Тест парсинга маркированного списка со звездочкой"""
    text = '* Элемент списка'
    blocks = parse_markdown_to_docx_structure(text)
    
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.LIST_ITEM
    assert blocks[0].content == 'Элемент списка'


def test_parse_numbered_item():
    """Тест парсинга нумерованного списка"""
    text = '1. Первый элемент'
    blocks = parse_markdown_to_docx_structure(text)
    
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.NUMBERED_ITEM
    assert blocks[0].content == 'Первый элемент'


def test_parse_paragraph():
    """Тест парсинга обычного параграфа"""
    text = 'Обычный текст параграфа'
    blocks = parse_markdown_to_docx_structure(text)
    
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.PARAGRAPH
    assert blocks[0].content == 'Обычный текст параграфа'


def test_parse_empty_line():
    """Тест парсинга пустой строки"""
    text = ''
    blocks = parse_markdown_to_docx_structure(text)
    
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.EMPTY


def test_parse_complex_markdown():
    """Тест парсинга сложного markdown документа"""
    text = """# Заголовок 1

## Заголовок 2

Обычный параграф.

- Список 1
- Список 2

1. Номер 1
2. Номер 2
"""
    blocks = parse_markdown_to_docx_structure(text)
    
    # Проверяем наличие всех типов блоков
    assert len(blocks) >= 8
    assert blocks[0].block_type == BlockType.HEADING1
    assert any(b.block_type == BlockType.HEADING2 for b in blocks)
    assert any(b.block_type == BlockType.PARAGRAPH for b in blocks)
    assert any(b.block_type == BlockType.LIST_ITEM for b in blocks)
    assert any(b.block_type == BlockType.NUMBERED_ITEM for b in blocks)


def test_format_text_for_docx():
    """Тест форматирования текста"""
    text = 'Строка 1\n\nСтрока 2  \n  Строка 3'
    formatted = format_text_for_docx(text)
    
    assert 'Строка 1' in formatted
    assert 'Строка 2' in formatted
    assert 'Строка 3' in formatted


def test_format_text_removes_extra_spaces():
    """Тест удаления лишних пробелов"""
    text = '   Строка с пробелами   \n\n   Еще строка   '
    formatted = format_text_for_docx(text)
    
    lines = formatted.split('\n')
    assert all(not line.startswith(' ') or not line.endswith(' ') for line in lines if line)
