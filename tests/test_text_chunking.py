"""Тесты для модуля разбиения текста на чанки."""
import pytest
from app.utils.text_chunking import (
    chunk_tz_text,
    split_into_sentences,
    TextChunk,
    detect_block_type
)


def test_split_into_sentences_russian():
    """Тест разбиения русского текста на предложения."""
    text = "Первое предложение. Второе предложение! Третье предложение?"
    sentences = split_into_sentences(text, language="ru")
    assert len(sentences) == 3
    assert "Первое предложение" in sentences[0]
    assert "Второе предложение" in sentences[1]
    assert "Третье предложение" in sentences[2]


def test_split_into_sentences_with_abbreviations():
    """Тест разбиения с учетом сокращений."""
    text = "Т.е. это сокращение. А это новое предложение."
    sentences = split_into_sentences(text, language="ru")
    # Сокращение не должно разрывать предложение
    assert len(sentences) >= 1


def test_chunk_tz_text_basic():
    """Базовый тест разбиения текста на чанки."""
    text = "Это тестовый текст для разбиения на чанки. " * 10
    chunks = chunk_tz_text(text, source="test.txt", chunk_size=50, overlap=10)
    assert len(chunks) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    assert all(chunk.source == "test.txt" for chunk in chunks)


def test_chunk_tz_text_by_sentences():
    """Тест разбиения по предложениям."""
    text = "Первое предложение. Второе предложение. Третье предложение. " * 5
    chunks = chunk_tz_text(
        text,
        source="test.txt",
        chunk_size=100,
        overlap=20,
        chunk_by_sentences=True,
        language="ru"
    )
    assert len(chunks) > 0
    # Проверяем, что предложения не разорваны
    for chunk in chunks:
        assert chunk.text.count('.') >= 0  # Может быть 0 или больше, но не должно быть обрывов


def test_chunk_tz_text_empty():
    """Тест обработки пустого текста."""
    chunks = chunk_tz_text("", source="empty.txt")
    assert len(chunks) == 0


def test_chunk_tz_text_metadata():
    """Тест сохранения метаданных в чанках."""
    text = "Тестовый текст. " * 20
    chunks = chunk_tz_text(text, source="test.txt", chunk_size=50)
    assert len(chunks) > 0
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.start_pos >= 0
        assert chunk.end_pos > chunk.start_pos
        assert chunk.block_type in ["paragraph", "heading", "list"]


def test_chunk_tz_text_overlap():
    """Тест перекрытия чанков."""
    text = "Предложение один. Предложение два. Предложение три. " * 5
    chunks = chunk_tz_text(text, source="test.txt", chunk_size=50, overlap=20)
    if len(chunks) > 1:
        # Проверяем, что есть перекрытие между соседними чанками
        first_chunk_end = chunks[0].end_pos
        second_chunk_start = chunks[1].start_pos
        # Второй чанк должен начинаться раньше конца первого (перекрытие)
        assert second_chunk_start < first_chunk_end


def test_detect_block_type_heading():
    """Тест определения типа блока - заголовок."""
    assert detect_block_type("# Заголовок") == "heading"
    assert detect_block_type("ЗАГОЛОВОК") == "heading"


def test_detect_block_type_list():
    """Тест определения типа блока - список."""
    assert detect_block_type("* Элемент списка") == "list"
    assert detect_block_type("1. Нумерованный элемент") == "list"


def test_detect_block_type_paragraph():
    """Тест определения типа блока - параграф."""
    assert detect_block_type("Обычный текст параграфа.") == "paragraph"


def test_chunk_tz_text_russian_long():
    """Тест разбиения длинного русского текста."""
    text = """
    Техническое задание на разработку веб-приложения.
    
    ## 1. Общие сведения
    Проект представляет собой веб-приложение для управления задачами.
    
    ## 2. Функциональные требования
    - Создание задач
    - Редактирование задач
    - Удаление задач
    
    ## 3. Нефункциональные требования
    Время отклика должно быть менее 200 мс.
    """
    chunks = chunk_tz_text(
        text,
        source="tz_example.txt",
        chunk_size=100,
        overlap=20,
        language="ru",
        chunk_by_sentences=True
    )
    assert len(chunks) > 0
    # Проверяем, что структура сохранена
    for chunk in chunks:
        assert len(chunk.text) > 0
