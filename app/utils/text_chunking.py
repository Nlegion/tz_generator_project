"""Утилиты для разбиения текста ТЗ на чанки с учетом русского языка."""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TextChunk:
    """Класс для хранения чанка текста с метаданными."""
    text: str
    source: str  # Источник (имя файла)
    chunk_index: int  # Индекс чанка в документе
    start_pos: int  # Начальная позиция в исходном тексте
    end_pos: int  # Конечная позиция в исходном тексте
    block_type: str = "paragraph"  # Тип блока (paragraph, heading, list, etc.)


def split_into_sentences(text: str, language: str = "ru") -> List[str]:
    """
    Разбивает текст на предложения с учетом русского языка.

    Args:
        text: Текст для разбиения
        language: Язык текста (по умолчанию "ru")

    Returns:
        Список предложений
    """
    if language == "ru":
        # Паттерн для русского языка: точка, восклицательный, вопросительный знак
        # Учитываем сокращения (т.е., и т.д., и т.п., др.)
        sentence_endings = r'(?<=[.!?])\s+(?=[А-ЯЁ])'
        # Исключаем сокращения
        text = re.sub(r'\b(т\.е\.|и\.т\.д\.|и\.т\.п\.|др\.|пр\.|стр\.|г\.|гг\.|в\.|н\.э\.)', 
                     lambda m: m.group().replace('.', '·'), text)
        
        sentences = re.split(sentence_endings, text)
        # Восстанавливаем точки в сокращениях
        sentences = [s.replace('·', '.') for s in sentences]
    else:
        # Простое разбиение для других языков
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)

    # Фильтруем пустые предложения
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def chunk_tz_text(
    text: str,
    source: str = "unknown",
    chunk_size: int = 500,
    overlap: int = 50,
    language: str = "ru",
    chunk_by_sentences: bool = True
) -> List[TextChunk]:
    """
    Разбивает текст ТЗ на чанки с учетом русского языка.

    Args:
        text: Текст для разбиения
        source: Источник текста (имя файла)
        chunk_size: Размер чанка в символах
        overlap: Перекрытие между чанками в символах
        language: Язык текста (по умолчанию "ru")
        chunk_by_sentences: Разбивать по предложениям (сохранять целостность)

    Returns:
        Список объектов TextChunk
    """
    if not text or not text.strip():
        logger.debug('empty_text_chunking', source=source)
        return []

    chunks: List[TextChunk] = []
    text = text.strip()

    if chunk_by_sentences and language == "ru":
        # Разбиваем по предложениям для сохранения целостности
        sentences = split_into_sentences(text, language)
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        start_pos = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Если добавление предложения превысит размер чанка (и чанк не пуст)
            if current_length + sentence_length > chunk_size and current_chunk:
                # Сохраняем текущий чанк
                chunk_text = ' '.join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_index,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    block_type="paragraph"
                ))
                
                # Начинаем новый чанк с overlap
                if overlap > 0 and current_chunk:
                    # Берем последние предложения для overlap
                    overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else current_chunk[-1]
                    if len(overlap_text) > overlap:
                        overlap_text = overlap_text[-overlap:]
                    current_chunk = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text)
                    start_pos = end_pos - len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0
                    start_pos = end_pos
                
                chunk_index += 1
            
            # Добавляем предложение к текущему чанку
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 за пробел
        
        # Добавляем последний чанк
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                source=source,
                chunk_index=chunk_index,
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text),
                block_type="paragraph"
            ))
    else:
        # Простое разбиение по символам (без учета предложений)
        start_pos = 0
        chunk_index = 0
        
        while start_pos < len(text):
            end_pos = min(start_pos + chunk_size, len(text))
            chunk_text = text[start_pos:end_pos]
            
            # Если не последний чанк, пытаемся закончить на границе слова
            if end_pos < len(text):
                # Ищем последний пробел или знак препинания
                last_space = chunk_text.rfind(' ')
                if last_space > chunk_size * 0.7:  # Если пробел не слишком близко к началу
                    end_pos = start_pos + last_space
                    chunk_text = text[start_pos:end_pos]
            
            chunks.append(TextChunk(
                text=chunk_text.strip(),
                source=source,
                chunk_index=chunk_index,
                start_pos=start_pos,
                end_pos=end_pos,
                block_type="paragraph"
            ))
            
            # Переход к следующему чанку с учетом overlap
            start_pos = max(start_pos + 1, end_pos - overlap)
            chunk_index += 1

    logger.debug(
        'text_chunked',
        source=source,
        total_chunks=len(chunks),
        total_length=len(text),
        avg_chunk_size=sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
    )

    return chunks


def detect_block_type(text: str) -> str:
    """
    Определяет тип блока текста (заголовок, список, параграф).

    Args:
        text: Текст блока

    Returns:
        Тип блока: "heading", "list", "paragraph"
    """
    text_stripped = text.strip()
    
    # Заголовок (начинается с # или все заглавные)
    if text_stripped.startswith('#'):
        return "heading"
    if text_stripped.isupper() and len(text_stripped) < 100:
        return "heading"
    
    # Список (начинается с маркера или номера)
    if re.match(r'^[\*\-\+]\s', text_stripped) or re.match(r'^\d+[\.\)]\s', text_stripped):
        return "list"
    
    return "paragraph"
