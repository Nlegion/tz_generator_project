from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
import os
import uuid
import structlog
from docx import Document

from app.services.tz_generator import TZGenerator
from app.services.tz_validator import TZValidator
from app.services.docx_service import DocxGenerator
from app.services.rag_service import RAGService
from app.models.tz_schema import (
    TZRequest, TZResponse, ValidationRequest, ValidationResponse,
    ChatRequest, ChatResponse
)
from app.models.llm_model import model
from app.core.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["tz"])

# Инициализация сервисов
tz_generator = TZGenerator()
tz_validator = TZValidator()
docx_generator = DocxGenerator()

# RAG сервис (инициализируется в main.py при startup)
rag_service: Optional[RAGService] = None


@router.post("/generate", response_model=TZResponse)
async def generate_tz(
        request: TZRequest,
        background_tasks: BackgroundTasks
):
    """Генерация нового ТЗ"""
    try:
        # Генерация текста
        result = tz_generator.generate(
            project_description=request.description,
            project_type=request.project_type,
            style=request.style,
            use_examples=request.use_examples
        )

        # Создание документа
        task_id = str(uuid.uuid4())
        output_dir = "./data/outputs/"
        output_path = os.path.join(output_dir, f"{task_id}.docx")

        docx_generator.create_document(result, output_path)

        return {
            "task_id": task_id,
            "text": result["text"],
            "metadata": result["metadata"],
            "download_url": f"/api/v1/download/{task_id}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationResponse)
async def validate_tz(
        file: Optional[UploadFile] = File(None),
        text: Optional[str] = Form(None)
):
    """Проверка существующего ТЗ"""
    try:
        tz_text = None

        # Получаем текст для проверки
        if file:
            # Проверка размера файла (максимум 10 МБ)
            file_content = await file.read()
            if len(file_content) > 10 * 1024 * 1024:
                msg = 'Файл слишком большой (максимум 10 МБ)'
                logger.warning('file_too_large', filename=file.filename, size=len(file_content))
                raise HTTPException(status_code=400, detail=msg)

            # Обработка .docx файлов
            if file.filename and file.filename.endswith('.docx'):
                try:
                    # Сохраняем временный файл для парсинга
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                        tmp_file.write(file_content)
                        tmp_path = tmp_file.name

                    doc = Document(tmp_path)
                    tz_text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
                    os.unlink(tmp_path)
                    logger.debug('docx_parsed', filename=file.filename, text_length=len(tz_text))
                except (OSError, ValueError) as e:
                    msg = f'Ошибка парсинга .docx файла: {e}'
                    logger.error('docx_parse_error', filename=file.filename, error=str(e))
                    raise HTTPException(status_code=400, detail=msg) from e
            else:
                # Текстовый файл
                try:
                    tz_text = file_content.decode('utf-8')
                except UnicodeDecodeError as e:
                    msg = f'Ошибка декодирования файла: {e}'
                    logger.error('file_decode_error', filename=file.filename, error=str(e))
                    raise HTTPException(status_code=400, detail=msg) from e

        elif text:
            tz_text = text
        else:
            msg = 'Не предоставлен текст для проверки (файл или текст)'
            logger.warning('validation_no_input')
            raise HTTPException(status_code=400, detail=msg)

        if not tz_text or not tz_text.strip():
            msg = 'Текст для проверки пуст'
            logger.warning('validation_empty_text')
            raise HTTPException(status_code=400, detail=msg)

        # Валидация
        validation_result = tz_validator.validate(tz_text)

        logger.info(
            'validation_complete',
            score=validation_result['completeness_score'],
            issues_count=len(validation_result.get('issues', []))
        )

        return {
            'is_valid': validation_result['completeness_score'] >= 70,
            'score': validation_result['completeness_score'],
            'issues': validation_result['issues'],
            'suggestions': validation_result['suggestions'],
            'feedback': validation_result['overall_feedback']
        }

    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError) as e:
        msg = f'Ошибка валидации ТЗ: {e}'
        logger.error('validation_error', error=str(e))
        raise HTTPException(status_code=500, detail=msg) from e


@router.get("/download/{task_id}")
async def download_tz(task_id: str):
    """Скачивание сгенерированного .docx"""
    file_path = f"./data/outputs/{task_id}.docx"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден")

    return FileResponse(
        path=file_path,
        filename=f"ТЗ_{task_id[:8]}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_with_llm(request: ChatRequest):
    """Диалог с LLM для уточнения требований проекта"""
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Формируем промпт для диалога
        context_info = ''
        if request.context:
            context_info = f"\nКонтекст проекта: {request.context.get('description', '')}"
        
        chat_prompt = f"""Ты — помощник системного аналитика. Помоги пользователю уточнить требования к проекту.

{context_info}

Сообщение пользователя: {request.message}

Ответь кратко и по делу, задавай уточняющие вопросы если нужно."""

        # Генерируем ответ
        if model is None:
            msg = 'Model not initialized'
            logger.error('chat_error', error=msg)
            raise HTTPException(status_code=500, detail=msg)
        response_text = model.generate(chat_prompt, max_tokens=500, temperature=0.7)

        logger.info('chat_complete', conversation_id=conversation_id, message_length=len(request.message))

        return {
            'response': response_text,
            'conversation_id': conversation_id,
            'context': request.context
        }

    except (ValueError, RuntimeError, TimeoutError) as e:
        msg = f'Ошибка генерации ответа: {e}'
        logger.error('chat_error', error=str(e), conversation_id=request.conversation_id)
        raise HTTPException(status_code=500, detail=msg) from e


@router.get("/examples")
async def get_examples():
    """Получение списка доступных примеров ТЗ"""
    examples_dir = './data/tz_examples/'
    examples = []

    try:
        if os.path.exists(examples_dir):
            for filename in os.listdir(examples_dir):
                file_path = os.path.join(examples_dir, filename)
                if os.path.isfile(file_path) and (filename.endswith('.txt') or filename.endswith('.docx')):
                    examples.append({
                        'name': filename,
                        'size': os.path.getsize(file_path)
                    })
    except OSError as e:
        logger.warning('examples_list_error', error=str(e))
        # Возвращаем пустой список при ошибке

    return {'examples': examples}


@router.get("/rag/stats")
async def get_rag_stats():
    """Получение статистики RAG системы"""
    global rag_service
    
    if rag_service is None:
        # Пытаемся получить из tz_generator
        from app.services.tz_generator import rag_service as tz_rag_service
        rag_service = tz_rag_service
    
    if rag_service is None:
        return {
            "status": "unavailable",
            "enabled": settings.RAG_ENABLED,
            "message": "RAG сервис не инициализирован"
        }
    
    stats = rag_service.get_stats()
    return stats


@router.post("/rag/reindex")
async def reindex_rag(force: bool = False):
    """Переиндексация примеров ТЗ в RAG системе"""
    global rag_service
    
    if rag_service is None:
        from app.services.tz_generator import rag_service as tz_rag_service
        rag_service = tz_rag_service
    
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG сервис не инициализирован")
    
    if not rag_service.is_available:
        raise HTTPException(status_code=503, detail="RAG недоступен")
    
    try:
        result = rag_service.index_examples(force_reindex=force)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Ошибка индексации"))
        return result
    except Exception as e:
        logger.error('rag_reindex_error', error=str(e))
        raise HTTPException(status_code=500, detail=f"Ошибка переиндексации: {str(e)}")