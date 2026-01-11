from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class TZRequest(BaseModel):
    description: str = Field(..., description='Описание проекта для генерации ТЗ')
    project_type: str = Field(default='general', description='Тип проекта (general, web, mobile, desktop, api)')
    style: str = Field(default='formal', description='Стиль документа (formal, concise, detailed)')
    use_examples: bool = Field(default=True, description='Использовать примеры из базы')


class TZResponse(BaseModel):
    task_id: str = Field(..., description='Идентификатор задачи')
    text: str = Field(..., description='Сгенерированный текст ТЗ')
    metadata: Dict[str, Any] = Field(..., description='Метаданные генерации')
    download_url: str = Field(..., description='URL для скачивания .docx файла')


class ValidationRequest(BaseModel):
    text: Optional[str] = Field(None, description='Текст ТЗ для проверки')
    # file будет обрабатываться через UploadFile в routes


class ValidationResponse(BaseModel):
    is_valid: bool = Field(..., description='Валидность ТЗ (score >= 70)')
    score: int = Field(..., ge=0, le=100, description='Оценка полноты ТЗ (0-100)')
    issues: List[Dict[str, Any]] = Field(default_factory=list, description='Список найденных проблем')
    suggestions: List[str] = Field(default_factory=list, description='Предложения по улучшению')
    feedback: str = Field(..., description='Общая обратная связь')


class ChatMessage(BaseModel):
    role: str = Field(..., description='Роль в диалоге (user, assistant)')
    content: str = Field(..., description='Содержимое сообщения')


class ChatRequest(BaseModel):
    message: str = Field(..., description='Сообщение пользователя')
    conversation_id: Optional[str] = Field(None, description='ID диалога для продолжения')
    context: Optional[Dict[str, Any]] = Field(None, description='Контекст проекта')


class ChatResponse(BaseModel):
    response: str = Field(..., description='Ответ модели')
    conversation_id: str = Field(..., description='ID диалога')
    context: Optional[Dict[str, Any]] = Field(None, description='Обновленный контекст')
