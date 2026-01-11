from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging_config import setup_logging, get_logger
from app.utils.gpu_detection import get_device_info

# Настройка логирования
setup_logging()
logger = get_logger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="TZ Generator API",
    description="API для генерации и проверки технических заданий",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение маршрутов
app.include_router(api_router)

# Статические файлы и шаблоны
templates = Jinja2Templates(directory="app/templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Роуты для фронтенда
@app.get("/")
async def frontend():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.get("/generate")
async def generate_page():
    return templates.TemplateResponse("generate.html", {"request": {}})


@app.get("/validate")
async def validate_page():
    return templates.TemplateResponse("validate.html", {"request": {}})


# Эндпоинт состояния системы
@app.get("/health")
async def health_check():
    """Проверка состояния системы и информации о режиме работы"""
    model_loaded = False
    device_mode = 'unknown'
    gpu_available = False

    try:
        from app.models.llm_model import model
        if model is not None and hasattr(model, 'llm') and model.llm is not None:
            model_loaded = True
            device_mode = getattr(model, 'device_mode', 'unknown')
            gpu_available = getattr(model, 'gpu_available', False)
    except Exception:
        pass

    device_info = get_device_info()

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device_mode": device_mode,
        "gpu_available": gpu_available,
        "device_info": device_info,
        "examples_count": len(os.listdir(settings.EXAMPLES_DIR)) if os.path.exists(settings.EXAMPLES_DIR) else 0,
        "settings": {
            "use_gpu": settings.USE_GPU,
            "auto_detect_gpu": settings.AUTO_DETECT_GPU,
            "gpu_fallback_enabled": settings.GPU_FALLBACK_ENABLED,
            "cpu_threads": settings.CPU_THREADS
        }
    }


@app.on_event("startup")
async def startup_event():
    """Действия при запуске приложения"""
    logger.info('app_startup', action='startup')

    # Создание необходимых директорий
    os.makedirs('./data/outputs', exist_ok=True)
    os.makedirs('./data/tz_examples', exist_ok=True)
    os.makedirs('./data/templates', exist_ok=True)
    os.makedirs(settings.RAG_DB_PATH, exist_ok=True)

    # Инициализация RAG
    if settings.RAG_ENABLED:
        try:
            from app.services.rag_service import RAGService
            rag_service = RAGService()
            
            # Передаем экземпляр в tz_generator и routes
            import app.services.tz_generator as tz_gen_module
            tz_gen_module.rag_service = rag_service
            
            import app.api.routes as routes_module
            routes_module.rag_service = rag_service
            
            # Индексация примеров
            if rag_service.is_available:
                index_result = rag_service.index_examples(force_reindex=settings.RAG_REINDEX_ON_STARTUP)
                logger.info('rag_startup_complete', index_result=index_result)
            else:
                logger.warning('rag_startup_failed', message='RAG недоступен, будет использован fallback')
        except Exception as e:
            logger.error('rag_startup_error', error=str(e), message='RAG не инициализирован')
    else:
        logger.info('rag_disabled_in_config')

    logger.info(
        'app_startup_complete',
        outputs_dir=os.path.abspath('./data/outputs'),
        examples_dir=os.path.abspath('./data/tz_examples'),
        templates_dir=os.path.abspath('./data/templates'),
        rag_enabled=settings.RAG_ENABLED
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Действия при остановке приложения"""
    logger.info('app_shutdown', action='shutdown')