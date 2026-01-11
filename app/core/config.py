import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Пути к файлам
    MODEL_PATH: str = "./data/model/GigaChat3-10B-A1.8B-q6_k.gguf"
    EXAMPLES_DIR: str = "./data/tz_examples/"
    TEMPLATE_PATH: str = "./data/templates/tz_template.docx"
    OUTPUT_DIR: str = "./data/outputs/"

    # Параметры модели
    N_GPU_LAYERS: int = 40
    CONTEXT_SIZE: int = 8192
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.3

    # Параметры генерации
    GENERATION_TIMEOUT: int = 300  # секунд

    # GPU/CPU настройки
    USE_GPU: bool = True  # Принудительное использование GPU (если доступна)
    AUTO_DETECT_GPU: bool = True  # Автоматическое определение доступности GPU
    GPU_FALLBACK_ENABLED: bool = True  # Разрешить fallback на CPU при ошибке GPU

    # CPU режим настройки
    CPU_THREADS: int = 4  # Количество потоков для CPU режима
    CPU_BATCH_SIZE: int = 512  # Размер батча для CPU режима

    # RAG настройки
    RAG_ENABLED: bool = True  # Включить/выключить RAG
    RAG_EMBEDDINGS_MODEL: str = "intfloat/multilingual-e5-small"  # Модель embeddings (оптимизирована для русского)
    RAG_LANGUAGE: str = "ru"  # Основной язык для обработки (русский)
    RAG_CHUNK_SIZE: int = 500  # Размер чанка в символах (учитывая русский текст)
    RAG_CHUNK_OVERLAP: int = 50  # Перекрытие чанков в символах
    RAG_TOP_K: int = 3  # Количество релевантных чанков для извлечения
    RAG_MAX_CONTEXT_LENGTH: int = 2000  # Максимальная длина контекста в символах
    RAG_DB_PATH: str = "./data/rag_db"  # Путь к векторной БД
    RAG_CHUNK_BY_SENTENCES: bool = True  # Разбивать по предложениям (важно для русского)
    RAG_REINDEX_ON_STARTUP: bool = False  # Переиндексировать при каждом запуске

    class Config:
        env_file = ".env"
        extra = "ignore"  # Игнорировать лишние поля из .env


settings = Settings()