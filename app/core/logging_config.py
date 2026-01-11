import logging
import structlog
import os
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(log_level: str = "INFO"):
    """
    Настройка структурированного логирования для проекта.
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
    """
    # Создаем директорию для логов
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Преобразуем строковый уровень в числовой
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 1. Файловый обработчик для всех сообщений (с ротацией)
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "tz_generator.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # 2. Rich-обработчик для консоли (только INFO и выше)
    console = Console()
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
        markup=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False
    )
    rich_handler.setLevel(numeric_level)
    
    # 3. Отдельный обработчик для ошибок (опционально, можно закомментировать)
    error_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "errors.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # 4. Настройка structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # 5. Форматтер для structlog в консоли
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True)
    )
    rich_handler.setFormatter(console_formatter)
    
    # Форматтер для файлового вывода (более структурированный)
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer()
    )
    file_handler.setFormatter(file_formatter)
    
    # 6. Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Удаляем стандартные обработчики, если есть
    root_logger.handlers.clear()
    
    # Добавляем наши обработчики
    root_logger.addHandler(file_handler)
    root_logger.addHandler(rich_handler)
    root_logger.addHandler(error_handler)  # Опционально
    
    # Устанавливаем уровень логирования для некоторых шумных библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("llama_cpp").setLevel(logging.WARNING)
    
    # Получаем логгер для этого модуля
    logger = structlog.get_logger(__name__)
    logger.info(
        "Логирование инициализировано",
        log_level=log_level,
        log_dir=os.path.abspath(log_dir)
    )
    
    return logger


def get_logger(name: str = None):
    """
    Получение структурированного логгера.
    
    Args:
        name: Имя логгера (обычно __name__)
    
    Returns:
        BoundLogger для структурированного логирования
    """
    return structlog.get_logger(name or __name__)