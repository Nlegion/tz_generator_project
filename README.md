# TZ Generator

Система для автоматической генерации и проверки технических заданий (ТЗ) с использованием локальной LLM модели GigaChat3-10B и технологии RAG (Retrieval-Augmented Generation) для эффективной работы с коллекцией примеров ТЗ.

## Основные возможности

- **Генерация ТЗ**: Автоматическое создание технических заданий на основе описания проекта
- **Валидация ТЗ**: Проверка качества и полноты существующих технических заданий
- **RAG система**: Семантический поиск релевантных фрагментов из коллекции ТЗ без перегрузки контекста модели
- **GPU/CPU fallback**: Автоматическое определение и использование GPU с fallback на CPU
- **Экспорт в .docx**: Генерация документов с применением стилей и форматирования
- **Веб-интерфейс**: Удобный UI для работы с системой
- **API**: RESTful API для интеграции с другими системами

## Технологический стек

- **Backend**: FastAPI, Python 3.12
- **LLM**: GigaChat3-10B-A1.8B (GGUF) через llama-cpp-python
- **RAG**: sentence-transformers + ChromaDB
- **Документы**: python-docx
- **Логирование**: structlog + rich
- **Тестирование**: pytest, pytest-asyncio, pytest-cov

## Установка

### Требования

- Python 3.12+
- CUDA 12.1+ (для GPU режима, опционально)
- NVIDIA GPU с поддержкой CUDA (опционально, для ускорения)
- ~8GB RAM (для модели и RAG системы)
- Модель GigaChat3-10B-A1.8B-q6_k.gguf (разместить в `data/model/`)

### Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv .venv

# Активация виртуального окружения
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### Установка llama-cpp-python с CUDA (для GPU)

```bash
# Для Windows с CUDA 12.1
set CMAKE_ARGS=-DGGML_CUBLAS=ON
set FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Подготовка модели

Поместите файл модели `GigaChat3-10B-A1.8B-q6_k.gguf` в директорию `data/model/`.

### Подготовка примеров ТЗ

Поместите примеры технических заданий (файлы `.txt` или `.docx`) в директорию `data/tz_examples/`. Они будут автоматически проиндексированы при первом запуске.

## Конфигурация

Настройки приложения находятся в `app/core/config.py` и могут быть переопределены через файл `.env`:

```env
# Пути
MODEL_PATH=./data/model/GigaChat3-10B-A1.8B-q6_k.gguf
EXAMPLES_DIR=./data/tz_examples/
OUTPUT_DIR=./data/outputs/

# LLM параметры
N_GPU_LAYERS=40
CONTEXT_SIZE=8192
MAX_TOKENS=2000
TEMPERATURE=0.3

# GPU/CPU настройки
USE_GPU=true
AUTO_DETECT_GPU=true
GPU_FALLBACK_ENABLED=true
CPU_THREADS=4

# RAG настройки
RAG_ENABLED=true
RAG_EMBEDDINGS_MODEL=intfloat/multilingual-e5-small
RAG_LANGUAGE=ru
RAG_CHUNK_SIZE=500
RAG_TOP_K=3
RAG_MAX_CONTEXT_LENGTH=2000
```

## Запуск

### Локальный запуск

```bash
# Активация виртуального окружения
.venv\Scripts\activate  # Windows
# или
source .venv/bin/activate  # Linux/Mac

# Запуск сервера
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Приложение будет доступно по адресу: http://localhost:8000

### Docker

```bash
# Сборка образа
docker build -t tz-generator .

# Запуск контейнера
docker-compose up
```

## Использование

### Веб-интерфейс

1. **Генерация ТЗ**: Откройте http://localhost:8000/generate
   - Введите описание проекта
   - Выберите тип проекта и стиль документа
   - Нажмите "Сгенерировать ТЗ"
   - Скачайте готовый .docx файл

2. **Валидация ТЗ**: Откройте http://localhost:8000/validate
   - Загрузите файл ТЗ или вставьте текст
   - Получите оценку качества и рекомендации

### API

#### Генерация ТЗ

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Веб-приложение для управления задачами",
    "project_type": "web",
    "style": "formal",
    "use_examples": true
  }'
```

#### Валидация ТЗ

```bash
# Валидация по тексту
curl -X POST "http://localhost:8000/api/v1/validate" \
  -F "text=Техническое задание на разработку..."

# Валидация по файлу
curl -X POST "http://localhost:8000/api/v1/validate" \
  -F "file=@tz_example.docx"
```

#### Статистика RAG

```bash
curl "http://localhost:8000/api/v1/rag/stats"
```

#### Переиндексация RAG

```bash
curl -X POST "http://localhost:8000/api/v1/rag/reindex?force=true"
```

### Полная документация API

Интерактивная документация доступна по адресам:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## RAG система

Система использует RAG (Retrieval-Augmented Generation) для эффективной работы с коллекцией примеров ТЗ:

- **Семантический поиск**: Поиск релевантных фрагментов по смыслу, а не только по ключевым словам
- **Оптимизация для русского языка**: Использование специализированных моделей embeddings
- **Эффективное использование контекста**: Только релевантные фрагменты вместо полных документов
- **Автоматическая индексация**: Примеры индексируются при первом запуске

### Настройка RAG

```python
# В .env или config.py
RAG_ENABLED=true                    # Включить/выключить RAG
RAG_EMBEDDINGS_MODEL=intfloat/multilingual-e5-small  # Модель embeddings
RAG_TOP_K=3                         # Количество релевантных чанков
RAG_MAX_CONTEXT_LENGTH=2000         # Максимальная длина контекста
```

## GPU/CPU Fallback

Система автоматически определяет доступность GPU и переключается на CPU при необходимости:

- **Автоматическое определение GPU**: Проверка через nvidia-smi и переменные окружения
- **Fallback на CPU**: Автоматическое переключение при недоступности GPU
- **Оптимизация для CPU**: Настройка количества потоков и размера батча

### Настройка GPU/CPU

```python
USE_GPU=true                        # Принудительное использование GPU
AUTO_DETECT_GPU=true                # Автоматическое определение
GPU_FALLBACK_ENABLED=true           # Разрешить fallback на CPU
CPU_THREADS=4                       # Потоки для CPU режима
```

## Тестирование

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# С покрытием
pytest tests/ --cov=app --cov-report=html

# Конкретный модуль
pytest tests/test_rag_service.py -v
```

### Тесты в Docker

```bash
# Запуск тестов в контейнере
docker-compose -f docker-compose.test.yml run --rm tests

# Или через скрипт
.\scripts\run_tests_docker.ps1  # Windows
./scripts/run_tests_docker.sh   # Linux/Mac
```

### Покрытие кода

Текущее покрытие: **84.81%**

Отчет доступен в `htmlcov/index.html` после запуска тестов с флагом `--cov-report=html`.

## Структура проекта

```
tz_generator_project/
├── app/
│   ├── api/              # API endpoints
│   ├── core/             # Конфигурация и логирование
│   ├── models/           # Модели данных и LLM
│   ├── services/         # Бизнес-логика (генерация, валидация, RAG, docx)
│   ├── templates/        # HTML шаблоны
│   └── utils/             # Утилиты (форматирование, файлы, чанкинг, GPU)
├── data/
│   ├── model/            # LLM модель
│   ├── outputs/          # Сгенерированные документы
│   ├── rag_db/           # Векторная БД Chroma (создается автоматически)
│   ├── templates/        # Шаблоны .docx
│   └── tz_examples/      # Примеры ТЗ для RAG
├── tests/                # Тесты
├── scripts/              # Вспомогательные скрипты
├── Dockerfile            # Docker образ для продакшена
├── Dockerfile.test       # Docker образ для тестов
├── docker-compose.yml    # Docker Compose конфигурация
└── requirements.txt      # Зависимости Python
```

## Особенности работы с русским языком

- **Модель embeddings**: `intfloat/multilingual-e5-small` оптимизирована для русского языка
- **Чанкинг**: Разбиение по предложениям с учетом русской пунктуации
- **Семантический поиск**: Понимание морфологии и синтаксиса русского языка
- **Обработка документов**: Поддержка структуры русских технических документов

## Производительность

- **GPU режим**: Использование GPU значительно ускоряет генерацию (особенно на Tesla T4)
- **CPU режим**: Работает на CPU, но медленнее (подходит для разработки и тестирования)
- **RAG индексация**: Первая индексация может занять время, последующие запуски быстрые
- **Кэширование**: Модели embeddings кэшируются автоматически

## Развертывание

### Docker Compose

```bash
# Запуск с GPU (если доступен)
docker-compose up -d

# Запуск без GPU (автоматический fallback на CPU)
docker-compose up -d
```

### Переменные окружения для Docker

```env
USE_GPU=true
AUTO_DETECT_GPU=true
GPU_FALLBACK_ENABLED=true
RAG_ENABLED=true
```

## Логирование

Система использует структурированное логирование через `structlog`:

- Логи сохраняются в `logs/tz_generator.log`
- Ошибки в `logs/errors.log`
- Консольный вывод с форматированием через `rich`

## Troubleshooting

### Модель не загружается

- Проверьте путь к модели в `MODEL_PATH`
- Убедитесь, что файл модели существует
- Проверьте права доступа к файлу

### RAG не работает

- Проверьте установку `sentence-transformers` и `chromadb`
- Убедитесь, что `RAG_ENABLED=true` в настройках
- Проверьте логи на наличие ошибок инициализации
- Система автоматически переключится на старый метод при ошибках RAG

### GPU не определяется

- Проверьте установку CUDA и драйверов NVIDIA
- Убедитесь, что `nvidia-smi` работает
- Система автоматически переключится на CPU режим

### Медленная генерация

- Используйте GPU для ускорения
- Уменьшите `MAX_TOKENS` для более быстрой генерации
- Оптимизируйте `RAG_TOP_K` и `RAG_MAX_CONTEXT_LENGTH`
