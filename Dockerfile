# Используем базовый образ с CUDA для Tesla T4
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Настройка рабочей директории
WORKDIR /app

# Установка Python зависимостей
COPY requirements.txt .

# Установка llama-cpp-python с поддержкой CUDA
# Для Tesla T4 (compute capability 7.5)
ENV CMAKE_ARGS="-DGGML_CUBLAS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
ENV FORCE_CMAKE=1
ENV CUDA_COMPUTE_CAP=75

RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install llama-cpp-python --force-reinstall --no-cache-dir --verbose && \
    pip install -r requirements.txt

# Копирование кода
COPY app/ ./app/
COPY data/templates/ ./data/templates/
COPY pyproject.toml ./

# Создание необходимых директорий
RUN mkdir -p ./data/outputs ./data/tz_examples ./data/templates ./data/model ./logs

# Скачивание модели (опционально, можно монтировать volume)
# RUN python scripts/download_model.py

# Открытие порта
EXPOSE 8000

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Команда запуска
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]