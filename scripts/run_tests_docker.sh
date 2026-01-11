#!/bin/bash
# Скрипт для запуска тестов в Docker контейнере

set -e

echo "Building test image..."
docker build -f Dockerfile.test -t tz-generator-tests .

echo "Running tests in container..."
docker run --rm \
    -v "$(pwd)/htmlcov:/app/htmlcov" \
    -v "$(pwd)/logs:/app/logs" \
    tz-generator-tests

echo "Tests completed! Coverage report available in htmlcov/"
