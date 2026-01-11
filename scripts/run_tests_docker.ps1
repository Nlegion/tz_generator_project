# Скрипт для запуска тестов в Docker контейнере (PowerShell)

Write-Host "Building test image..." -ForegroundColor Green
docker build -f Dockerfile.test -t tz-generator-tests .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build image" -ForegroundColor Red
    exit 1
}

Write-Host "Running tests in container..." -ForegroundColor Green
$htmlcovPath = Join-Path $PSScriptRoot "..\htmlcov"
$logsPath = Join-Path $PSScriptRoot "..\logs"

New-Item -ItemType Directory -Force -Path $htmlcovPath | Out-Null
New-Item -ItemType Directory -Force -Path $logsPath | Out-Null

docker run --rm `
    -v "${htmlcovPath}:/app/htmlcov" `
    -v "${logsPath}:/app/logs" `
    tz-generator-tests

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nTests completed successfully! Coverage report available in htmlcov/" -ForegroundColor Green
} else {
    Write-Host "Tests failed!" -ForegroundColor Red
    exit 1
}
