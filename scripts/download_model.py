import requests
import os
from tqdm import tqdm
import sys

# === КОНФИГУРАЦИЯ ===
# ИСПРАВЛЕННЫЙ URL: заменяем '/blob/' на '/resolve/'
MODEL_URL = "https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-GGUF/resolve/main/GigaChat3-10B-A1.8B-q6_k.gguf"
MODEL_PATH = "./data/model/GigaChat3-10B-A1.8B-q6_k.gguf"


# === ФУНКЦИЯ ЗАГРУЗКИ С ПРОВЕРКОЙ ===
def download_file(url: str, destination: str):
    """
    Загружает файл с прогресс-баром.
    Проверяет, не скачан ли файл уже частично, для возможности дозагрузки.
    """
    # Создаем директорию, если её нет
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Проверяем, существует ли уже файл
    if os.path.exists(destination):
        file_size = os.path.getsize(destination)
        # Пробуем возобновить загрузку, отправив заголовок Range
        headers = {'Range': f'bytes={file_size}-'} if file_size > 0 else {}
        print(f"Файл {destination} уже существует ({file_size} байт). Пробуем возобновить загрузку...")
        mode = 'ab'  # Дозапись
    else:
        headers = {}
        mode = 'wb'  # Новая запись
        file_size = 0

    # Делаем запрос с заголовками для возможного возобновления
    response = requests.get(url, headers=headers, stream=True, timeout=30)

    # Обрабатываем ответ сервера
    if response.status_code == 416:  # Запрошенный диапазон не выполним (файл уже полностью скачан)
        print("Файл уже загружен полностью.")
        return True
    elif response.status_code not in [200, 206]:  # 206 - Partial Content для возобновления
        print(f"Ошибка при загрузке: HTTP {response.status_code}")
        return False

    # Получаем общий размер файла
    total_size = int(response.headers.get('content-length', 0)) + file_size

    # Открываем файл в нужном режиме (дозапись или новая запись)
    with open(destination, mode) as file, tqdm(
            desc=f"Загрузка {os.path.basename(destination)}",
            total=total_size,
            initial=file_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            dynamic_ncols=True  # Автоподстройка под ширину терминала
    ) as progress_bar:

        # Скачиваем данные частями
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # Фильтруем keep-alive chunks
                size = file.write(chunk)
                progress_bar.update(size)

    return True


# === ТОЧКА ВХОДА ===
if __name__ == "__main__":
    print("=" * 60)
    print("Скачивание модели GigaChat3-10B (q6_k версия)")
    print("Размер файла: ~8.8 ГБ")
    print("=" * 60)

    # Проверяем наличие модели
    if os.path.exists(MODEL_PATH):
        print(f"Модель уже существует: {MODEL_PATH}")
        print("Размер: {:.2f} ГБ".format(os.path.getsize(MODEL_PATH) / (1024 ** 3)))
        answer = input("Хотите перезаписать? (y/N): ").strip().lower()
        if answer != 'y':
            print("Загрузка отменена.")
            sys.exit(0)
        else:
            print("Удаляю существующий файл...")
            os.remove(MODEL_PATH)

    try:
        # Загружаем модель
        print(f"\nИсточник: {MODEL_URL}")
        print(f"Назначение: {MODEL_PATH}")
        print("\nНачинаю загрузку... (это может занять время)")

        success = download_file(MODEL_URL, MODEL_PATH)

        if success:
            print("\n" + "=" * 60)
            print("✅ Загрузка успешно завершена!")
            print(f"Модель сохранена в: {MODEL_PATH}")

            # Показываем информацию о скачанном файле
            if os.path.exists(MODEL_PATH):
                size_gb = os.path.getsize(MODEL_PATH) / (1024 ** 3)
                print(f"Размер файла: {size_gb:.2f} ГБ")
            print("=" * 60)
        else:
            print("\n❌ Загрузка не удалась. Проверьте URL и подключение к интернету.")
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        print("\n❌ Ошибка подключения. Проверьте интернет-соединение.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Загрузка прервана пользователем.")
        if os.path.exists(MODEL_PATH):
            size = os.path.getsize(MODEL_PATH)
            print(f"Частично скачано: {size / (1024 ** 3):.2f} ГБ")
            print("Для возобновления запустите скрипт снова.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        sys.exit(1)