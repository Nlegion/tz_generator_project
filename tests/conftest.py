import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Generator

# Устанавливаем переменную окружения для тестового режима
os.environ['PYTEST_CURRENT_TEST'] = '1'

# Мокируем llama_cpp ДО импорта app, чтобы избежать ошибки импорта
mock_llama_class = MagicMock()
mock_llama_instance = MagicMock()
mock_llama_instance.create_chat_completion.return_value = {
    'choices': [{'message': {'content': 'Mock response'}}]
}
mock_llama_class.return_value = mock_llama_instance

# Создаем мок-модуль для llama_cpp
import sys
from unittest.mock import MagicMock

class MockLlama:
    def __init__(self, *args, **kwargs):
        self.create_chat_completion = MagicMock(return_value={
            'choices': [{'message': {'content': 'Mock response'}}]
        })

# Заменяем модуль llama_cpp на мок
sys.modules['llama_cpp'] = MagicMock()
sys.modules['llama_cpp'].Llama = MockLlama

# Импортируем после мокирования
from app.main import app
from app.core.config import Settings

# Создаем мок модели для использования в тестах
@pytest.fixture(scope='session', autouse=True)
def setup_mock_model():
    """Настройка мок-модели для всех тестов"""
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        'choices': [{'message': {'content': 'Mock response'}}]
    }
    
    # Заменяем глобальную модель на мок
    import app.models.llm_model as llm_module
    mock_model = MagicMock()
    mock_model.generate.return_value = 'Mock generated text'
    mock_model.validate.return_value = {
        'completeness_score': 85,
        'issues': [],
        'suggestions': [],
        'overall_feedback': 'OK'
    }
    
    llm_module.model = mock_model
    
    # Также заменяем в сервисах
    import app.services.tz_generator as gen_module
    import app.services.tz_validator as val_module
    import app.api.routes as routes_module
    
    # Обновляем экземпляры сервисов
    if hasattr(gen_module, 'TZGenerator'):
        # Переинициализируем генератор
        pass
    
    yield mock_model


@pytest.fixture
def temp_dir():
    """Создание временной директории для тестов"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_examples_dir(temp_dir):
    """Создание временной директории для примеров ТЗ"""
    examples_dir = os.path.join(temp_dir, 'examples')
    os.makedirs(examples_dir, exist_ok=True)
    return examples_dir


@pytest.fixture
def temp_output_dir(temp_dir):
    """Создание временной директории для выходных файлов"""
    output_dir = os.path.join(temp_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_tz_text():
    """Пример текста ТЗ для тестов"""
    return """# Техническое задание

## 1. Введение
Проект представляет собой веб-приложение для управления задачами.

## 2. Функциональные требования
- Создание задач
- Редактирование задач
- Удаление задач

## 3. Нефункциональные требования
- Время отклика < 200ms
- Поддержка 1000+ пользователей
"""


@pytest.fixture
def sample_tz_markdown():
    """Пример markdown текста для тестов"""
    return """# Заголовок 1

## Заголовок 2

### Заголовок 3

Обычный параграф текста.

- Маркированный список 1
- Маркированный список 2

1. Нумерованный список 1
2. Нумерованный список 2
"""


@pytest.fixture
def mock_llm_model():
    """Мок LLM модели для тестов"""
    mock_model = MagicMock()
    
    # Мок метода generate
    def mock_generate(prompt, **kwargs):
        if 'validation' in prompt.lower() or 'анализ' in prompt.lower():
            return '''{
                "completeness_score": 85,
                "issues": [
                    {"type": "missing_section", "description": "Отсутствует раздел глоссария", "severity": "low"}
                ],
                "suggestions": ["Добавьте глоссарий терминов"],
                "overall_feedback": "ТЗ достаточно полное, но можно улучшить"
            }'''
        return '# Техническое задание\n\n## 1. Введение\nПроект для тестирования.\n\n## 2. Требования\n- Требование 1\n- Требование 2'
    
    mock_model.generate = Mock(side_effect=mock_generate)
    
    return mock_model


@pytest.fixture
def mock_llm_instance(mock_llm_model):
    """Патч для глобального экземпляра модели"""
    with patch('app.models.llm_model.model', mock_llm_model):
        with patch('app.services.tz_generator.model', mock_llm_model):
            with patch('app.services.tz_validator.model', mock_llm_model):
                yield mock_llm_model


@pytest.fixture
def test_settings(temp_dir, temp_examples_dir, temp_output_dir):
    """Тестовые настройки"""
    return Settings(
        MODEL_PATH='./data/model/GigaChat3-10B-A1.8B-q6_k.gguf',
        EXAMPLES_DIR=temp_examples_dir,
        TEMPLATE_PATH=os.path.join(temp_dir, 'template.docx'),
        OUTPUT_DIR=temp_output_dir,
        N_GPU_LAYERS=0,
        CONTEXT_SIZE=2048,
        MAX_TOKENS=500,
        TEMPERATURE=0.3,
        GENERATION_TIMEOUT=60
    )


@pytest.fixture
def test_client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def sample_tz_file(temp_examples_dir, sample_tz_text):
    """Создание тестового файла с примером ТЗ"""
    file_path = os.path.join(temp_examples_dir, 'sample_tz.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_tz_text)
    return file_path


@pytest.fixture
def sample_docx_file(temp_dir):
    """Создание тестового .docx файла"""
    from docx import Document
    
    file_path = os.path.join(temp_dir, 'sample_tz.docx')
    doc = Document()
    doc.add_heading('Техническое задание', 0)
    doc.add_paragraph('Пример ТЗ для тестирования')
    doc.add_heading('Требования', 1)
    doc.add_paragraph('Требование 1', style='List Bullet')
    doc.add_paragraph('Требование 2', style='List Bullet')
    doc.save(file_path)
    return file_path


@pytest.fixture
def docx_template_file(temp_dir):
    """Создание тестового шаблона .docx"""
    from docx import Document
    from docx.shared import Pt
    from docx.enum.style import WD_STYLE_TYPE
    
    file_path = os.path.join(temp_dir, 'template.docx')
    doc = Document()
    
    # Создание стилей
    styles = doc.styles
    heading1 = styles.add_style('TZTitle', WD_STYLE_TYPE.PARAGRAPH)
    heading1.font.name = 'Times New Roman'
    heading1.font.size = Pt(18)
    heading1.font.bold = True
    
    doc.save(file_path)
    return file_path
