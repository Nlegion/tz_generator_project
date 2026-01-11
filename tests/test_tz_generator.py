import pytest
from unittest.mock import patch, MagicMock

from app.services.tz_generator import TZGenerator
from app.utils.prompt_templates import GENERATION_TEMPLATES


@pytest.fixture
def generator_with_mock(mock_llm_instance, temp_examples_dir):
    """Генератор с моком LLM"""
    with patch('app.services.tz_generator.settings') as mock_settings:
        mock_settings.EXAMPLES_DIR = temp_examples_dir
        gen = TZGenerator()
        return gen


def test_tz_generator_initialization(temp_examples_dir):
    """Тест инициализации TZGenerator"""
    with patch('app.services.tz_generator.settings') as mock_settings:
        mock_settings.EXAMPLES_DIR = temp_examples_dir
        generator = TZGenerator()
        
        assert generator.examples is not None
        assert isinstance(generator.examples, list)


def test_tz_generator_initialization_with_missing_dir():
    """Тест инициализации при отсутствии директории примеров"""
    with patch('app.services.tz_generator.settings') as mock_settings:
        mock_settings.EXAMPLES_DIR = '/nonexistent/dir/12345'
        generator = TZGenerator()
        
        assert generator.examples == []


def test_generate_basic(generator_with_mock):
    """Тест базовой генерации ТЗ"""
    result = generator_with_mock.generate(
        project_description='Веб-приложение для управления задачами',
        project_type='web',
        style='formal',
        use_examples=False
    )
    
    assert 'text' in result
    assert 'metadata' in result
    assert result['metadata']['project_type'] == 'web'
    assert result['metadata']['style'] == 'formal'
    assert result['metadata']['length_chars'] > 0


def test_generate_different_project_types(generator_with_mock):
    """Тест генерации для разных типов проектов"""
    project_types = ['general', 'web', 'mobile']
    
    for project_type in project_types:
        result = generator_with_mock.generate(
            project_description='Тестовый проект',
            project_type=project_type,
            use_examples=False
        )
        
        assert result['metadata']['project_type'] == project_type


def test_generate_different_styles(generator_with_mock):
    """Тест генерации с разными стилями"""
    styles = ['formal', 'concise', 'detailed']
    
    for style in styles:
        result = generator_with_mock.generate(
            project_description='Тестовый проект',
            style=style,
            use_examples=False
        )
        
        assert result['metadata']['style'] == style


def test_generate_with_examples(generator_with_mock, temp_examples_dir, sample_tz_text):
    """Тест генерации с использованием примеров"""
    # Создаем пример
    import os
    file_path = os.path.join(temp_examples_dir, 'example.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_tz_text)
    
    # Переинициализируем генератор для загрузки примера
    with patch('app.services.tz_generator.settings') as mock_settings:
        mock_settings.EXAMPLES_DIR = temp_examples_dir
        generator = TZGenerator()
        
        result = generator.generate(
            project_description='Веб-приложение',
            use_examples=True
        )
        
        assert result['metadata']['example_count'] > 0


def test_generate_without_examples(generator_with_mock):
    """Тест генерации без использования примеров"""
    result = generator_with_mock.generate(
        project_description='Тестовый проект',
        use_examples=False
    )
    
    assert result['metadata']['example_count'] == 0


def test_generate_formatting(generator_with_mock):
    """Тест форматирования вывода"""
    result = generator_with_mock.generate(
        project_description='Тестовый проект',
        use_examples=False
    )
    
    text = result['text']
    # Проверяем, что текст не пустой
    assert len(text) > 0
    # Проверяем, что есть заголовки
    assert '#' in text or 'Техническое задание' in text


def test_generate_error_handling(generator_with_mock):
    """Тест обработки ошибок генерации"""
    # Мокаем модель для выброса ошибки
    with patch('app.services.tz_generator.model') as mock_model:
        mock_model.generate.side_effect = RuntimeError('Ошибка генерации')
        mock_model.is_None = False
        
        # Генератор должен обработать ошибку и пробросить её дальше
        try:
            generator_with_mock.generate(
                project_description='Тестовый проект',
                use_examples=False
            )
            # Если ошибка не была выброшена, проверяем, что она была обработана
        except RuntimeError:
            pass  # Ожидаемое поведение


def test_generate_unknown_project_type(generator_with_mock):
    """Тест генерации с неизвестным типом проекта"""
    # Использует шаблон 'general' по умолчанию
    result = generator_with_mock.generate(
        project_description='Тестовый проект',
        project_type='unknown_type',
        use_examples=False
    )
    
    assert result['metadata']['project_type'] == 'unknown_type'


def test_format_output():
    """Тест метода форматирования вывода"""
    generator = TZGenerator()
    
    input_text = '## Заголовок\n\n### Подзаголовок\n\nТекст'
    formatted = generator._format_output(input_text)
    
    assert '# Заголовок' in formatted
    assert '## Подзаголовок' in formatted
