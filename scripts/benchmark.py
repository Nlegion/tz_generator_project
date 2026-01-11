import time
from app.models.llm_model import model

def benchmark_generation():
    """Тестирование скорости генерации"""
    test_prompts = [
        "Создай ТЗ для простого веб-сайта",
        "Создай ТЗ для мобильного приложения заметок",
        "Создай ТЗ для системы аналитики"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"Тест {i+1}: {prompt[:50]}...")
        
        start_time = time.time()
        result = model.generate(prompt, max_tokens=500)
        end_time = time.time()
        
        generation_time = end_time - start_time
        token_count = len(result.split())
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0
        
        results.append({
            "prompt": prompt,
            "time_seconds": round(generation_time, 2),
            "tokens": token_count,
            "tokens_per_second": round(tokens_per_second, 2)
        })
    
    # Вывод результатов
    print("\nРезультаты тестирования:")
    for r in results:
        print(f"  {r['prompt'][:30]}...: {r['time_seconds']}s, {r['tokens_per_second']} t/s")

if __name__ == "__main__":
    benchmark_generation()