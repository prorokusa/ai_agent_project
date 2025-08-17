import json # Добавлен для имитации JSON tool_calls
from typing import List, Dict, Optional, Any, Union
from interfaces.llm import AbstractLLM

class SimpleInferenceLLM(AbstractLLM):
    """
    Пример простой LLM для демонстрации. Не поддерживает Tool Calling напрямую,
    возвращает фиктивный ответ, если предполагается вызов инструмента.
    """
    def __init__(self, model_name: str = "dummy-model"):
        self.model_name = model_name
        print(f"Инициализирован SimpleInferenceLLM с моделью: {self.model_name}")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Union[str, Dict[str, Any]]:
        print(f"\n--- LLM ({self.model_name}) запрос ---")
        if system_prompt:
            print(f"Системный промпт: {system_prompt}")
        if history:
            print("История чата:")
            for msg in history:
                print(f"  {msg['role']}: {msg['content']}")
        print(f"Текущий промпт: {prompt}")
        if tools:
            print(f"Доступные инструменты (фиктивно): {', '.join([t['function']['name'] for t in tools])}")
        print(f"Параметры LLM: {kwargs}")
        print("---------------------------------")

        # Имитация ответа LLM
        # В этой фиктивной LLM нет реальной логики Function Calling,
        # поэтому она всегда возвращает текстовый ответ или имитирует вызов инструмента
        
        # Если промпт явно говорит об инструменте, имитируем его использование
        if ("вычисли" in prompt.lower() or "посчитай" in prompt.lower()) and tools and any(t['function']['name'] == 'calculator' for t in tools):
            # Имитируем, что LLM могла бы вызвать инструмент
            print(f"[SimpleInferenceLLM]: Имитирую вызов инструмента 'calculator' с выражением '12+34'")
            return {
                "tool_calls": [{
                    "id": "call_calc_dummy", # Фиктивный ID
                    "function": { # В OpenAI API tool_calls содержит 'function' объект
                        "name": "calculator",
                        "arguments": json.dumps({"expression": "12+34"}) # Аргументы должны быть JSON-строкой
                    }
                }]
            }
        elif ("найди" in prompt.lower() or "поищи" in prompt.lower()) and tools and any(t['function']['name'] == 'google_cse_search' for t in tools): # <--- ИЗМЕНЕНИЕ ЗДЕСЬ
            print(f"[SimpleInferenceLLM]: Имитирую вызов инструмента 'google_cse_search' с запросом '{prompt}'")
            return {
                "tool_calls": [{
                    "id": "call_search_dummy", # Фиктивный ID
                    "function": {
                        "name": "google_cse_search", # <--- ИЗМЕНЕНИЕ ЗДЕСЬ
                        "arguments": json.dumps({"query": prompt.replace("найди", "").replace("поищи", "").strip()})
                    }
                }]
            }
        
        return f"Это ответ от {self.model_name} на ваш запрос: '{prompt}'. (Параметры: {kwargs}). " \
               f"Мой системный промпт был: '{system_prompt[:30]}...' (если был)."

    def get_embedding(self, text: str) -> List[float]:
        # Простая имитация эмбеддинга (например, сумма ASCII значений символов)
        # В реальном приложении здесь будет вызов API для получения эмбеддинга
        return [float(sum(ord(c) for c in text)) % 100 / 100.0] * 128 # Возвращаем список из 128 float для примера