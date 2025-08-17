import json 
from typing import List, Dict, Optional, Any, Union
from interfaces.llm import AbstractLLM

class SimpleInferenceLLM(AbstractLLM):
    """
    Пример простой LLM для демонстрации.
    """
    def __init__(self, model_name: str = "dummy-model"):
        self.model_name = model_name
        print(f"Инициализирован SimpleInferenceLLM с моделью: {self.model_name}")

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Union[str, Dict[str, Any]]:
        # ... (код generate_response остался асинхронным)
        
        if ("вычисли" in prompt.lower() or "посчитай" in prompt.lower()) and tools and any(t['function']['name'] == 'calculator' for t in tools):
            print(f"[SimpleInferenceLLM]: Имитирую вызов инструмента 'calculator' с выражением '12+34'")
            return {
                "tool_calls": [{
                    "id": "call_calc_dummy", 
                    "function": { 
                        "name": "calculator",
                        "arguments": json.dumps({"expression": "12+34"}) 
                    }
                }]
            }
        elif ("найди" in prompt.lower() or "поищи" in prompt.lower()) and tools and any(t['function']['name'] == 'google_cse_search' for t in tools): 
            print(f"[SimpleInferenceLLM]: Имитирую вызов инструмента 'google_cse_search' с запросом '{prompt}'")
            return {
                "tool_calls": [{
                    "id": "call_search_dummy", 
                    "function": {
                        "name": "google_cse_search", 
                        "arguments": json.dumps({"query": prompt.replace("найди", "").replace("поищи", "").strip()})
                    }
                }]
            }
        
        return f"Это ответ от {self.model_name} на ваш запрос: '{prompt}'. (Параметры: {kwargs}). " \
               f"Мой системный промпт был: '{system_prompt[:30]}...' (если был)."

    def get_embedding(self, text: str) -> List[float]: # <--- ИЗМЕНЕНИЕ: Метод get_embedding теперь синхронный
        return [float(sum(ord(c) for c in text)) % 100 / 100.0] * 128 