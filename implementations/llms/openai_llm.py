import os
import json 
from typing import List, Dict, Optional, Any, Union
from interfaces.llm import AbstractLLM
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageParam, 
    ChatCompletionToolParam 
)
from openai.types.chat.chat_completion_message_tool_call import Function 
from dotenv import load_dotenv

load_dotenv() 

class OpenAI_LLM(AbstractLLM):
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY не установлен. Пожалуйста, установите его как переменную окружения или передайте в конструктор.")
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        print(f"Инициализирован OpenAI_LLM с моделью: {self.model_name}")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Union[str, Dict[str, Any]]:
        messages: List[ChatCompletionMessageParam] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            for msg in history:
                if msg["role"] == "tool":
                    try:
                        tool_data = json.loads(msg["content"])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_data["tool_call_id"],
                            "content": json.dumps(tool_data.get("output", "")) 
                        })
                    except (json.JSONDecodeError, KeyError):
                        messages.append({"role": "tool", "content": msg['content'], "tool_call_id": "unknown_id_from_memory"})
                elif msg["role"] == "assistant":
                    try:
                        assistant_content = json.loads(msg["content"])
                        if "tool_calls" in assistant_content:
                            openai_tool_calls: List[ChatCompletionMessageToolCall] = []
                            for tc_info in assistant_content["tool_calls"]:
                                func = Function(name=tc_info['name'], arguments=json.dumps(tc_info['arguments']))
                                openai_tool_calls.append(ChatCompletionMessageToolCall(id=tc_info['id'], function=func, type='function'))
                            
                            messages.append({"role": "assistant", "tool_calls": openai_tool_calls})
                        else:
                            messages.append({"role": "assistant", "content": msg["content"]})
                    except (json.JSONDecodeError, KeyError):
                        messages.append({"role": "assistant", "content": msg["content"]})
                # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
                # Мы больше не добавляем 'file_content' как отдельную роль.
                # Вместо этого, логика в agent.py добавит содержимое файла как user-prompt.
                elif msg["role"] == "user": # Остальные роли, включая 'user', добавляются напрямую
                    messages.append({"role": "user", "content": msg["content"]})
                # Мы также можем игнорировать 'system_error' в history для OpenAI,
                # так как они не являются частью стандартных ролей диалога
                # или обрабатывать их как "system" сообщения, если это применимо к контексту LLM.
                # Для простоты, пока просто не добавляем их.
                # else: 
                #     # Если есть другие роли, которые не должны идти в OpenAI API,
                #     # их можно пропустить или преобразовать.
                #     pass 

        messages.append({"role": "user", "content": prompt}) # Текущий промпт пользователя, который может быть и содержимым файла
        
        openai_tools: Optional[List[ChatCompletionToolParam]] = None
        if tools:
            openai_tools = []
            for tool_def in tools:
                if tool_def["type"] == "function":
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_def["function"]["name"],
                            "description": tool_def["function"]["description"],
                            "parameters": tool_def["function"]["parameters"]
                        }
                    })

        api_params = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }
        
        if openai_tools: 
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**api_params)
            
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                return {
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments)
                        } for tc in tool_calls
                    ]
                }
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(f"Ошибка при вызове OpenAI LLM: {e}")
            return f"Извините, произошла ошибка при генерации ответа: {e}"

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Ошибка при генерации эмбеддинга OpenAI: {e}")
            return []