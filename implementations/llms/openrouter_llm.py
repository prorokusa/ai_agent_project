import os
import json
import itertools 
from typing import List, Dict, Optional, Any, Union
from interfaces.llm import AbstractLLM
from openai import OpenAI, AsyncOpenAI # <--- ИЗМЕНЕНИЕ: Импортируем оба клиента
from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam
)
from openai.types.chat.chat_completion_message_tool_call import Function
from dotenv import load_dotenv

load_dotenv()

class OpenRouter_LLM(AbstractLLM):
    """
    Реализация LLM для OpenRouter API с ротацией ключей.
    Использует библиотеку OpenAI SDK, изменяя base_url.
    """
    def __init__(self, model_name: str = "google/gemini-2.0-flash-exp:free", api_keys: Optional[Union[str, List[str]]] = None):
        self.model_name = model_name
        
        if isinstance(api_keys, str):
            self.api_keys = [k.strip() for k in api_keys.split(',')]
        elif isinstance(api_keys, list):
            self.api_keys = api_keys
        else:
            env_keys_str = os.getenv("OPENROUTER_API_KEYS")
            if env_keys_str:
                self.api_keys = [k.strip() for k in env_keys_str.split(',')]
            else:
                raise ValueError("OPENROUTER_API_KEYS не установлен. Пожалуйста, установите его как переменную окружения (разделитель - запятая) или передайте в конструктор.")
        
        if not self.api_keys:
            raise ValueError("Список ключей OpenRouter API пуст.")
        
        self._api_key_iterator = itertools.cycle(self.api_keys)
        self.base_url = "https://openrouter.ai/api/v1/"
        
        # Для generate_response (async)
        # Клиент AsyncOpenAI будет создаваться для каждого запроса с новым ключом.

        # Для get_embedding (sync)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Предупреждение: OPENAI_API_KEY не установлен. Эмбеддинги не могут быть сгенерированы.")
            self.embedding_client = None # <--- ИЗМЕНЕНИЕ
        else:
            self.embedding_client = OpenAI(api_key=openai_api_key) # <--- ИЗМЕНЕНИЕ: Синхронный клиент для эмбеддингов
        
        print(f"Инициализирован OpenRouter_LLM с моделью: {self.model_name} и {len(self.api_keys)} ключами OpenRouter.")

    def _get_current_client(self) -> AsyncOpenAI: 
        """Возвращает новый клиент AsyncOpenAI с текущим ключом из ротации."""
        current_key = next(self._api_key_iterator)
        return AsyncOpenAI(base_url=self.base_url, api_key=current_key) 

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Union[str, Dict[str, Any]]:
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
                elif msg["role"] == "user": 
                    messages.append({"role": "user", "content": msg["content"]})

        messages.append({"role": "user", "content": prompt})
        
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
            client = self._get_current_client() 
            response = await client.chat.completions.create(**api_params) 
            
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
            print(f"Ошибка при вызове OpenRouter LLM: {e}")
            return f"Извините, произошла ошибка при генерации ответа: {e}"

    def get_embedding(self, text: str) -> List[float]: # <--- ИЗМЕНЕНИЕ: Метод get_embedding теперь синхронный
        if not self.embedding_client:
            print("Предупреждение: Клиент для эмбеддингов не инициализирован (OPENAI_API_KEY отсутствует).")
            return []
        
        try:
            response = self.embedding_client.embeddings.create( # <--- ИЗМЕНЕНИЕ: Используем sync embedding_client
                input=[text],
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Ошибка при генерации эмбеддинга через OpenAI для OpenRouter_LLM: {e}")
            return []