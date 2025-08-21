import os
import json
import itertools 
from typing import List, Dict, Optional, Any, Union
from interfaces.llm import AbstractLLM
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam
)
from openai.types.chat.chat_completion_message_tool_call import Function
from dotenv import load_dotenv

load_dotenv()

# --- НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ САНИТАРИЗАЦИИ СТРОК ---
def _sanitize_string(s: Any) -> str:
    """
    Преобразует входные данные в строку и удаляет или заменяет символы, которые
    могут вызывать ошибки кодировки UTF-8 (например, одиночные суррогаты).
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        # Если вход не строка, пытаемся преобразовать его в строку.
        # Это может быть полезно для нестроковых данных, которые случайно попадают сюда.
        s = str(s)
    
    # Кодируем в UTF-8, заменяя проблемные символы на '�' (U+FFFD),
    # затем декодируем обратно. Это "очищает" строку.
    return s.encode('utf-8', 'replace').decode('utf-8')

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
            self.embedding_client = None
        else:
            self.embedding_client = OpenAI(api_key=openai_api_key)
        
        print(f"Инициализирован OpenRouter_LLM с моделью: {self.model_name} и {len(self.api_keys)} ключами OpenRouter.")

    def _get_current_client(self) -> AsyncOpenAI: 
        """Возвращает новый клиент AsyncOpenAI с текущим ключом из ротации."""
        current_key = next(self._api_key_iterator)
        return AsyncOpenAI(base_url=self.base_url, api_key=current_key) 

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Union[str, Dict[str, Any]]:
        messages: List[ChatCompletionMessageParam] = []

        if system_prompt:
            # Применяем санитаризацию к system_prompt
            messages.append({"role": "system", "content": _sanitize_string(system_prompt)})
        
        if history:
            for msg in history:
                if msg["role"] == "tool":
                    try:
                        tool_data = json.loads(msg["content"])
                        # Санитаризуем вывод инструмента перед добавлением в сообщение
                        sanitized_output = _sanitize_string(tool_data.get("output", ""))
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_data["tool_call_id"],
                            "content": sanitized_output
                        })
                    except (json.JSONDecodeError, KeyError):
                        # Санитаризуем содержимое, если оно не является корректным JSON
                        messages.append({"role": "tool", "content": _sanitize_string(msg['content']), "tool_call_id": "unknown_id_from_memory"})
                elif msg["role"] == "assistant":
                    try:
                        # Важно: если контент - это JSON-строка, содержащая tool_calls,
                        # мы должны сначала распарсить ее, санитаризовать аргументы, а затем восстановить.
                        # Если это простой текст, то санитаризуем его напрямую.
                        assistant_content = json.loads(msg["content"])
                        if "tool_calls" in assistant_content:
                            openai_tool_calls: List[ChatCompletionMessageToolCall] = []
                            for tc_info in assistant_content["tool_calls"]:
                                # Санитаризуем аргументы инструмента
                                # Сначала убедимся, что arguments - это строка, затем санитаризуем
                                args_str = json.dumps(tc_info.get('arguments', {})) # Safely get arguments as JSON string
                                sanitized_args = _sanitize_string(args_str)
                                func = Function(name=tc_info['name'], arguments=sanitized_args)
                                openai_tool_calls.append(ChatCompletionMessageToolCall(id=tc_info['id'], function=func, type='function'))
                            
                            messages.append({"role": "assistant", "tool_calls": openai_tool_calls})
                        else:
                            # Санитаризуем обычный текстовый ответ ассистента
                            messages.append({"role": "assistant", "content": _sanitize_string(msg["content"])})
                    except (json.JSONDecodeError, KeyError):
                        # Санитаризуем содержимое, если оно не является корректным JSON или не имеет 'tool_calls'
                        messages.append({"role": "assistant", "content": _sanitize_string(msg["content"])})
                elif msg["role"] == "user": 
                    # Санитаризуем пользовательский ввод
                    messages.append({"role": "user", "content": _sanitize_string(msg["content"])})

        # Санитаризуем текущий промпт
        messages.append({"role": "user", "content": _sanitize_string(prompt)})
        
        openai_tools: Optional[List[ChatCompletionToolParam]] = None
        if tools:
            openai_tools = []
            for tool_def in tools:
                if tool_def["type"] == "function":
                    # Санитаризуем описание функции, если оно есть
                    sanitized_description = _sanitize_string(tool_def["function"].get("description", ""))
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_def["function"]["name"],
                            "description": sanitized_description,
                            "parameters": tool_def["function"]["parameters"] # Параметры обычно уже в корректном JSON формате
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
                            "arguments": json.loads(tc.function.arguments) # Аргументы должны быть уже чистыми после предыдущей санитаризации
                        } for tc in tool_calls
                    ]
                }
            else:
                return _sanitize_string(response.choices[0].message.content) # Санитаризуем ответ LLM
        except Exception as e:
            print(f"Ошибка при вызове OpenRouter LLM: {e}")
            return f"Извините, произошла ошибка при генерации ответа: {_sanitize_string(str(e))}"

    def get_embedding(self, text: str) -> List[float]:
        if not self.embedding_client:
            print("Предупреждение: Клиент для эмбеддингов не инициализирован (OPENAI_API_KEY отсутствует).")
            return []
        
        try:
            # Применяем санитаризацию к тексту перед генерацией эмбеддинга
            sanitized_text = _sanitize_string(text)
            response = self.embedding_client.embeddings.create(
                input=[sanitized_text],
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Ошибка при генерации эмбеддинга через OpenAI для OpenRouter_LLM: {e}")
            return []