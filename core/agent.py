import os
import json 
from typing import List, Dict, Union, Any, Optional

from interfaces.llm import AbstractLLM
from interfaces.memory import Memory
from interfaces.vector_store import VectorStore
from interfaces.tool import Tool
from core.tool_manager import ToolManager

class AIAgent:
    """
    Основной класс AI Агента, оркеструющий LLM, инструменты, память и векторное хранилище.
    Теперь включает более сложную логику для выбора и выполнения инструментов
    с помощью функционала LLM Tool Calling (если LLM его поддерживает).
    """
    def __init__(self, 
                 llm: AbstractLLM, 
                 memory: Memory,  
                 vector_store: Optional[VectorStore] = None,
                 system_prompt: Optional[str] = None,
                 max_tool_iterations: int = 3): 
        
        self.llm = llm
        self.tool_manager = ToolManager()
        self.memory = memory
        self.vector_store = vector_store
        self._system_prompt = system_prompt
        self.max_tool_iterations = max_tool_iterations

        print("AI Агент инициализирован.")
        if self._system_prompt:
            print(f"Агент инициализирован с системным промптом: '{self._system_prompt[:50]}...'")

    def register_tool(self, tool: Tool):
        """Регистрирует инструмент через менеджер инструментов."""
        self.tool_manager.register_tool(tool)

    def _format_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Форматирует описания зарегистрированных инструментов в формат,
        понятный для LLM Function Calling (например, OpenAI tools).
        Важно: Здесь должны быть точные схемы JSON Schema для параметров!
        """
        formatted_tools = []
        for tool_info in self.tool_manager.list_tools():
            tool_name = tool_info["name"]
            tool_description = tool_info["description"]
            
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }

            if tool_name == "calculator":
                parameters["properties"] = {"expression": {"type": "string", "description": "Математическое выражение для вычисления."}}
                parameters["required"] = ["expression"]
            elif tool_name == "google_cse_search": 
                parameters["properties"] = {"query": {"type": "string", "description": "Поисковый запрос для интернета."}}
                parameters["required"] = ["query"]

            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": parameters
                }
            })
        return formatted_tools

    def process_message(self, text_input: Optional[str] = None, file_input: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Обрабатывает входное сообщение (текст или файл) и генерирует ответ.
        Использует LLM для принятия решений о вызове инструментов и RAG.
        Args:
            text_input (Optional[str]): Входной текст.
            file_input (Optional[str]): Путь к входному файлу.
        Returns:
            Union[str, Dict[str, Any]]: Текстовый ответ или словарь с информацией о файле.
        """
        if text_input:
            print(f"\nАгент получил текстовое сообщение: '{text_input}'")
            self.memory.add_message(role="user", content=text_input)
            
            current_llm_input_prompt = text_input 
            
            for i in range(self.max_tool_iterations):
                print(f"\n--- Итерация {i+1} агента ---")
                
                retrieved_context = []
                if self.vector_store:
                    if current_llm_input_prompt and not current_llm_input_prompt.startswith("Продолжи диалог"):
                        retrieved_docs = self.vector_store.similarity_search(current_llm_input_prompt, k=3)
                        if retrieved_docs:
                            retrieved_context = [f"Контекст из памяти: {doc}" for doc in retrieved_docs]
                            print(f"Извлечен контекст из векторного хранилища: {retrieved_context}")
                
                combined_system_prompt = self._system_prompt
                if retrieved_context:
                    combined_system_prompt += "\n\n" + "\n".join(retrieved_context)

                try:
                    conversation_history = self.memory.get_history()
                    available_tools_for_llm = self._format_tools_for_llm()

                    llm_response = self.llm.generate_response(
                        prompt=current_llm_input_prompt,
                        system_prompt=combined_system_prompt,
                        history=conversation_history,
                        tools=available_tools_for_llm 
                    )
                    
                    if isinstance(llm_response, dict) and "tool_calls" in llm_response:
                        tool_calls_info = llm_response["tool_calls"]
                        
                        self.memory.add_message(role="assistant", content=json.dumps({"tool_calls": tool_calls_info}))
                        
                        for tool_call in tool_calls_info:
                            tool_id = tool_call["id"] 
                            tool_name = tool_call["name"]
                            tool_args = tool_call["arguments"]
                            
                            try:
                                print(f"LLM запросила выполнение инструмента '{tool_name}' с аргументами: {tool_args}")
                                tool_output = self.tool_manager.execute_tool(tool_name, **tool_args)
                                self.memory.add_message(role="tool", content=json.dumps({"tool_call_id": tool_id, "output": tool_output}))
                                
                            except ValueError as e:
                                error_msg = f"Ошибка: {e}. Инструмент '{tool_name}' не найден или аргументы неверны: {tool_args}"
                                self.memory.add_message(role="system_error", content=error_msg)
                                print(error_msg)
                                self.memory.add_message(role="tool", content=json.dumps({"tool_call_id": tool_id, "output": f"Ошибка выполнения: {error_msg}"}))
                            except Exception as e:
                                error_msg = f"Непредвиденная ошибка при выполнении инструмента '{tool_name}': {e}"
                                self.memory.add_message(role="system_error", content=error_msg)
                                print(error_msg)
                                self.memory.add_message(role="tool", content=json.dumps({"tool_call_id": tool_id, "output": f"Ошибка выполнения: {error_msg}"}))
                        
                        current_llm_input_prompt = "" 
                        
                    else:
                        response_content = llm_response
                        break 
                        
                except Exception as e:
                    print(f"Ошибка при вызове LLM или парсинге ответа: {e}")
                    response_content = f"Извините, произошла внутренняя ошибка при обработке запроса: {e}"
                    self.memory.add_message(role="system_error", content=response_content)
                    break 

            if not response_content:
                response_content = "Я выполнил некоторые действия с инструментами, но пока не могу сформулировать окончательный ответ. Попробуйте уточнить запрос."
            
            self.memory.add_message(role="assistant", content=response_content)
            return response_content
        
        elif file_input:
            print(f"\nАгент получил файл: '{file_input}'")
            file_content = ""
            try:
                with open(file_input, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                processed_content = f"Содержимое файла '{os.path.basename(file_input)}':\n{file_content[:1000]}..." 
                
                # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
                # Вместо добавления в память с ролью 'file_content',
                # мы сразу формируем промпт для LLM, чтобы она проанализировала файл
                file_analysis_prompt = (
                    f"Проанализируй следующий документ:\n\n"
                    f"```document\n{file_content}\n```\n\n" # Оборачиваем в маркдаун для читабельности LLM
                    f"Кратко summarize его ключевые моменты и предложи вопросы, которые можно задать по этому документу."
                )
                
                # Мы передаем этот промпт LLM как обычное сообщение пользователя,
                # и оно будет добавлено в историю с ролью 'user'.
                self.memory.add_message(role="user", content=file_analysis_prompt)

                response_from_llm = self.llm.generate_response(
                    prompt=file_analysis_prompt, # Передаем сформированный промпт
                    system_prompt=self._system_prompt, 
                    history=self.memory.get_history()
                )
                
                self.memory.add_message(role="assistant", content=response_from_llm)
                return {"status": "file_processed", "filename": os.path.basename(file_input), "response": response_from_llm}
            except Exception as e:
                error_message = f"Ошибка при обработке файла '{file_input}': {e}"
                self.memory.add_message(role="system_error", content=error_message)
                return {"status": "error", "message": error_message}
        else:
            return "Не получено никакого ввода (текста или файла)."