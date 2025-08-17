import os
import json 
from typing import List, Dict, Union, Any, Optional, Callable, Awaitable 

from interfaces.llm import AbstractLLM
from interfaces.memory import Memory
from interfaces.vector_store import VectorStore
from interfaces.tool import Tool
from core.tool_manager import ToolManager

class AgentContext:
    def __init__(self, 
                 text_input: Optional[str] = None, 
                 file_input_path: Optional[str] = None):
        self.text_input: Optional[str] = text_input
        self.file_input_path: Optional[str] = file_input_path
        self.file_content: Optional[str] = None 
        self.rag_context: List[str] = [] 
        self.llm_response_raw: Union[str, Dict[str, Any], None] = None 
        self.tool_calls: List[Dict[str, Any]] = [] 
        self.tool_outputs: List[Dict[str, Any]] = [] 
        self.final_response: Optional[str] = None 
        self.error_message: Optional[str] = None 
        self.processed_successfully: bool = False 
        self.current_prompt_for_llm: str = "" 
        self.metadata: Dict[str, Any] = {} 

AgentProcessor = Callable[['AIAgent', AgentContext], Awaitable[None]]

class AIAgent:
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
        
        self._pre_llm_processors: List[AgentProcessor] = []
        self._post_llm_processors: List[AgentProcessor] = []
        self._post_tool_execution_processors: List[AgentProcessor] = []
        self._final_response_processors: List[AgentProcessor] = []

        print("AI Агент инициализирован.")
        if self._system_prompt:
            print(f"Агент инициализирован с системным промптом: '{self._system_prompt[:50]}...'")

    def register_tool(self, tool: Tool):
        self.tool_manager.register_tool(tool)

    def register_processor(self, stage: str, processor: AgentProcessor):
        if stage == 'pre_llm':
            self._pre_llm_processors.append(processor)
        elif stage == 'post_llm':
            self._post_llm_processors.append(processor)
        elif stage == 'post_tool_execution':
            self._post_tool_execution_processors.append(processor)
        elif stage == 'final_response':
            self._final_response_processors.append(processor)
        else:
            raise ValueError(f"Неизвестный этап обработки: {stage}")
        print(f"Зарегистрирован процессор для этапа '{stage}'.")

    def _format_tools_for_llm(self) -> List[Dict[str, Any]]:
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

    async def _run_processors(self, processors: List[AgentProcessor], context: AgentContext):
        for processor in processors:
            await processor(self, context)

    async def process_message(self, text_input: Optional[str] = None, file_input: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        context = AgentContext(text_input=text_input, file_input_path=file_input)
        
        if context.file_input_path:
            print(f"\nАгент получил файл: '{context.file_input_path}'")
            try:
                with open(context.file_input_path, 'r', encoding='utf-8') as f:
                    context.file_content = f.read()
                context.text_input = (
                    f"Проанализируй следующий документ:\n\n"
                    f"```document\n{context.file_content[:1000]}...\n```\n\n"
                    f"Кратко summarize его ключевые моменты и предложи вопросы, которые можно задать по этому документу."
                )
                self.memory.add_message(role="user", content=context.text_input) 
                context.current_prompt_for_llm = context.text_input 
            except Exception as e:
                context.error_message = f"Ошибка при обработке файла '{context.file_input_path}': {e}"
                self.memory.add_message(role="system_error", content=context.error_message)
                context.final_response = {"status": "error", "message": context.error_message}
                context.processed_successfully = False
                return context.final_response
        elif context.text_input:
            self.memory.add_message(role="user", content=context.text_input)
            context.current_prompt_for_llm = context.text_input
        else:
            context.final_response = "Не получено никакого ввода (текста или файла)."
            context.processed_successfully = False
            return context.final_response

        for i in range(self.max_tool_iterations):
            print(f"\n--- Итерация {i+1} агента ---")
            
            await self._run_processors(self._pre_llm_processors, context)

            retrieved_context = []
            if self.vector_store and context.current_prompt_for_llm: 
                # <--- ИЗМЕНЕНИЕ ЗДЕСЬ: ДОБАВИТЬ await
                retrieved_docs = await self.vector_store.similarity_search(context.current_prompt_for_llm, k=3) 
                if retrieved_docs:
                    context.rag_context = [f"Контекст из памяти: {doc}" for doc in retrieved_docs]
                    print(f"Извлечен контекст из векторного хранилища: {context.rag_context}")
            
            combined_system_prompt = self._system_prompt
            if context.rag_context:
                combined_system_prompt += "\n\n" + "\n".join(context.rag_context)

            try:
                conversation_history = self.memory.get_history()
                available_tools_for_llm = self._format_tools_for_llm()

                context.llm_response_raw = await self.llm.generate_response( 
                    prompt=context.current_prompt_for_llm,
                    system_prompt=combined_system_prompt,
                    history=conversation_history,
                    tools=available_tools_for_llm
                )
                
                await self._run_processors(self._post_llm_processors, context)

                if isinstance(context.llm_response_raw, dict) and "tool_calls" in context.llm_response_raw:
                    context.tool_calls = context.llm_response_raw["tool_calls"]
                    
                    self.memory.add_message(role="assistant", content=json.dumps({"tool_calls": context.tool_calls}))
                    
                    for tool_call in context.tool_calls:
                        tool_id = tool_call["id"] 
                        tool_name = tool_call["name"]
                        tool_args = tool_call["arguments"]
                        
                        try:
                            print(f"LLM запросила выполнение инструмента '{tool_name}' с аргументами: {tool_args}")
                            tool_output = self.tool_manager.execute_tool(tool_name, **tool_args)
                            context.tool_outputs.append({"tool_call_id": tool_id, "output": tool_output})
                            self.memory.add_message(role="tool", content=json.dumps({"tool_call_id": tool_id, "output": tool_output}))
                            
                        except Exception as e:
                            error_msg = f"Ошибка при выполнении инструмента '{tool_name}': {e}. Аргументы: {tool_args}"
                            self.memory.add_message(role="system_error", content=error_msg)
                            print(error_msg)
                            context.tool_outputs.append({"tool_call_id": tool_id, "output": f"Ошибка выполнения: {error_msg}"})
                            self.memory.add_message(role="tool", content=json.dumps({"tool_call_id": tool_id, "output": f"Ошибка выполнения: {error_msg}"}))
                    
                    await self._run_processors(self._post_tool_execution_processors, context)

                    context.current_prompt_for_llm = "" 
                    
                else:
                    context.final_response = context.llm_response_raw
                    context.processed_successfully = True
                    break 
                        
            except Exception as e:
                context.error_message = f"Ошибка при вызове LLM или парсинге ответа: {e}"
                context.final_response = f"Извините, произошла внутренняя ошибка при обработке запроса: {e}"
                self.memory.add_message(role="system_error", content=context.error_message)
                context.processed_successfully = False
                break 

        if not context.final_response:
            context.final_response = "Я выполнил некоторые действия с инструментами, но пока не могу сформулировать окончательный ответ. Попробуйте уточнить запрос."
            context.processed_successfully = False
        
        await self._run_processors(self._final_response_processors, context)
        
        self.memory.add_message(role="assistant", content=context.final_response)
        return context.final_response