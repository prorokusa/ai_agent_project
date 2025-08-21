# core/agent.py (ОБНОВЛЕНИЕ)
import os
import json 
from typing import List, Dict, Union, Any, Optional, Callable, Awaitable 
import logging 

from interfaces.llm import AbstractLLM
from interfaces.memory import Memory
from interfaces.vector_store import VectorStore
from interfaces.tool import Tool
from core.tool_manager import ToolManager

logger = logging.getLogger(__name__)

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

# --- ЗАРАНЕЕ ЗАГОТОВЛЕННЫЕ НАБОРЫ ПОЛЕЙ ДЛЯ ИЗВЛЕЧЕНИЯ ---
_PREDEFINED_FIELD_SETS: Dict[str, str] = {
    "кадастровый_объект": """
- ФИО собственника
- Адрес земельного участка
- Адрес объекта недвижимости (здание, строение, сооружение, объект незавершенного строительства)
- Площадь земельного участка
- Площадь объекта недвижимости
- Кадастровый номер земельного участка
- Кадастровый номер объекта недвижимости
- Вид, номер, дата и время государственной регистрации права
- Виды разрешенного использования
- Категория земель, к которой отнесен земельный участок
- Назначение объекта недвижимости
- Наименование объекта недвижимости
- Адреса для связи с правообладателями
""",
    "паспортные_данные": """
- Серия и номер паспорта собственника
- Кем выдан паспорт собственника
- Когда выдан паспорт собственника
- Адрес регистрации собственника
- Серия и номер паспорта доверенного лица
- Кем выдан паспорт доверенного лица
- Когда выдан паспорт доверенного лица
- Адрес регистрации доверенного лица
""",
    "реквизиты_организации": """
- Наименование организации или ИП
- ФИО руководителя
- Юридический адрес
- ИНН
- ОГРН(ИП)
- Наименование банка
- Корреспондентский счет
- БИК
- Номер счета
"""
}

# Добавляем общий набор, который включает все вышеперечисленные
_PREDEFINED_FIELD_SETS["все_данные"] = (
    _PREDEFINED_FIELD_SETS["кадастровый_объект"] + "\n" +
    _PREDEFINED_FIELD_SETS["паспортные_данные"] + "\n" +
    _PREDEFINED_FIELD_SETS["реквизиты_организации"]
).strip()


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
            elif tool_name == "text_extractor":
                parameters["properties"] = {
                    "file_path": {"type": "string", "description": "Полный путь к файлу для извлечения текста (например, /home/user/document.pdf)."},
                    "language": {"type": "string", "description": "Опциональный код языка для OCR (например, 'en' для английского, 'ru' для русского). Используется для изображений и PDF. По умолчанию определяется автоматически."} 
                }
                parameters["required"] = ["file_path"]
            elif tool_name == "structured_data_extractor": # ОБНОВЛЕННЫЙ ИНСТРУМЕНТ
                parameters["properties"] = {
                    "file_path": {"type": "string", "description": "Полный путь к файлу для извлечения структурированных данных."},
                    "field_set_name": {"type": "string", "description": f"Название предопределенного набора полей для извлечения. Доступные наборы: {', '.join(_PREDEFINED_FIELD_SETS.keys())}. Используй 'все_данные' для извлечения всей доступной информации."}
                }
                parameters["required"] = ["file_path", "field_set_name"]
            elif tool_name == "vector_store_cleaner": # ОБНОВЛЕННЫЙ ИНСТРУМЕНТ
                parameters["properties"] = {
                    "confirm": {"type": "boolean", "description": "Необходимо установить в 'true' для подтверждения очистки векторного хранилища. Используйте с осторожностью!."}
                }
                parameters["required"] = ["confirm"]
            elif tool_name == "ftp_audio_processor":
                parameters["properties"] = {
                    "ftp_host": {"type": "string", "description": "Адрес FTP-сервера (например, 'ftp.example.com')."},
                    "ftp_user": {"type": "string", "description": "Имя пользователя для подключения к FTP."},
                    "ftp_password": {"type": "string", "description": "Пароль для подключения к FTP."},
                    "remote_path": {"type": "string", "description": "Путь к папке на FTP-сервере для мониторинга (например, '/audio_uploads/')."},
                    "local_download_dir": {"type": "string", "description": "Локальная директория для временного скачивания файлов перед транскрибацией (например, '/tmp/ftp_audio_downloads/')."},
                    "allowed_audio_extensions": {"type": "array", "items": {"type": "string"}, "description": "Опциональный список расширений аудиофайлов для отслеживания (например, ['.mp3', '.wav'])."},
                    "clear_remote_after_processing": {"type": "boolean", "description": "Если 'true', удаляет файлы с FTP-сервера после успешной транскрибации. Используйте с осторожностью! По умолчанию 'false'."}
                }
                parameters["required"] = ["ftp_host", "ftp_user", "ftp_password", "remote_path", "local_download_dir"]
            elif tool_name == "google_calendar_tool":
                parameters["properties"] = {
                    "summary": {"type": "string", "description": "Название или краткое описание события в календаре."},
                    "start_time": {"type": "string", "description": "Время начала события в формате ISO 8601 (например, '2025-08-21T09:00:00+03:00')."},
                    "end_time": {"type": "string", "description": "Время окончания события в формате ISO 8601 (например, '2025-08-21T10:00:00+03:00')."},
                    "description": {"type": "string", "description": "Подробное описание события."},
                    "location": {"type": "string", "description": "Место проведения события."},
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "Список email-адресов участников."}
                }
                parameters["required"] = ["summary", "start_time", "end_time"]
            elif tool_name == "google_tasks_tool":
                parameters["properties"] = {
                    "title": {"type": "string", "description": "Название или краткое описание задачи."},
                    "due_date": {"type": "string", "description": "Дата срока выполнения задачи в формате ISO 8601 (только дата, например, '2025-08-21')."},
                    "notes": {"type": "string", "description": "Подробное описание задачи."},
                    "task_list_id": {"type": "string", "description": "ID списка задач, в который следует добавить задачу. Если не указан, используется основной список."}
                }
                parameters["required"] = ["title"]
            elif tool_name == "google_keep_note_tool":
                parameters["properties"] = {
                    "title": {"type": "string", "description": "Заголовок для новой заметки."},
                    "content": {"type": "string", "description": "Основное содержание заметки."}
                }
                parameters["required"] = ["title", "content"]

            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": parameters
                }
            })
        return formatted_tools

    def _chunk_text(self, text: str, max_chunk_size: int = 1500, overlap_size: int = 200) -> List[str]:
        if not text:
            return []
        chunks = []
        text_length = len(text)
        current_pos = 0
        while current_pos < text_length:
            end_pos = min(current_pos + max_chunk_size, text_length)
            chunk = text[current_pos:end_pos]
            chunks.append(chunk)
            current_pos += (max_chunk_size - overlap_size)
            if current_pos >= text_length:
                break 
        return chunks

    async def _run_processors(self, processors: List[AgentProcessor], context: AgentContext):
        for processor in processors:
            await processor(self, context)

    async def process_message(self, text_input: Optional[str] = None, file_input: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        context = AgentContext(text_input=text_input, file_input_path=file_input)
        
        # --- ОБНОВЛЕННАЯ ЛОГИКА ОБРАБОТКИ ФАЙЛОВОГО ВВОДА ---
        if context.file_input_path:
            if not os.path.exists(context.file_input_path):
                context.final_response = f"Ошибка: Файл не найден по пути '{context.file_input_path}'."
                context.processed_successfully = False
                return context.final_response
            
            # Если получен путь к файлу, *исходный text_input* становится запросом к LLM
            # (чтобы LLM сама решила, какой инструмент использовать).
            # Если file_input был передан без text_input, создаем дефолтный запрос.
            if not context.text_input:
                context.text_input = f"Пожалуйста, проанализируй содержимое файла: {context.file_input_path}."

            print(f"\nАгент получил запрос с файлом: '{context.file_input_path}'. Исходный запрос LLM: '{context.text_input[:100]}...'")
            self.memory.add_message(role="user", content=context.text_input) 
            context.current_prompt_for_llm = context.text_input 
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
                retrieved_docs = await self.vector_store.similarity_search(context.current_prompt_for_llm, k=3) 
                if retrieved_docs:
                    context.rag_context = [f"Контекст из векторного хранилища: {doc}" for doc in retrieved_docs]
                    print(f"Извлечен контекст из векторного хранилища: {context.rag_context}")
            
            combined_system_prompt = self._system_prompt
            if context.rag_context:
                combined_system_prompt += "\n\n" + "\n".strip().join(context.rag_context)

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
                        tool_id = tool_call.get("id", "no_id")
                        tool_name = tool_call["name"]
                        tool_args = tool_call["arguments"]
                        
                        try:
                            print(f"LLM запросила выполнение инструмента '{tool_name}' с аргументами: {tool_args}")
                            tool_output = await self.tool_manager.execute_tool(tool_name, **tool_args) 
                            
                            logger.info(f"DEBUG: Полный вывод инструмента '{tool_name}' (ID: {tool_id}):\n{str(tool_output)[:500]}...\n--- КОНЕЦ ВЫВОДА ИНСТРУМЕНТА ---")
                            
                            # --- УПРОЩЕННАЯ ОБРАБОТКА ВЫВОДА ИНСТРУМЕНТОВ ---
                            # StructuredDataExtractorTool и VectorStoreCleanerTool теперь сами форматируют вывод
                            formatted_output = (
                                f"**Результат выполнения инструмента '{tool_name}' (ID: {tool_id}):**\n"
                                f"```\n{str(tool_output)}\n```"
                            )
                            
                            context.tool_outputs.append({"tool_call_id": tool_id, "output": formatted_output})
                            self.memory.add_message(role="tool", content=json.dumps({"tool_call_id": tool_id, "output": formatted_output}))
                            
                        except Exception as e:
                            error_msg = f"Ошибка при выполнении инструмента '{tool_name}': {type(e).__name__}: {e}. Аргументы: {tool_args}"
                            self.memory.add_message(role="system_error", content=error_msg)
                            print(error_msg)
                            context.tool_outputs.append({"tool_call_id": tool_id, "output": f"Ошибка выполнения: {error_msg}"})
                            self.memory.add_message(role="tool", content=json.dumps({"tool_call_id": tool_id, "output": f"Ошибка выполнения: {error_msg}"}))
                    
                    await self._run_processors(self._post_tool_execution_processors, context)

                    tool_results_for_llm = [output_item["output"] for output_item in context.tool_outputs]
                    tool_results_str = "\n\n".join(tool_results_for_llm)

                    context.current_prompt_for_llm = (
                        f"Мой исходный запрос: '{context.text_input}'\n\n"
                        f"Я успешно выполнил необходимые действия и получил следующие данные:\n\n"
                        f"{tool_results_str}\n\n"
                        f"**Теперь, пожалуйста, используя только эти полученные данные, сформулируй полный и связный ответ на мой исходный запрос. "
                        f"Если текста недостаточно для ответа, прямо сообщи об этом, опираясь на предоставленный текст.**"
                    )
                    
                else:
                    context.final_response = context.llm_response_raw
                    context.processed_successfully = True
                    break
                        
            except Exception as e:
                context.error_message = f"Ошибка при вызове LLM или парсинге ответа: {type(e).__name__}: {e}"
                context.final_response = f"Извините, произошла внутренняя ошибка при обработке запроса: {type(e).__name__}: {e}"
                self.memory.add_message(role="system_error", content=context.error_message)
                context.processed_successfully = False
                break

        if not context.final_response:
            context.final_response = "Я выполнил некоторые действия с инструментами, но пока не могу сформулировать окончательный ответ. Попробуйте уточнить запрос."
            context.processed_successfully = False
        
        await self._run_processors(self._final_response_processors, context)
        
        self.memory.add_message(role="assistant", content=context.final_response)
        return context.final_response