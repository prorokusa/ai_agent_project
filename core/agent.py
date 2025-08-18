import os
import json 
from typing import List, Dict, Union, Any, Optional, Callable, Awaitable 

from interfaces.llm import AbstractLLM
from interfaces.memory import Memory
from interfaces.vector_store import VectorStore
from interfaces.tool import Tool
from core.tool_manager import ToolManager # Импорт ToolManager

# Класс AgentContext используется для передачи состояния и данных
# по всему конвейеру обработки запроса агента.
class AgentContext:
    def __init__(self, 
                 text_input: Optional[str] = None, 
                 file_input_path: Optional[str] = None):
        self.text_input: Optional[str] = text_input # Исходный текстовый ввод пользователя (что пользователь сказал)
        self.file_input_path: Optional[str] = file_input_path # Путь к входному файлу (если есть)
        self.file_content: Optional[str] = None # Содержимое файла, если он был прочитан (для небольших файлов)
        self.rag_context: List[str] = [] # Контекст, извлеченный из векторного хранилища (для RAG)
        self.llm_response_raw: Union[str, Dict[str, Any], None] = None # Сырой ответ от LLM (может быть текст или JSON с вызовами инструментов)
        self.tool_calls: List[Dict[str, Any]] = [] # Распарсенные вызовы инструментов из ответа LLM
        self.tool_outputs: List[Dict[str, Any]] = [] # Результаты выполнения инструментов
        self.final_response: Optional[str] = None # Окончательный ответ агента пользователю
        self.error_message: Optional[str] = None # Сообщение об ошибке, если что-то пошло не так
        self.processed_successfully: bool = False # Флаг успешности обработки запроса
        self.current_prompt_for_llm: str = "" # Текущий промпт, который будет отправлен в LLM (может изменяться плагинами)
        self.metadata: Dict[str, Any] = {} # Произвольные метаданные для передачи между этапами обработки или для плагинов

# AgentProcessor - это тип для асинхронных функций-обработчиков (плагинов),
# которые принимают экземпляр AIAgent и AgentContext.
AgentProcessor = Callable[['AIAgent', AgentContext], Awaitable[None]]

# Основной класс AI Агента.
# Координирует работу LLM, инструментов, памяти и векторного хранилища.
class AIAgent:
    def __init__(self, 
                 llm: AbstractLLM, 
                 memory: Memory,  
                 vector_store: Optional[VectorStore] = None, # Векторное хранилище опционально
                 system_prompt: Optional[str] = None, # Системный промпт для LLM
                 max_tool_iterations: int = 3): # Максимальное количество итераций для вызова инструментов
        
        self.llm = llm # Экземпляр LLM
        self.tool_manager = ToolManager() # Менеджер инструментов для регистрации и выполнения
        self.memory = memory # Экземпляр памяти для хранения истории чата
        self.vector_store = vector_store # Экземпляр векторного хранилища
        self._system_prompt = system_prompt # Исходный системный промпт
        self.max_tool_iterations = max_tool_iterations # Лимит на вызовы инструментов в одной итерации
        
        # Списки для хранения зарегистрированных функций-обработчиков (плагинов)
        self._pre_llm_processors: List[AgentProcessor] = []
        self._post_llm_processors: List[AgentProcessor] = []
        self._post_tool_execution_processors: List[AgentProcessor] = []
        self._final_response_processors: List[AgentProcessor] = []

        print("AI Агент инициализирован.")
        if self._system_prompt:
            print(f"Агент инициализирован с системным промптом: '{self._system_prompt[:50]}...'")

    def register_tool(self, tool: Tool):
        """
        Регистрирует новый инструмент в менеджере инструментов агента.
        """
        self.tool_manager.register_tool(tool)

    def register_processor(self, stage: str, processor: AgentProcessor):
        """
        Регистрирует пользовательскую функцию-обработчик (плагин) для определенного этапа.
        
        Args:
            stage (str): Этап обработки, к которому привязывается процессор.
                         Возможные значения: 'pre_llm', 'post_llm', 'post_tool_execution', 'final_response'.
            processor (AgentProcessor): Асинхронная функция, которая будет вызвана на этом этапе.
        """
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
        """
        Форматирует информацию о зарегистрированных инструментах в формат,
        понятный для LLM (OpenAI-совместимый формат Tool Calling).
        """
        formatted_tools = []
        for tool_info in self.tool_manager.list_tools():
            tool_name = tool_info["name"]
            tool_description = tool_info["description"]
            
            # Базовая структура параметров, которая будет заполнена ниже.
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }

            # Специальное описание параметров для каждого известного инструмента.
            # Важно: если вы добавляете новый инструмент, убедитесь, что его параметры
            # описаны здесь, чтобы LLM могла правильно его вызывать.
            if tool_name == "calculator":
                parameters["properties"] = {"expression": {"type": "string", "description": "Математическое выражение для вычисления."}}
                parameters["required"] = ["expression"]
            elif tool_name == "google_cse_search": 
                parameters["properties"] = {"query": {"type": "string", "description": "Поисковый запрос для интернета."}}
                parameters["required"] = ["query"]
            elif tool_name == "text_extractor": # Описание для TextExtractorTool
                parameters["properties"] = {
                    "file_path": {"type": "string", "description": "Полный путь к файлу для извлечения текста (например, /home/user/document.pdf)."},
                    "language": {"type": "string", "description": "Опциональный код языка для OCR (например, 'en' для английского, 'ru' для русского). Используется для изображений и PDF. По умолчанию определяется автоматически."} 
                }
                parameters["required"] = ["file_path"] # 'language' остается опциональным, поэтому его нет в required
            # Если у вас есть другие инструменты, добавьте их сюда аналогичным образом:
            # elif tool_name == "my_new_tool":
            #     parameters["properties"] = {"arg1": {"type": "string", "description": "Описание аргумента 1."}}
            #     parameters["required"] = ["arg1"]

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
        """
        Выполняет все зарегистрированные функции-обработчики для заданного этапа.
        """
        for processor in processors:
            await processor(self, context)

    async def process_message(self, text_input: Optional[str] = None, file_input: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Основной метод для обработки входящего сообщения пользователя.
        Управляет всем циклом работы агента: от получения ввода до формирования ответа.
        
        Args:
            text_input (Optional[str]): Текстовый ввод от пользователя.
            file_input (Optional[str]): Путь к файлу, который нужно обработать.
            
        Returns:
            Union[str, Dict[str, Any]]: Окончательный ответ агента или сообщение об ошибке.
        """
        # Создаем новый контекст для каждого запроса.
        context = AgentContext(text_input=text_input, file_input_path=file_input)
        
        # Обработка файлового ввода: чтение содержимого файла и преобразование в текстовый промпт.
        if context.file_input_path:
            print(f"\nАгент получил файл: '{context.file_input_path}'")
            try:
                # ВНИМАНИЕ: Для больших файлов лучше использовать TextExtractorTool.
                # Это чтение предназначено для простых текстовых файлов,
                # или для случаев, когда LLM должна *видеть* часть файла напрямую в промпте.
                with open(context.file_input_path, 'r', encoding='utf-8') as f:
                    # Ограничиваем чтение файла для промпта первыми 1000 символами,
                    # чтобы избежать слишком больших промптов.
                    context.file_content = f.read()
                context.text_input = (
                    f"Проанализируй следующий документ:\n\n"
                    f"```document\n{context.file_content[:1000]}...\n```\n\n"
                    f"Кратко summarize его ключевые моменты и предложи вопросы, которые можно задать по этому документу."
                )
                # Добавляем преобразованный ввод в память как сообщение пользователя.
                self.memory.add_message(role="user", content=context.text_input) 
                context.current_prompt_for_llm = context.text_input 
            except Exception as e:
                context.error_message = f"Ошибка при обработке файла '{context.file_input_path}': {e}"
                self.memory.add_message(role="system_error", content=context.error_message)
                context.final_response = {"status": "error", "message": context.error_message}
                context.processed_successfully = False
                return context.final_response
        elif context.text_input:
            # Если есть только текстовый ввод, добавляем его в память.
            self.memory.add_message(role="user", content=context.text_input)
            context.current_prompt_for_llm = context.text_input
        else:
            # Если нет ни текстового, ни файлового ввода, возвращаем ошибку.
            context.final_response = "Не получено никакого ввода (текста или файла)."
            context.processed_successfully = False
            return context.final_response

        # Главный цикл итераций агента (для выполнения цепочки вызовов инструментов).
        # Агент может совершать несколько вызовов инструментов и LLM в рамках одного запроса.
        for i in range(self.max_tool_iterations):
            print(f"\n--- Итерация {i+1} агента ---")
            
            # 1. Запуск пре-LLM обработчиков (плагинов).
            # Позволяет модифицировать контекст или промпт перед вызовом LLM.
            await self._run_processors(self._pre_llm_processors, context)

            # 2. Извлечение контекста из векторного хранилища (RAG - Retrieval Augmented Generation).
            # Поиск релевантной информации в базе знаний агента.
            retrieved_context = []
            if self.vector_store and context.current_prompt_for_llm: 
                # Вызов асинхронного метода similarity_search.
                retrieved_docs = await self.vector_store.similarity_search(context.current_prompt_for_llm, k=3) 
                if retrieved_docs:
                    # Добавляем извлеченный контекст к системному промпту для обогащения запроса LLM.
                    context.rag_context = [f"Контекст из памяти: {doc}" for doc in retrieved_docs]
                    print(f"Извлечен контекст из векторного хранилища: {context.rag_context}")
            
            # Формирование комбинированного системного промпта.
            # Объединяет базовый системный промпт агента с контекстом, извлеченным из RAG.
            combined_system_prompt = self._system_prompt
            if context.rag_context:
                combined_system_prompt += "\n\n" + "\n".join(context.rag_context)

            try:
                # Получаем текущую историю чата из памяти.
                conversation_history = self.memory.get_history()
                # Форматируем информацию о доступных инструментах в формат, понятный для LLM.
                available_tools_for_llm = self._format_tools_for_llm()

                # 3. Вызов LLM для генерации ответа или вызова инструмента.
                context.llm_response_raw = await self.llm.generate_response( 
                    prompt=context.current_prompt_for_llm, # Текущий промпт для LLM
                    system_prompt=combined_system_prompt, # Комбинированный системный промпт
                    history=conversation_history, # Полная история диалога
                    tools=available_tools_for_llm # Описание доступных инструментов
                )
                
                # 4. Запуск пост-LLM обработчиков (плагинов).
                # Позволяет обработать сырой ответ LLM.
                await self._run_processors(self._post_llm_processors, context)

                # 5. Обработка вызовов инструментов, если LLM их запросила.
                if isinstance(context.llm_response_raw, dict) and "tool_calls" in context.llm_response_raw:
                    context.tool_calls = context.llm_response_raw["tool_calls"]
                    
                    # Добавляем вызовы инструментов в память (для сохранения контекста диалога).
                    self.memory.add_message(role="assistant", content=json.dumps({"tool_calls": context.tool_calls}))
                    
                    for tool_call in context.tool_calls:
                        tool_id = tool_call.get("id", "no_id") # Получаем ID вызова инструмента
                        tool_name = tool_call["name"] # Имя инструмента
                        tool_args = tool_call["arguments"] # Аргументы для инструмента
                        
                        try:
                            print(f"LLM запросила выполнение инструмента '{tool_name}' с аргументами: {tool_args}")
                            # Выполняем инструмент через ToolManager.
                            tool_output = await self.tool_manager.execute_tool(tool_name, **tool_args) 
                            context.tool_outputs.append({"tool_call_id": tool_id, "output": tool_output})
                            # Добавляем результат выполнения инструмента в память.
                            self.memory.add_message(role="tool", content=json.dumps({"tool_call_id": tool_id, "output": tool_output}))
                            
                        except Exception as e:
                            # Обработка ошибок при выполнении инструмента.
                            error_msg = f"Ошибка при выполнении инструмента '{tool_name}': {e}. Аргументы: {tool_args}"
                            self.memory.add_message(role="system_error", content=error_msg)
                            print(error_msg)
                            context.tool_outputs.append({"tool_call_id": tool_id, "output": f"Ошибка выполнения: {error_msg}"})
                            self.memory.add_message(role="tool", content=json.dumps({"tool_call_id": tool_id, "output": f"Ошибка выполнения: {error_msg}"}))
                    
                    # 6. Запуск обработчиков после выполнения инструментов.
                    await self._run_processors(self._post_tool_execution_processors, context)

                    # --- ВАЖНОЕ ИЗМЕНЕНИЕ ДЛЯ ПРАВИЛЬНОГО ПОТОКА АГЕНТА ПОСЛЕ ВЫЗОВА ИНСТРУМЕНТА ---
                    # Если LLM вызвала инструменты и они выполнились, на следующей итерации
                    # LLM должна получить инструкцию сформулировать окончательный ответ,
                    # используя результаты этих инструментов и исходный запрос пользователя.
                    if context.tool_outputs: # Если были результаты от инструментов в этой итерации
                        # Формируем промпт для LLM, который явно указывает ей на результаты и исходную задачу.
                        tool_results_for_llm = []
                        for output in context.tool_outputs:
                            tool_call_id = output.get('tool_call_id', '')
                            tool_name = "неизвестный инструмент"
                            # Находим имя инструмента, если возможно
                            for tc in context.tool_calls:
                                if tc.get('id') == tool_call_id:
                                    tool_name = tc['name']
                                    # Получаем аргументы из tool_call для использования в промпте
                                    tool_args_for_prompt = tc.get('arguments', {})
                                    break
                            
                            output_content = output.get('output', '')
                            
                            # Здесь мы делаем ключевое изменение:
                            # Явно указываем, что это "извлеченный текст из файла" для text_extractor,
                            # и ПЕРЕДАЕМ ПОЛНЫЙ ТЕКСТ.
                            if tool_name == "text_extractor" and isinstance(output_content, str):
                                file_path_in_args = tool_args_for_prompt.get("file_path", "неизвестный файл")
                                formatted_output = (
                                    f"**Успешно извлеченный текст из файла '{file_path_in_args}':**\n" 
                                    f"```markdown\n{output_content}\n```" # ТЕКСТ БЕЗ ОБРЕЗКИ
                                )
                            else:
                                # Для других инструментов или ошибок
                                formatted_output = (
                                    f"**Результат выполнения инструмента '{tool_name}' (ID: {tool_call_id}):**\n"
                                    f"```\n{str(output_content)}\n```" # ТЕКСТ БЕЗ ОБРЕЗКИ
                                )
                            
                            tool_results_for_llm.append(formatted_output)

                        tool_results_str = "\n\n".join(tool_results_for_llm)

                        context.current_prompt_for_llm = (
                            f"Мой исходный запрос: '{context.text_input}'\n\n"
                            f"Я успешно выполнил необходимые действия и получил следующие данные:\n\n"
                            f"{tool_results_str}\n\n" # Здесь уже форматированный текст
                            f"**Теперь, пожалуйста, используя только эти полученные данные, сформулируй полный и связный ответ на мой исходный запрос. "
                            f"Не запрашивай путь к файлу снова и не сообщай об отсутствии текста, если он был предоставлен выше. "
                            f"Если текста недостаточно для ответа, прямо сообщи об этом, опираясь на предоставленный текст.**"
                        )
                    else: 
                        # Если инструментов не было или они не дали результатов,
                        # но мы почему-то оказались здесь (что не должно быть при break выше),
                        # возвращаемся к исходному промпту.
                        context.current_prompt_for_llm = context.text_input 
                    
                else:
                    # Если LLM не запросила инструменты, это означает, что она дала окончательный текстовый ответ.
                    context.final_response = context.llm_response_raw
                    context.processed_successfully = True
                    break # Завершаем цикл итераций, так как ответ готов.
                        
            except Exception as e:
                # Общая обработка ошибок при вызове LLM или парсинге ее ответа.
                context.error_message = f"Ошибка при вызове LLM или парсинге ответа: {e}"
                context.final_response = f"Извините, произошла внутренняя ошибка при обработке запроса: {e}"
                self.memory.add_message(role="system_error", content=context.error_message)
                context.processed_successfully = False
                break # Завершаем цикл при ошибке.

        # Если после всех итераций не удалось получить окончательный ответ (например, из-за ошибки
        # или агент застрял в цикле инструментов без финального ответа), формируем дефолтный ответ.
        if not context.final_response:
            context.final_response = "Я выполнил некоторые действия с инструментами, но пока не могу сформулировать окончательный ответ. Попробуйте уточнить запрос."
            context.processed_successfully = False
        
        # 7. Запуск обработчиков перед возвратом окончательного ответа.
        await self._run_processors(self._final_response_processors, context)
        
        # Добавляем окончательный ответ агента в память.
        self.memory.add_message(role="assistant", content=context.final_response)
        return context.final_response