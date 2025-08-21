import os
import json
import asyncio
import logging
from dotenv import load_dotenv
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

from core.agent import AIAgent, AgentContext, AgentProcessor, _PREDEFINED_FIELD_SETS # Импортируем наборы полей
from implementations.llms.simple_inference_llm import SimpleInferenceLLM
from implementations.llms.openai_llm import OpenAI_LLM
from implementations.llms.openrouter_llm import OpenRouter_LLM

from implementations.tools.calculator_tool import CalculatorTool
from implementations.tools.web_search_tool import GoogleCSESearchTool
from implementations.tools.text_extractor_tool import TextExtractorTool 
from implementations.tools.structured_data_extractor_tool import StructuredDataExtractorTool # НОВЫЙ ИМПОРТ
from implementations.tools.vector_store_cleaner_tool import VectorStoreCleanerTool # НОВЫЙ ИМПОРТ

from implementations.memory.chat_history_memory import ChatHistoryMemory
from implementations.vector_stores.chromadb_store import ChromaDBStore
from implementations.vector_stores.supabase_store import SupabaseVectorStore
from implementations.vector_stores.simple_vector_store import SimpleVectorStore 

from utils.audio_transcriber import AudioTranscriber
from utils.local_file_monitor import LocalFileMonitor

import config 

audio_transcriber = None
try:
    audio_transcriber = AudioTranscriber()
    logger.info("AudioTranscriber успешно инициализирован.")
except ValueError as e:
    logger.error(f"Ошибка при инициализации AudioTranscriber: {e}")
    logger.error("Функции транскрибации аудио будут недоступны.")
except Exception as e:
    logger.error(f"Неизвестная ошибка при инициализации AudioTranscriber: {e}")
    logger.error("Функции транскрибации аудио будут недоступны.")

local_monitor = None
ai_agent: AIAgent = None 

async def handle_monitored_audio_file(audio_filepath: str):
    logger.info(f"Monitor: Обнаружен новый аудиофайл: {audio_filepath}. Начинаю обработку...")
    
    global ai_agent
    if ai_agent is None:
        logger.error("Monitor: AI Агент не инициализирован. Невозможно обработать файл.")
        return
    if audio_transcriber is None:
        logger.error("Monitor: Модуль AudioTranscriber не инициализирован. Невозможно транскрибировать аудио.")
        return

    transcribed_text = audio_transcriber.transcribe_audio(audio_filepath)
    
    if transcribed_text:
        logger.info(f"Monitor: Аудио транскрибировано. Передаю текст агенту: '{transcribed_text[:100]}...'")
        response = await ai_agent.process_message(text_input=transcribed_text)
        logger.info(f"Monitor: Агент ответил на транскрипцию: {response}")
    else:
        logger.warning(f"Monitor: Не удалось транскрибировать аудиофайл: {audio_filepath}.")

async def my_custom_pre_llm_processor(agent: AIAgent, context: AgentContext):
    print(f"\n[PLUGIN: pre_llm] Выполняется перед запросом LLM...")
    import datetime
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    context.current_prompt_for_llm = f"Сегодня {current_date}. " + context.current_prompt_for_llm
    context.metadata["current_date"] = current_date
    print(f"[PLUGIN: pre_llm] Добавлена дата в промпт: {context.current_prompt_for_llm[:50]}...")

async def my_custom_final_response_processor(agent: AIAgent, context: AgentContext):
    print(f"\n[PLUGIN: final_response] Получен окончательный ответ агента.")
    if context.processed_successfully:
        print(f"[PLUGIN: final_response] Ответ успешный. Сообщение: {context.final_response[:50]}...")
    else:
        print(f"[PLUGIN: final_response] Ответ содержит ошибку. Ошибка: {context.error_message}")
    print(f"[PLUGIN: final_response] Метаданные контекста: {context.metadata}")

async def process_single_audio_input(agent: AIAgent, audio_filepath: str) -> str | None:
    if audio_transcriber is None:
        logger.error("Модуль AudioTranscriber не был успешно инициализирован. Невозможно транскрибировать аудио.")
        return None

    if not os.path.exists(audio_filepath):
        logger.error(f"Ошибка: Аудиофайл '{audio_filepath}' не найден.")
        return None

    logger.info(f"Начинаем транскрибацию аудиофайла: {audio_filepath} через OpenAI API...")
    transcribed_text = audio_transcriber.transcribe_audio(audio_filepath)

    if transcribed_text:
        logger.info(f"Транскрибация завершена. Полученный текст (первые 200 символов):")
        logger.info(f"'{transcribed_text[:200]}...'")
        
        logger.info("Передача транскрибированного текста агенту для обработки...")
        response = await agent.process_message(text_input=transcribed_text)
        logger.info(f"Агент (из аудио): {response}")
        return transcribed_text
    else:
        logger.warning("Не удалось транскрибировать аудио. Дальнейшая обработка агентом пропущена.")
        return None

async def main():
    global local_monitor, ai_agent

    # 1. Выбор LLM:
    # my_llm = OpenRouter_LLM(model_name="qwen/qwen3-coder:free") 
    my_llm = OpenAI_LLM(model_name="gpt-4o-mini") 
    # my_llm = SimpleInferenceLLM()

    # 2. Инициализация памяти:
    my_memory = ChatHistoryMemory()

    # 3. Инициализация векторного хранилища:
    my_vector_store = SupabaseVectorStore(llm_for_embedding=my_llm) 
    # my_vector_store = ChromaDBStore(llm_for_embedding=my_llm, persist_directory="./chroma_db")
    # my_vector_store = SimpleVectorStore(llm_for_embedding=my_llm)
    
    # 4. Системный промпт для агента:
    # ОБНОВЛЕННЫЙ СИСТЕМНЫЙ ПРОМПТ
    system_prompt = (
        "Ты очень полезный, дружелюбный и компетентный AI ассистент, специализирующийся на анализе документов, особенно юридических и кадастровых. "
        "Твоя задача - точно отвечать на вопросы, использовать предоставленные инструменты, когда это уместно, "
        "и извлекать информацию из векторного хранилища для ответов на вопросы о личных данных или контексте. "
        "Всегда старайся быть вежливым и кратким, если это возможно. "
        "Если пользователь задает вопрос, на который может ответить инструмент, предложи его использовать или используй его напрямую, если это очевидно. "
        "Если вопрос касается твоей собственной информации или контекста, используй поиск по векторному хранилищу. "
        "После выполнения инструмента или поиска в векторном хранилище, обобщи результаты и предоставь связный ответ."
        "Будь готов использовать функцию `google_cse_search` для поиска актуальной информации в интернете, когда это необходимо."
        "Особенно используй `google_cse_search`, если вопрос касается текущих событий, статистических данных или общих фактов, отсутствующих в твоей памяти."
        "Ты можешь извлекать весь текст из файла с помощью `text_extractor`. "
        "Для извлечения конкретных, структурированных данных из файла по заданному набору полей (например, 'кадастровый_объект', 'паспортные_данные', 'реквизиты_организации' или 'все_данные'), используй инструмент `structured_data_extractor`. "
        "Если файл большой, `structured_data_extractor` автоматически проиндексирует его содержимое в векторное хранилище и сразу попытается извлечь запрошенные поля."
        "Ты можешь очистить всю информацию из векторного хранилища, используя инструмент `vector_store_cleaner`. Для подтверждения операции, убедись, что аргумент 'confirm' установлен в 'true'."
        "После выполнения инструмента, **всегда** включай его результат (если он является ответом на вопрос пользователя) в свой финальный ответ. Формулируй ответ пользователю ясно, кратко и дружелюбно, прямо отвечая на его исходный вопрос, используя полученные данные."
    )
    # 5. Инициализация основного AI Агента.
    ai_agent = AIAgent(llm=my_llm, memory=my_memory, vector_store=my_vector_store, system_prompt=system_prompt)

    # Регистрация плагинов (без изменений)
    # ai_agent.register_processor('pre_llm', my_custom_pre_llm_processor)
    # ai_agent.register_processor('final_response', my_custom_final_response_processor)

    # --- Регистрация инструментов для агента ---
    ai_agent.register_tool(CalculatorTool())
    ai_agent.register_tool(GoogleCSESearchTool())
    ai_agent.register_tool(TextExtractorTool())
    # РЕГИСТРИРУЕМ НОВЫЕ ИНСТРУМЕНТЫ
    # Передаем LLM и vector_store в StructuredDataExtractorTool
    ai_agent.register_tool(StructuredDataExtractorTool(text_extractor_tool=TextExtractorTool(), llm_for_parsing=my_llm, vector_store=my_vector_store, predefined_field_sets=_PREDEFINED_FIELD_SETS))
    ai_agent.register_tool(VectorStoreCleanerTool(vector_store=my_vector_store))

    # Загрузка начальных документов в векторное хранилище (без изменений)
    await my_vector_store.clear()
    await ai_agent.vector_store.add_documents([
        "Мое любимое хобби - это чтение книг по истории и философия.",
        "Я живу в городе Москва, работаю инженером в IT-компании, мой адрес: ул. Пушкина, 10.",
        "Последний раз мы обсуждали планы на отпуск в горах Кавказа в июле 2024 года.",
        "Имя моего питомца - Барсик, он рыжий кот и очень любит играть с лазерной указкой.",
        "Мои предпочтения в еде включают итальянскую кухню, особенно пиццу и пасту."
    ], metadatas=[
        {"type": "hobby"},
        {"type": "personal_info", "city": "Москва"},
        {"type": "past_discussion", "year": 2024},
        {"type": "pet_info"},
        {"type": "food_prefs"}
    ])
    logger.info("Документы для векторного хранилища загружены.")

    parser = argparse.ArgumentParser(description="AI Agent Project with Audio Transcription and Folder Monitoring.")
    parser.add_argument("--audio", type=str,
                        help="Путь к ОДНОМУ аудиофайлу для транскрибации (например, 'path/to/my_audio.mp3').")
    args = parser.parse_args()

    supported_audio_extensions_for_monitor = [
        '.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm', 
        '.awb', '.amr'
    ]

    if config.MONITOR_DIRECTORY:
        if not os.path.isdir(config.MONITOR_DIRECTORY):
            logger.error(f"Ошибка: Путь '{config.MONITOR_DIRECTORY}' из config.py не является существующей директорией. Автоматический мониторинг папки не запущен.")
        else:
            logger.info(f"Настройка MONITOR_DIRECTORY найдена в config.py. Запуск мониторинга папки: {config.MONITOR_DIRECTORY}")
            try:
                local_monitor = LocalFileMonitor(
                    directory_path=config.MONITOR_DIRECTORY,
                    interval=config.MONITOR_INTERVAL_SECONDS,
                    callback_func=handle_monitored_audio_file,
                    allowed_extensions=supported_audio_extensions_for_monitor
                )
                monitor_task = local_monitor.run_as_task()
                logger.info("Мониторинг запущен в фоновом режиме. Нажмите Ctrl+C для остановки.")
                await asyncio.Future()
            except ValueError as e:
                logger.error(f"Ошибка при инициализации монитора: {e}. Проверьте путь к директории в config.py.")
            except asyncio.CancelledError:
                logger.info("Мониторинг директории отменен.")
            finally:
                if local_monitor:
                    local_monitor.stop_monitoring()
                if 'monitor_task' in locals() and not monitor_task.done():
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass
                logger.info("Программа завершена после мониторинга.")
            return

    if args.audio:
        logger.info(f"Обнаружен аргумент --audio. Попытка обработки файла: {args.audio}")
        processed_text = await process_single_audio_input(ai_agent, args.audio)
        if processed_text:
            logger.info("Аудио успешно транскрибировано и текст передан для обработки агентом.")
        else:
            logger.error("Обработка аудио завершилась с ошибкой или не дала результата.")
        return

    print("\n--- Начало интерактивного диалога с Агентом ---")
    print("Введите 'exit' для завершения.")
    print("Чтобы отправить текстовый или PDF/JPG/PNG файл, используйте 'file:<путь_к_файлу>'")
    print("Чтобы извлечь структурированные данные из файла, например: 'извлеки данные из файла 1.pdf по набору кадастровый_объект'")
    print(f"Доступные наборы данных: {', '.join(_PREDEFINED_FIELD_SETS.keys())}")
    print("Для очистки векторного хранилища, скажите что-то вроде: 'очисти раг' или 'удалить все данные из памяти'")
    print("Чтобы отправить аудиофайл, введите 'audio:<путь_к_файлу.mp3>'")
    print(f"Автоматический мониторинг папки настроен в config.py (если MONITOR_DIRECTORY задан).")
    
    while True:
        user_input = input("\nВы: ")
        
        if user_input.lower() == 'exit':
            print("Завершение диалога.")
            break
        
        # --- ОБРАБОТКА ВВОДА АГЕНТОМ ---
        # Теперь все запросы проходят через LLM, которая решает, какой инструмент использовать.
        # Просто передаем user_input (и file_path, если есть) в process_message.

        if user_input.lower().startswith("file:"):
            file_path = user_input[len("file:"):].strip()
            response = await ai_agent.process_message(file_input=file_path, text_input=f"Проанализируй файл {os.path.basename(file_path)}.")
            print(f"Агент (обработка файла): {response}")
        elif user_input.lower().startswith("audio:"):
            audio_path = user_input[len("audio:"):].strip()
            await process_single_audio_input(ai_agent, audio_path)
        else:
            # Для всех остальных запросов (текстовых, включая команды на извлечение или очистку)
            # просто передаем их агенту. LLM сама решит, какой инструмент вызвать.
            response = await ai_agent.process_message(text_input=user_input)
            print(f"Агент: {response}")

    print("\n--- Полная история чата ---")
    for msg in ai_agent.memory.get_history():
        if msg["role"] == "assistant" and isinstance(msg["content"], str) and "tool_calls" in msg["content"]:
            try:
                content_obj = json.loads(msg["content"])
                if "tool_calls" in content_obj:
                    print(f"assistant (вызов инструмента): {json.dumps(content_obj['tool_calls'], indent=2, ensure_ascii=False)}")
                else:
                    print(f"{msg['role']}: {msg['content']}")
            except json.JSONDecodeError:
                print(f"{msg['role']}: {msg['content']}")
        elif msg["role"] == "tool" and isinstance(msg["content"], str):
             try:
                content_obj = json.loads(msg["content"])
                output_preview = str(content_obj.get('output', ''))[:500] + ('...' if len(str(content_obj.get('output', ''))) > 500 else '')
                print(f"tool (результат): Tool ID: {content_obj.get('tool_call_id', 'N/A')}, Output Preview: {json.dumps(output_preview, indent=2, ensure_ascii=False)}")
             except json.JSONDecodeError:
                print(f"{msg['role']}: {msg['content']}")
        else:
            print(f"{msg['role']}: {msg['content']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nПрограмма остановлена пользователем (Ctrl+C).")
        if local_monitor:
            local_monitor.stop_monitoring()