import os
import json
import asyncio
import logging
from dotenv import load_dotenv
import argparse # ВОТ ЭТОТ ИМПОРТ НУЖНО ДОБАВИТЬ!

# Импортируем все необходимые классы
from implementations.llms.simple_inference_llm import SimpleInferenceLLM
from implementations.llms.openai_llm import OpenAI_LLM
from implementations.llms.openrouter_llm import OpenRouter_LLM

from implementations.tools.calculator_tool import CalculatorTool
from implementations.tools.web_search_tool import GoogleCSESearchTool

from implementations.memory.chat_history_memory import ChatHistoryMemory

from implementations.vector_stores.chromadb_store import ChromaDBStore
from implementations.vector_stores.supabase_store import SupabaseVectorStore

from core.agent import AIAgent, AgentContext
from core.agent import AgentProcessor

# ИМПОРТИРУЕМ ВАШИ МОДУЛИ
from utils.audio_transcriber import AudioTranscriber
from utils.local_file_monitor import LocalFileMonitor

# ИМПОРТИРУЕМ НАСТРОЙКИ ИЗ config.py
import config 

# Настройка базового логирования для всего приложения
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env в самом начале
load_dotenv()

# --- ГЛОБАЛЬНЫЕ ЭКЗЕМПЛЯРЫ ---
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
ai_agent = None 

# --- Callback-функция для монитора файлов ---
async def handle_monitored_audio_file(audio_filepath: str):
    """
    Эта функция будет вызываться LocalFileMonitor при обнаружении нового аудиофайла.
    Она транскрибирует файл и передает текст агенту.
    """
    logger.info(f"Monitor: Обнаружен новый аудиофайл: {audio_filepath}. Начинаю обработку...")
    
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

# --- Пример пользовательского плагина (middleware) ---
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


# --- Функция для обработки одного аудиофайла и передачи агенту ---
async def process_single_audio_input(agent: AIAgent, audio_filepath: str) -> str | None:
    """
    Принимает путь к аудиофайлу, транскрибирует его с помощью OpenAI Whisper API
    и передает текст для дальнейшей обработки логике агента.

    :param agent: Экземпляр AIAgent для обработки текста.
    :param audio_filepath: Полный путь к аудиофайлу.
    :return: Транскрибированный текст или None, если транскрибация не удалась.
    """
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


# --- Основная функция main.py ---
async def main():
    global local_monitor, ai_agent

    # --- Инициализация основной логики агента ---
    my_llm = OpenRouter_LLM(model_name="qwen/qwen3-coder:free")
    my_memory = ChatHistoryMemory()
    my_vector_store = SupabaseVectorStore(llm_for_embedding=my_llm)
    system_prompt = (
        "Ты очень полезный, дружелюбный и компетентный AI ассистент. "
        "Твоя задача - точно отвечать на вопросы, использовать предоставленные инструменты, когда это уместно, "
        "и извлекать информацию из векторного хранилища для ответов на вопросы о личных данных или контексте. "
        "Всегда старайся быть вежливым и кратким, если это возможно. "
        "Если пользователь задает вопрос, на который может ответить инструмент, предложи его использовать или используй его напрямую, если это очевидно. "
        "Если вопрос касается твоей собственной информации или контекста, используй поиск по векторному хранилищу. "
        "После выполнения инструмента или поиска в векторном хранилище, обобщи результаты и предоставь связный ответ."
        "Будь готов использовать функцию `google_cse_search` для поиска актуальной информации в интернете, когда это необходимо."
        "Особенно используй `google_cse_search`, если вопрос касается текущих событий, статистических данных или общих фактов, отсутствующих в твоей памяти."
        "Если пользователь просит проанализировать файл, ожидай, что содержание файла будет передано тебе как часть user-сообщения, и ты должен его проанализировать."
        "Ты используешь модель OpenRouter для генерации ответов."
        "После выполнения инструмента, **всегда** включай его результат (если он является ответом на вопрос пользователя) в свой финальный ответ. Формулируй ответ пользователю ясно, кратко и дружелюбно, прямо отвечая на его исходный вопрос, используя полученные данные."
    )
    ai_agent = AIAgent(llm=my_llm, memory=my_memory, vector_store=my_vector_store, system_prompt=system_prompt)

    # ai_agent.register_processor('pre_llm', my_custom_pre_llm_processor)
    # ai_agent.register_processor('final_response', my_custom_final_response_processor)

    ai_agent.register_tool(CalculatorTool())
    ai_agent.register_tool(GoogleCSESearchTool())

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

    # --- Аргументы командной строки ---
    parser = argparse.ArgumentParser(description="AI Agent Project with Audio Transcription and Folder Monitoring.")
    parser.add_argument("--audio", type=str,
                        help="Путь к ОДНОМУ аудиофайлу для транскрибации (например, 'path/to/my_audio.mp3').")
    args = parser.parse_args()

    # --- Определяем поддерживаемые расширения для мониторинга (для LocalFileMonitor) ---
    supported_audio_extensions_for_monitor = [
        '.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm', # Поддерживаемые OpenAI API
        '.awb', '.amr' # Те, что pydub может конвертировать
    ]

    # --- Логика запуска мониторинга из config.py ---
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

    # --- Логика обработки одного аудиофайла из командной строки ---
    if args.audio:
        logger.info(f"Обнаружен аргумент --audio. Попытка обработки файла: {args.audio}")
        processed_text = await process_single_audio_input(ai_agent, args.audio)
        if processed_text:
            logger.info("Аудио успешно транскрибировано и текст передан для обработки агентом.")
        else:
            logger.error("Обработка аудио завершилась с ошибкой или не дала результата.")
        return

    # --- Начало интерактивного диалога с Агентом (если нет других CLI аргументов) ---
    print("\n--- Начало интерактивного диалога с Агентом ---")
    print("Введите 'exit' для завершения.")
    print("Чтобы отправить текстовый файл, введите 'file:<путь_к_файлу.txt>' (например, 'file:temp_doc.txt').")
    print("Чтобы отправить аудиофайл, введите 'audio:<путь_к_файлу.mp3>' (например, 'audio:voice_memo.mp3').")
    print("Автоматический мониторинг папки настроен в config.py (если MONITOR_DIRECTORY задан).")
    print("Вы также можете использовать флаг --audio при запуске скрипта для обработки одного файла.")

    while True:
        user_input = input("\nВы: ")
        if user_input.lower() == 'exit':
            print("Завершение диалога.")
            break
        
        if user_input.lower().startswith("file:"):
            file_path = user_input[len("file:"):].strip()
            if os.path.exists(file_path):
                response = await ai_agent.process_message(file_input=file_path)
                print(f"Агент (обработка текстового файла): {response}")
            else:
                print(f"Ошибка: Файл не найден по пути '{file_path}'.")
        elif user_input.lower().startswith("audio:"):
            audio_path = user_input[len("audio:"):].strip()
            await process_single_audio_input(ai_agent, audio_path)
        else:
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
                print(f"tool (результат): Tool ID: {content_obj.get('tool_call_id', 'N/A')}, Output: {json.dumps(content_obj.get('output', ''), indent=2, ensure_ascii=False)}")
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