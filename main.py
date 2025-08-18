import os
import json
import asyncio
import logging
from dotenv import load_dotenv # Для загрузки переменных окружения из .env файла
import argparse # Для обработки аргументов командной строки

# --- Настройка логирования для всего приложения ---
# Устанавливает базовый уровень логирования (INFO), чтобы видеть информационные сообщения.
# Формат сообщений включает время, имя логгера, уровень и само сообщение.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Загрузка переменных окружения из .env файла ---
# Это должно быть выполнено в самом начале, чтобы все модули могли получить доступ к API-ключам и другим настройкам.
# Создайте файл .env в корневой директории вашего проекта (рядом с main.py)
# и добавьте в него следующие переменные:
#
# OPENAI_API_KEY="ВАШ_КЛЮЧ_OPENAI_API"
# OPENROUTER_API_KEYS="ВАШ_КЛЮЧ_OPENROUTER_1,ВАШ_КЛЮЧ_OPENROUTER_2" # Можно указать несколько через запятую
# GOOGLE_CSE_API_KEY="ВАШ_КЛЮЧ_GOOGLE_CSE_API"
# GOOGLE_CSE_ID="ВАШ_ИДЕНТИФИКАТОР_ПОИСКОВОЙ_СИСТЕМЫ_GOOGLE"
# SUPABASE_URL="ВАШ_URL_SUPABASE"
# SUPABASE_ANON_KEY="ВАШ_ANON_KEY_SUPABASE"
# LLAMAINDEX_CLOUD_API_KEY="ВАШ_КЛЮЧ_LLAMAINDEX_CLOUD_API" # Для TextExtractorTool (парсинг документов)
# MISTRAL_API_KEY="ВАШ_КЛЮЧ_MISTRAL_AI" # Для TextExtractorTool (OCR изображений)
#
load_dotenv()

# --- Импорты модулей ядра (core) ---
# AIAgent: Центральный класс, управляющий логикой агента.
# AgentContext: Объект для передачи состояния и данных по конвейеру обработки агента.
# AgentProcessor: Тип для функций-обработчиков (плагинов) агента.
from core.agent import AIAgent, AgentContext, AgentProcessor

# --- Импорты реализаций LLM (Large Language Models) ---
# AbstractLLM: Абстрактный базовый класс для LLM (находится в interfaces/llm.py).
# SimpleInferenceLLM: Простая "фиктивная" LLM для демонстрации или тестирования без реальных API.
from implementations.llms.simple_inference_llm import SimpleInferenceLLM
# OpenAI_LLM: Реализация для работы с OpenAI API (gpt-3.5-turbo, gpt-4 и т.д.).
from implementations.llms.openai_llm import OpenAI_LLM
# OpenRouter_LLM: Реализация для работы с OpenRouter API, предоставляющим доступ к различным LLM.
from implementations.llms.openrouter_llm import OpenRouter_LLM

# --- Импорты реализаций инструментов (Tools) ---
# Tool: Абстрактный базовый класс для инструментов (находится в interfaces/tool.py).
# CalculatorTool: Инструмент для выполнения математических вычислений.
from implementations.tools.calculator_tool import CalculatorTool
# GoogleCSESearchTool: Инструмент для поиска информации в интернете через Google Custom Search Engine.
from implementations.tools.web_search_tool import GoogleCSESearchTool
# TextExtractorTool: НОВЫЙ ИНСТРУМЕНТ - для извлечения текста из различных типов файлов (документов, изображений).
# Убедитесь, что этот файл находится по пути: implementations/tools/text_extractor_tool.py
from implementations.tools.text_extractor_tool import TextExtractorTool 

# --- Импорты реализаций памяти (Memory) ---
# Memory: Абстрактный базовый класс для памяти (находится в interfaces/memory.py).
# ChatHistoryMemory: Простая реализация памяти, хранящая историю чата в оперативной памяти.
from implementations.memory.chat_history_memory import ChatHistoryMemory

# --- Импорты реализаций векторных хранилищ (Vector Stores) ---
# VectorStore: Абстрактный базовый класс для векторных хранилищ (находится в interfaces/vector_store.py).
# ChromaDBStore: Реализация для работы с локальной векторной базой данных ChromaDB.
from implementations.vector_stores.chromadb_store import ChromaDBStore
# SupabaseVectorStore: Реализация для работы с Supabase (PostgreSQL + pgvector) в качестве векторного хранилища.
from implementations.vector_stores.supabase_store import SupabaseVectorStore
# SimpleVectorStore: Простая "фиктивная" векторная база данных для демонстрации.
from implementations.vector_stores.simple_vector_store import SimpleVectorStore 


# --- Импорты вспомогательных утилит (utils) ---
# AudioTranscriber: Модуль для транскрибации аудиофайлов в текст с использованием OpenAI Whisper API.
# Требует OPENAI_API_KEY в .env. Для конвертации аудиоформатов необходим FFmpeg.
from utils.audio_transcriber import AudioTranscriber
# LocalFileMonitor: Модуль для отслеживания новых файлов в указанной локальной папке.
from utils.local_file_monitor import LocalFileMonitor

# --- Импорт файла конфигурации проекта ---
# config: Содержит общие настройки проекта, такие как пути к директориям и интервалы.
import config 


# --- ГЛОБАЛЬНЫЕ ЭКЗЕМПЛЯРЫ МОДУЛЕЙ ---
# Эти экземпляры создаются один раз при запуске приложения.

# AudioTranscriber: Модуль для транскрибации аудио.
# Инициализация здесь; если API ключ не найден, будет выведена ошибка.
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

# LocalFileMonitor: Модуль для мониторинга локальной папки.
# Будет инициализирован позже, если MONITOR_DIRECTORY задан в config.py.
local_monitor = None

# ai_agent: Единственный экземпляр AI Агента в этой версии.
# Он будет обрабатывать все запросы и использовать все зарегистрированные инструменты.
ai_agent: AIAgent = None 


# --- Callback-функция для монитора файлов ---
# Эта функция будет вызываться LocalFileMonitor при обнаружении нового аудиофайла.
# Она транскрибирует файл и передает текст основному агенту.
async def handle_monitored_audio_file(audio_filepath: str):
    logger.info(f"Monitor: Обнаружен новый аудиофайл: {audio_filepath}. Начинаю обработку...")
    
    global ai_agent # Обращаемся к глобальной переменной ai_agent
    if ai_agent is None:
        logger.error("Monitor: AI Агент не инициализирован. Невозможно обработать файл.")
        return
    if audio_transcriber is None:
        logger.error("Monitor: Модуль AudioTranscriber не инициализирован. Невозможно транскрибировать аудио.")
        return

    transcribed_text = audio_transcriber.transcribe_audio(audio_filepath)
    
    if transcribed_text:
        logger.info(f"Monitor: Аудио транскрибировано. Передаю текст агенту: '{transcribed_text[:100]}...'")
        # Передача текста текущему ai_agent
        response = await ai_agent.process_message(text_input=transcribed_text)
        logger.info(f"Monitor: Агент ответил на транскрипцию: {response}")
    else:
        logger.warning(f"Monitor: Не удалось транскрибировать аудиофайл: {audio_filepath}.")

# --- Пример пользовательского плагина (middleware) ---
# Эти функции демонстрируют, как можно внедрить дополнительную логику
# на различных этапах конвейера обработки агента.
# Они будут зарегистрированы для ai_agent.

# my_custom_pre_llm_processor: Выполняется перед запросом к LLM.
# Можно использовать для модификации промпта, добавления контекста и т.д.
# Пример: Добавляет текущую дату в промпт.
async def my_custom_pre_llm_processor(agent: AIAgent, context: AgentContext):
    print(f"\n[PLUGIN: pre_llm] Выполняется перед запросом LLM...")
    import datetime
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    context.current_prompt_for_llm = f"Сегодня {current_date}. " + context.current_prompt_for_llm
    context.metadata["current_date"] = current_date
    print(f"[PLUGIN: pre_llm] Добавлена дата в промпт: {context.current_prompt_for_llm[:50]}...")

# my_custom_final_response_processor: Выполняется перед формированием окончательного ответа агента.
# Можно использовать для логирования, форматирования ответа или выполнения завершающих действий.
async def my_custom_final_response_processor(agent: AIAgent, context: AgentContext):
    print(f"\n[PLUGIN: final_response] Получен окончательный ответ агента.")
    if context.processed_successfully:
        print(f"[PLUGIN: final_response] Ответ успешный. Сообщение: {context.final_response[:50]}...")
    else:
        print(f"[PLUGIN: final_response] Ответ содержит ошибку. Ошибка: {context.error_message}")
    print(f"[PLUGIN: final_response] Метаданные контекста: {context.metadata}")


# --- Функция для обработки одного аудиофайла и передачи агенту ---
# Используется, когда файл указывается как аргумент командной строки (--audio).
async def process_single_audio_input(agent: AIAgent, audio_filepath: str) -> str | None:
    """
    Принимает путь к аудиофайлу, транскрибирует его с помощью OpenAI Whisper API
    и передает текст для дальнейшей обработки логике переданного агента.

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
# Точка входа в приложение. Инициализирует и запускает всю логику.
async def main():
    global local_monitor, ai_agent # Объявляем глобальные переменные для модификации

    # --- Инициализация AI Агента ---

    # 1. Выбор LLM:
    # Используйте OpenRouter_LLM для доступа к широкому спектру моделей через единый API.
    # Убедитесь, что OPENROUTER_API_KEYS настроен в .env.
    # Если предпочитаете OpenAI, используйте OpenAI_LLM и настройте OPENAI_API_KEY.
    # Для отладки без API можно использовать SimpleInferenceLLM.
    my_llm = OpenRouter_LLM(model_name="qwen/qwen3-coder:free") 
    # my_llm = OpenAI_LLM(model_name="gpt-3.5-turbo")
    # my_llm = SimpleInferenceLLM()

    # 2. Инициализация памяти:
    # ChatHistoryMemory - простая память в ОЗУ.
    my_memory = ChatHistoryMemory()

    # 3. Инициализация векторного хранилища:
    # SupabaseVectorStore - для постоянного хранения данных в Supabase.
    # Требует SUPABASE_URL и SUPABASE_ANON_KEY в .env, а также настроенной таблицы и функции pgvector.
    # ChromaDBStore - для локального векторного хранилища. Можно указать persist_directory.
    # SimpleVectorStore - простая, для демонстрации, не подходит для продакшена.
    my_vector_store = SupabaseVectorStore(llm_for_embedding=my_llm) 
    # my_vector_store = ChromaDBStore(llm_for_embedding=my_llm, persist_directory="./chroma_db")
    # my_vector_store = SimpleVectorStore(llm_for_embedding=my_llm)
    
    # 4. Системный промпт для агента:
    # Очень важно четко описать его возможности и все доступные инструменты.
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
        "\n\n--- НОВЫЕ ВОЗМОЖНОСТИ ---"
        "Ты также можешь извлекать текстовое содержимое из различных типов файлов (PDF, DOCX, TXT, JPG, PNG, JPEG) с помощью инструмента `text_extractor`. "
        "Используй его, когда пользователь просит проанализировать содержимое файла или извлечь из него текст. "
        "Аргумент: `file_path` - полный путь к файлу. "
        "Пример: `извлеки текст из файла /home/user/document.pdf`"
    )
    # 5. Инициализация основного AI Агента.
    ai_agent = AIAgent(llm=my_llm, memory=my_memory, vector_store=my_vector_store, system_prompt=system_prompt)

    # --- Регистрация плагинов (обработчиков) для агента (опционально) ---
    # Закомментируйте/раскомментируйте по необходимости.
    # ai_agent.register_processor('pre_llm', my_custom_pre_llm_processor)
    # ai_agent.register_processor('final_response', my_custom_final_response_processor)

    # --- Регистрация инструментов для агента ---
    # Эти инструменты доступны агенту для выполнения его задач.
    ai_agent.register_tool(CalculatorTool()) # Для математических операций
    ai_agent.register_tool(GoogleCSESearchTool()) # Для веб-поиска
    ai_agent.register_tool(TextExtractorTool()) # НОВЫЙ ИНСТРУМЕНТ: для извлечения текста из файлов (PDF, DOCX, JPG, PNG и т.д.)

    # --- Загрузка начальных документов в векторное хранилище ---
    # Очищаем векторное хранилище (если используется персистентное) и загружаем начальные документы.
    # Это важно для работы RAG (Retrieval Augmented Generation).
    await my_vector_store.clear() # Очистка Supabase таблицы (или ChromaDB/SimpleVectorStore)
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

    # --- Обработка аргументов командной строки ---
    # Позволяет запускать скрипт с определенными флагами для выполнения специфических задач.
    parser = argparse.ArgumentParser(description="AI Agent Project with Audio Transcription and Folder Monitoring.")
    # --audio: Позволяет указать путь к одному аудиофайлу для транскрибации и обработки.
    parser.add_argument("--audio", type=str,
                        help="Путь к ОДНОМУ аудиофайлу для транскрибации (например, 'path/to/my_audio.mp3').")
    args = parser.parse_args()

    # --- Определяем поддерживаемые расширения для мониторинга (для LocalFileMonitor) ---
    # Список расширений аудиофайлов, которые LocalFileMonitor будет отслеживать
    # и AudioTranscriber сможет обработать.
    supported_audio_extensions_for_monitor = [
        '.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm', # Поддерживаемые OpenAI API
        '.awb', '.amr' # Те, что pydub может конвертировать (требуют FFmpeg)
    ]

    # --- Логика запуска мониторинга локальной директории ---
    # Если MONITOR_DIRECTORY указан в config.py, запускается фоновый мониторинг.
    # В этом режиме интерактивный диалог НЕ запускается автоматически.
    # Если вы хотите, чтобы мониторинг работал параллельно с интерактивным режимом,
    # эту логику нужно было бы вынести в отдельную асинхронную задачу в начале main().
    if config.MONITOR_DIRECTORY:
        if not os.path.isdir(config.MONITOR_DIRECTORY):
            logger.error(f"Ошибка: Путь '{config.MONITOR_DIRECTORY}' из config.py не является существующей директорией. Автоматический мониторинг папки не запущен.")
        else:
            logger.info(f"Настройка MONITOR_DIRECTORY найдена в config.py. Запуск мониторинга папки: {config.MONITOR_DIRECTORY}")
            try:
                local_monitor = LocalFileMonitor(
                    directory_path=config.MONITOR_DIRECTORY,
                    interval=config.MONITOR_INTERVAL_SECONDS,
                    callback_func=handle_monitored_audio_file, # Указываем нашу callback-функцию
                    allowed_extensions=supported_audio_extensions_for_monitor
                )
                monitor_task = local_monitor.run_as_task() # Запуск мониторинга как фоновой задачи asyncio
                logger.info("Мониторинг запущен в фоновом режиме. Нажмите Ctrl+C для остановки.")
                await asyncio.Future() # Держим главный цикл событий активным до отмены
            except ValueError as e:
                logger.error(f"Ошибка при инициализации монитора: {e}. Проверьте путь к директории в config.py.")
            except asyncio.CancelledError:
                logger.info("Мониторинг директории отменен.")
            finally:
                # Гарантируем остановку монитора и ожидание завершения задачи при выходе.
                if local_monitor:
                    local_monitor.stop_monitoring()
                if 'monitor_task' in locals() and not monitor_task.done():
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass
                logger.info("Программа завершена после мониторинга.")
            return # Завершаем выполнение, так как мониторинг был основной задачей.

    # --- Логика обработки одного аудиофайла из командной строки ---
    # Если был указан аргумент --audio, обрабатываем только один файл и завершаем.
    if args.audio:
        logger.info(f"Обнаружен аргумент --audio. Попытка обработки файла: {args.audio}")
        # Используем ai_agent для обработки аудио.
        processed_text = await process_single_audio_input(ai_agent, args.audio)
        if processed_text:
            logger.info("Аудио успешно транскрибировано и текст передан для обработки агентом.")
        else:
            logger.error("Обработка аудио завершилась с ошибкой или не дала результата.")
        return # Завершаем выполнение после обработки одного файла.

    # --- Начало интерактивного диалога с Агентом ---
    # Этот блок кода запускается, если не было аргументов командной строки и не запущен мониторинг.
    print("\n--- Начало интерактивного диалога с Агентом ---")
    print("Введите 'exit' для завершения.")
    print("Чтобы отправить текстовый файл, введите 'file:<путь_к_файлу.txt>' (например, 'file:temp_doc.txt').")
    print("Чтобы отправить аудиофайл, введите 'audio:<путь_к_файлу.mp3>' (например, 'audio:voice_memo.mp3').")
    print(f"Автоматический мониторинг папки настроен в config.py (если MONITOR_DIRECTORY задан).")
    
    # Основной цикл интерактивного взаимодействия.
    while True:
        # Запрос ввода от пользователя.
        user_input = input("\nВы: ")
        
        # Команда для выхода из приложения.
        if user_input.lower() == 'exit':
            print("Завершение диалога.")
            break
        
        # --- ОБРАБОТКА ВВОДА АГЕНТОМ ---
        # Обработка ввода текстового файла.
        if user_input.lower().startswith("file:"):
            file_path = user_input[len("file:"):].strip()
            if os.path.exists(file_path):
                # Передаем путь к файлу агенту.
                response = await ai_agent.process_message(file_input=file_path)
                print(f"Агент (обработка текстового файла): {response}")
            else:
                print(f"Ошибка: Файл не найден по пути '{file_path}'.")
        # Обработка ввода аудиофайла.
        elif user_input.lower().startswith("audio:"):
            audio_path = user_input[len("audio:"):].strip()
            # Вызываем функцию обработки аудио, передавая агента.
            await process_single_audio_input(ai_agent, audio_path)
        # Обработка обычного текстового ввода.
        else:
            # Передаем текстовый ввод агенту.
            response = await ai_agent.process_message(text_input=user_input)
            print(f"Агент: {response}")

    # --- Вывод полной истории чата агента по завершении ---
    # Полезно для отладки и просмотра всего диалога.
    print("\n--- Полная история чата ---")
    for msg in ai_agent.memory.get_history():
        # Специальная обработка для вызовов инструментов и их результатов,
        # чтобы они отображались в более читаемом формате JSON.
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

# --- Запуск основной функции приложения ---
# Используем asyncio.run() для запуска асинхронной функции main().
# Обработка KeyboardInterrupt позволяет gracefully завершить работу при Ctrl+C.
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nПрограмма остановлена пользователем (Ctrl+C).")
        # При прерывании, если монитор был запущен, останавливаем его.
        if local_monitor:
            local_monitor.stop_monitoring()