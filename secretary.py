import asyncio
import os
import logging
from dotenv import load_dotenv
import datetime

# Настройка логирования для всего приложения
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

# Импортируем все необходимые компоненты из вашей архитектуры
from core.agent import AIAgent
from implementations.llms.openai_llm import OpenAI_LLM
# from implementations.llms.openrouter_llm import OpenRouter_LLM
# from implementations.llms.simple_inference_llm import SimpleInferenceLLM

from implementations.memory.chat_history_memory import ChatHistoryMemory

# Импортируем векторные хранилища (выберите одно или оставьте None для отключения RAG)
# from implementations.vector_stores.chromadb_store import ChromaDBStore
# from implementations.vector_stores.simple_vector_store import SimpleVectorStore
# from implementations.vector_stores.supabase_store import SupabaseVectorStore

# Импортируем все инструменты
from implementations.tools.calculator_tool import CalculatorTool
from implementations.tools.web_search_tool import GoogleCSESearchTool
from implementations.tools.text_extractor_tool import TextExtractorTool
from implementations.tools.ftp_audio_processor_tool import FtpAudioProcessorTool 
from implementations.tools.google_calendar_tool import GoogleCalendarTool
from implementations.tools.google_tasks_tool import GoogleTasksTool
from implementations.tools.google_keep_note_tool import GoogleKeepNoteTool

# Импорт модуля конфигурации и FTP-монитора
import config
from utils.ftp_monitor import FtpMonitor


async def main():
    logger.info("Запуск приложения агента в фоновом режиме...")

    # --- 1. Выбор и инициализация LLM ---
    llm = OpenAI_LLM(model_name="gpt-4o") # gpt-4o лучше всего подходит для таких задач
    # llm = OpenRouter_LLM(model_name="google/gemini-2.0-flash-exp:free")
    # llm = SimpleInferenceLLM()

    # --- 2. Инициализация памяти ---
    memory = ChatHistoryMemory()

    # --- 3. Инициализация векторного хранилища (для RAG) ---
    vector_store = None 
    # try:
    #     persist_dir = "./chroma_db_data"
    #     vector_store = ChromaDBStore(llm_for_embedding=llm, collection_name="cadastral_docs", persist_directory=persist_dir)
    #     # await vector_store.clear() 
    #     logger.info(f"ChromaDBStore инициализирован в {persist_dir}")
    # except Exception as e:
    #     logger.warning(f"Не удалось инициализировать ChromaDBStore: {e}. RAG будет отключен.")
    #     vector_store = None
    
    # vector_store = SimpleVectorStore(llm_for_embedding=llm)

    # --- 4. Системный промпт для агента ---
    system_prompt = (
        "Ты - высококвалифицированный личный ассистент и делопроизводитель для кадастрового инженера. "
        "Твоя ключевая функция - это организация и автоматизация рабочего процесса на основе голосовых записей. "
        "Ты постоянно получаешь транскрибированные аудиофайлы звонков или диктовок от FTP-монитора. "
        "**Твоя главная задача: внимательно, ПРОАКТИВНО и БЕЗУСЛОВНО анализировать суть каждого разговора "
        "и извлекать из него ВСЮ полезную, действенную информацию для последующего создания задач, событий "
        "или заметок в Google-сервисах. Твоя цель - максимально разгрузить пользователя от рутины.**\n\n"
        "**Принципы твоей работы:**\n"
        "1.  **Действуй по умолчанию и инферируй:** Если в разговоре есть хоть малейший намек на будущее действие, встречу, срок, важное напоминание, имя, контакт, номер дела – ты должен **АГРЕССИВНО ИНИЦИИРОВАТЬ** создание соответствующей записи в Google-сервисах. Ты должен не просто ждать прямых команд, а **активно интерпретировать и инферировать** намерения из разговора.\n"
        "2.  **ДЕТАЛИЗАЦИЯ И ПОЛНОТА:** Это КРАЙНЕ ВАЖНО. Извлекай **МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ДЕТАЛЕЙ** из транскрипции для заполнения полей. Это включает имена, конкретные объекты (номера участков, адреса), сроки, описания, контактные данные, номера телефонов, кадастровые номера и т.д. **Обязательно включай полный или наиболее релевантный фрагмент транскрибированного текста в поле 'description' для событий и 'notes' для задач.** Это будет исходным материалом для пользователя.\n"
        "3.  **Обработка неполных данных (создавай всегда!):** Если информация для инструмента неполна (например, нет точной даты, времени, но есть 'завтра' или 'на следующей неделе'), ты должен:\n"
        "    *   **Определить дату/время исходя из текущей даты:** Если сказано 'завтра', используй текущую дату + 1 день. Если 'на следующей неделе', используй ближайший понедельник следующей недели. Если время не указано, используй '09:00:00' или '10:00:00' по умолчанию для начала дня. Используй **Текущую дату и время**, предоставленную тебе в начале каждого запроса.\n"
        "    *   **Заполнять отсутствующие поля:** Если какое-то поле (например, 'описание' или 'местоположение') отсутствует, но инструмент требует его, передай пустую строку или 'Не указано'. **НИКОГДА не отказывайся вызвать инструмент из-за отсутствия необязательных полей!**\n"
        "    *   В названии или описании созданной записи (задачи, события) **явно указывай, что некоторые детали требуют уточнения**, например, 'Встреча (время требует уточнения)'.\n\n"
        "**Использование инструментов Google (приоритет от конкретного к общему):**\n"
        "*   **Google Календарь ('google_calendar_tool'):** Используй для встреч, созвонов, вебинаров, дедлайнов, когда есть конкретная или **инферируемая** дата и время (даже если время по умолчанию). **Приоритет: ВЫСОКИЙ**, если есть привязка ко времени/дате.\n"
        "*   **Google Задачи ('google_tasks_tool'):** Используй для поручений, задач, списков дел, когда нужно что-то сделать, и есть срок (даже 'сегодня', 'завтра', 'на этой неделе'). **Приоритет: ВЫСОКИЙ**, если есть дедлайн или конкретное действие.\n"
        "*   **Google Keep ('google_keep_note_tool'):** Используй ТОЛЬКО для важных заметок, идей, контактов, кратких напоминаний, которые **НЕ ИМЕЮТ ПРЯМОЙ ПРИВЯЗКИ** к дате/времени как событие или конкретная задача. Если есть хоть какая-то привязка, используй Календарь или Задачи. **Важно: Интеграция с Google Keep является имитацией из-за отсутствия официального API. Все, что ты создашь там, появится ТОЛЬКО в твоих логах.**\n\n"
        "**Важные контекстные данные:**\n"
        "*   Ты всегда получаешь **ТЕКУЩУЮ ДАТУ И ВРЕМЯ** в начале каждого транскрибированного запроса. Используй ее для точной интерпретации относительных сроков.\n"
        "*   **Имя пользователя, с которым ты работаешь:** Зоткин Алексей Анатольевич. Используй это имя в контексте записей, если это уместно (например, 'Задача для Алексея Зоткина', 'Заметка по разговору с Алексеем Зоткиным').\n"
        "*   Всегда форматируй даты и время в соответствии с ISO 8601 для инструментов (например, '2025-08-21T10:00:00+03:00' для времени, '2025-08-21' для даты).\n\n"
        "**Формат твоего ответа после обработки аудио:**\n"
        "После каждой транскрипции, кратко сообщи, что ты сделал. Например, 'Проанализировал аудио. Создал 1 задачу в Google Задачи, 1 событие в Google Календаре и записал 1 заметку в Google Keep.' Или 'Проанализировал аудио, не обнаружил явных инструкций для создания задач/событий/заметок.' Твой ответ будет виден в логах, поэтому он должен быть информативным и ясным.\n\n"
        "Помимо анализа аудио, ты также можешь выполнять расчеты ('calculator'), искать информацию в интернете ('google_cse_search') и извлекать текст из документов ('text_extractor'), если эти задачи будут явно упомянуты в контексте транскрибированного аудио. Будь точен, вежлив и предоставляй полный ответ, опираясь на полученные данные."
    )

    # --- 5. Инициализация AI Агента ---
    agent = AIAgent(
        llm=llm,
        memory=memory,
        vector_store=vector_store,
        system_prompt=system_prompt,
        max_tool_iterations=5
    )

    # --- 6. Регистрация всех инструментов ---
    logger.info("Регистрация инструментов агента...")
    agent.register_tool(CalculatorTool())
    agent.register_tool(GoogleCSESearchTool())
    agent.register_tool(TextExtractorTool())
    agent.register_tool(FtpAudioProcessorTool()) 
    agent.register_tool(GoogleCalendarTool())
    agent.register_tool(GoogleTasksTool())
    agent.register_tool(GoogleKeepNoteTool())
    logger.info("Все инструменты зарегистрированы.")

    # --- 7. Инициализация и запуск FTP-мониторинга как фоновой задачи ---
    ftp_monitor = FtpMonitor(agent)
    monitoring_task = None
    if ftp_monitor.enabled:
        await ftp_monitor._load_initial_ftp_files() 
        monitoring_task = ftp_monitor.run_as_task()
        logger.info("FTP-мониторинг запущен в фоновом режиме.")
    else:
        logger.warning("FTP-мониторинг не запущен (проверьте настройки в config.py и .env).")

    logger.info("Приложение агента запущено в фоновом режиме. Ожидание FTP-файлов...")
    logger.info("Нажмите Ctrl+C для завершения.")

    try:
        await asyncio.Future() 
    except asyncio.CancelledError:
        logger.info("Основной цикл приложения отменен.")
    finally:
        if monitoring_task:
            ftp_monitor.stop_monitoring()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Приложение завершено.")


if __name__ == "__main__":
    print("\n--- Инструкции по настройке и запуску ---")
    print("1. **Файл `.env` (для чувствительных данных):**")
    print("   Разместите файл `.env` в корневой директории вашего проекта.")
    print("   Содержимое '.env' (замените на ваши реальные данные):")
    print("""
# OPENAI API KEY
OPENAI_API_KEY=sk-...

# MISTRAL AI API KEY
MISTRAL_API_KEY=your_mistral_api_key_here

# LLAMAINDEX CLOUD API KEY
LLAMAINDEX_CLOUD_API_KEY=your_llamaindex_cloud_api_key_here

# Google Custom Search API Keys
GOOGLE_CSE_API_KEY=your_google_cse_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here

# OpenRouter API Keys (если используется OpenRouter_LLM)
# OPENROUTER_API_KEYS=sk-or-v1-...,sk-or-v1-...

# FTP Credentials (для ftp_monitor.py)
# !!! Обязательно замените на ваши реальные данные FTP для тестирования !!!
FTP_HOST=your_ftp_host.com
FTP_USER=your_ftp_username
FTP_PASSWORD=your_ftp_password_here

# Google API Client ID & Secret (для Google Calendar/Tasks/Keep)
# Получите из Google Cloud Console (OAuth 2.0 Client ID for Desktop app)
GOOGLE_CLIENT_ID=ВАШ_CLIENT_ID_ЗДЕСЬ.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=ВАШ_CLIENT_SECRET_ЗДЕСЬ

# Имя пользователя/владельца агента для включения в записи
AGENT_OWNER_NAME="Зоткин Алексей Анатольевич"
    """)
    print("\n2. **Файл `config.py` (для общих настроек):**")
    print("   Убедитесь, что `config.py` содержит необходимые настройки, особенно для FTP-мониторинга:")
    print("""
# Пример из config.py:
FTP_MONITOR_ENABLED = True # Измените на True для активации
FTP_MONITOR_INTERVAL_SECONDS = 30 # Интервал проверки
FTP_MONITOR_REMOTE_PATH = "/Documents/CallRecord/" # Путь на FTP (убедитесь, что это правильный путь на вашем FTP)
FTP_MONITOR_LOCAL_DOWNLOAD_DIR = "./temp_ftp_downloads/" # Локальная папка
FTP_MONITOR_CLEAR_REMOTE_AFTER_PROCESSING = False # Удалять ли с FTP (будьте осторожны)
FTP_MONITOR_ALLOWED_EXTENSIONS = [".mp3", ".wav", ".awb", ".amr"] # Разрешенные расширения

GOOGLE_CALENDAR_ENABLED=True # Измените на True для активации
GOOGLE_TASKS_ENABLED=True    # Измените на True для активации
GOOGLE_KEEP_ENABLED=False    # Измените на True для активации (но это заглушка)
    """)
    print("\n3. **Установка зависимостей:**")
    print("   pip install python-dotenv openai aioftp pydub")
    print("   pip install google-api-python-client google-auth-oauthlib google-auth")
    print("   pip install chromadb (если используете ChromaDB)")
    print("\n4. **Настройка Google Cloud Console (для GOOGLE_CALENDAR_ENABLED/GOOGLE_TASKS_ENABLED):**")
    print("   - Включите 'Google Calendar API' и 'Google Tasks API'.")
    print("   - Создайте 'OAuth 2.0 Client ID' (тип 'Desktop app').")
    print("   - В 'Authorized redirect URIs' добавьте 'http://localhost'.")
    print("   - При первом запуске скрипт выведет URL в консоль. Откройте этот URL в браузере на ВАШЕМ WINDOWS-ХОСТЕ, авторизуйтесь, скопируйте КОД АВТОРИЗАЦИИ из браузера и вставьте его обратно в консоль WSL.")
    print("\n--- Запуск агента ---")
    print("Выполните команду: python main.py")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nЗавершение работы агента по запросу пользователя (Ctrl+C).")
    except Exception as e:
        logger.critical(f"Критическая ошибка в главном цикле приложения: {e}", exc_info=True)