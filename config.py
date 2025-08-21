import os
# config.py

# --- Настройки модуля транскрибации (OpenAI Whisper API) ---
# OPENAI_API_KEY загружается из .env файла и не хранится здесь напрямую.
# В этом разделе могут быть другие настройки, если они появятся для AudioTranscriber.

# --- Настройки мониторинга локальных файлов ---
# Путь к директории для отслеживания новых аудиофайлов.
# Оставьте пустой строкой "", если вы не хотите использовать автоматический мониторинг папки.
MONITOR_DIRECTORY = "" # "./monitored_audio_files" # Пример: папка в корне проекта
# MONITOR_DIRECTORY = "/home/alexey/my_audio_inputs" # Пример: абсолютный путь

# Интервал проверки папки на новые файлы в секундах.
MONITOR_INTERVAL_SECONDS = 5


# --- Настройки LLM агента (пример) ---
# Модель для OpenRouter_LLM
# LLM_MODEL_NAME = "qwen/qwen3-coder:free"
# LLM_MODEL_NAME = "openai/gpt-3.5-turbo" # Если используете OpenAI_LLM
# Добавьте сюда любые другие настройки, которые вы хотите сделать конфигурируемыми,
# например, название таблицы Supabase, API ключи для Google CSE (если не в .env), и т.д.

# --- Настройки FTP-мониторинга ---
# Активировать ли фоновый FTP-мониторинг? (true/false)
FTP_MONITOR_ENABLED = os.getenv("FTP_MONITOR_ENABLED", "true").lower() == "true"
# Интервал проверки FTP-сервера в секундах
FTP_MONITOR_INTERVAL_SECONDS = int(os.getenv("FTP_MONITOR_INTERVAL_SECONDS", "60")) 
# Удаленный путь на FTP-сервере, который нужно мониторить
FTP_MONITOR_REMOTE_PATH = os.getenv("FTP_MONITOR_REMOTE_PATH", "/Documents/CallRecord/") 
# Локальная директория для временного скачивания файлов
FTP_MONITOR_LOCAL_DOWNLOAD_DIR = os.getenv("FTP_MONITOR_LOCAL_DOWNLOAD_DIR", "./temp_ftp_downloads/") 
# Удалять ли файлы с FTP после успешной обработки? (Используйте с осторожностью!)
FTP_MONITOR_CLEAR_REMOTE_AFTER_PROCESSING = os.getenv("FTP_MONITOR_CLEAR_REMOTE_AFTER_PROCESSING", "false").lower() == "true"
# Список расширений аудиофайлов для отслеживания (нижний регистр)
FTP_MONITOR_ALLOWED_EXTENSIONS = [
    ".mp3", ".wav", ".awb", ".amr", ".flac", ".m4a", ".mp4", 
    ".mpeg", ".mpga", ".oga", ".ogg", ".webm"
]

# --- Общие настройки Google API ---
# Включить ли инструменты Google Calendar/Tasks/Keep?
GOOGLE_CALENDAR_ENABLED = os.getenv("GOOGLE_CALENDAR_ENABLED", "false").lower() == "true"
GOOGLE_TASKS_ENABLED = os.getenv("GOOGLE_TASKS_ENABLED", "false").lower() == "true"
GOOGLE_KEEP_ENABLED = os.getenv("GOOGLE_KEEP_ENABLED", "false").lower() == "true"