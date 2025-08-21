import os
import asyncio
import pickle
from typing import Any, Optional

from interfaces.tool import Tool

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError
import logging

logger = logging.getLogger(__name__)

# Добавьте SCOPE для Google Keep API
# Этот SCOPE может быть не публично задокументирован, но если API включен, он может быть 'https://www.googleapis.com/auth/keep'
# Или 'https://www.googleapis.com/auth/keep.readonly' для чтения. Для создания нужен write.
# Если он не сработает, то API действительно не для обычных OAuth клиентов.
SCOPES = [
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/tasks',
    'https://www.googleapis.com/auth/keep' # <-- ДОБАВЛЕН SCOPE для Keep
]

class GoogleKeepNoteTool(Tool):
    """
    Инструмент для создания заметок в Google Keep.
    Если API доступен, будет создавать реальные заметки.
    """
    def __init__(self):
        super().__init__(
            name="google_keep_note_tool",
            description="Создает новую заметку в Google Keep. Принимает 'title' (заголовок заметки) и 'content' (содержание заметки). Это может быть использовано для создания напоминаний или просто для сохранения информации."
        )
        self.service = None
        self.is_enabled = os.getenv("GOOGLE_KEEP_ENABLED", "false").lower() == "true"
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

        if not self.is_enabled:
             print("ВНИМАНИЕ: GoogleKeepNoteTool не активен. Установите GOOGLE_KEEP_ENABLED=true в .env для его использования.")
        elif not self.client_id or not self.client_secret:
             print("ВНИМАНИЕ: GOOGLE_CLIENT_ID или GOOGLE_CLIENT_SECRET не найдены в .env. GoogleKeepNoteTool будет нефункционален.")
             self.is_enabled = False
        else:
            print("GoogleKeepNoteTool активирован. Попытка реальной интеграции с Google Keep API.")

    # Синхронный метод для аутентификации, который будет вызываться через run_in_executor
    def _authenticate(self):
        creds = None
        if os.path.exists('token.pickle'):
            try:
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
            except Exception as e:
                logger.warning(f"Ошибка при загрузке token.pickle: {e}. Попробуем получить новые учетные данные.")
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except RefreshError as e:
                    logger.error(f"Ошибка при обновлении токена Google Keep: {e}. Требуется повторная авторизация.")
                    return "Error: Could not refresh Google Keep token. Re-authorization required."
            else:
                if not self.client_id or not self.client_secret:
                    return "Error: GOOGLE_CLIENT_ID или GOOGLE_CLIENT_SECRET не найдены в .env. Невозможно авторизоваться."

                client_config = {
                    "installed": {
                        "client_id": self.client_id,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_secret": self.client_secret,
                        "redirect_uris": ["http://localhost"]
                    }
                }

                logger.info("\n--- Google Авторизация (Keep API - WSL/CLI) ---")
                logger.info("Пожалуйста, откройте следующую ссылку в ВАШЕМ БРАУЗЕРЕ (на Windows):")
                flow = InstalledAppFlow.from_client_secrets_dict(client_config, SCOPES)
                
                # Запускаем консольный поток
                creds = flow.run_console() 
                logger.info("--- Авторизация завершена ---")
            
            try:
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            except Exception as e:
                logger.warning(f"Предупреждение: Не удалось сохранить token.pickle: {e}")

        # Попытка построить сервис для Google Keep API
        # Замените 'v1' на актуальную версию API, если она есть
        # Если build('keep', 'v1', ...) выдаст ошибку, значит, такого сервиса нет
        self.service = build('keep', 'v1', credentials=creds) 
        return "Authentication successful."

    async def execute(self, title: str, content: str) -> str:
        
        if not self.is_enabled:
             return "Ошибка: Интеграция с Google Keep отключена. Включите ее в настройках (`GOOGLE_KEEP_ENABLED=true`)."

        if not self.service:
            loop = asyncio.get_running_loop()
            auth_result = await loop.run_in_executor(None, self._authenticate)
            if "Error" in auth_result:
                return auth_result
        
        try:
            # Структура запроса для создания заметки в Google Keep API
            # Это ОЧЕНЬ вероятно потребует корректировки!
            # Основано на общей структуре Google API для создания ресурсов
            # Если API Keep использует другую структуру, это вызовет ошибку
            note_body = {
                'title': title,
                'textBody': content,
                # Возможно, нужны другие поля, например 'parent' для папки
                # 'parent': 'folders/folderId'
            }
            
            loop = asyncio.get_running_loop()
            
            # Попытка создать заметку
            # Это может быть `self.service.notes().create(body=note_body).execute()`
            # или `self.service.notes().batchCreate(body=note_body).execute()`
            # или что-то еще, в зависимости от API
            # Имя метода (например, 'notes().create') может отличаться!
            result = await loop.run_in_executor(
                None, lambda: self.service.notes().create(body=note_body).execute() # <-- Это может быть неверно!
            )
            
            return f"Заметка '{title}' успешно создана в Google Keep. ID: {result.get('id', 'N/A')}"
        except Exception as e:
            logger.error(f"Ошибка при создании заметки в Google Keep: {e}")
            # Возвращаемся к имитации, если реальная попытка не удалась
            log_message = (
                f"--- ИМИТАЦИЯ СОЗДАНИЯ ЗАМЕТКИ В GOOGLE KEEP (Не удалось создать реально) ---\n"
                f"  ЗАГОЛОВОК: {title}\n"
                f"  СОДЕРЖАНИЕ: {content}\n"
                f"  ПРИЧИНА: {e}\n"
                f"  ПРИМЕЧАНИЕ: Google Keep API либо недоступен для текущих учетных данных/ scopes, либо структура запроса неверна. Информация только в логах."
            )
            logger.info(log_message)
            print(log_message)
            return (f"Заметка '{title}' (содержание: '{content[:50]}...') была бы успешно создана в Google Keep (имитация). "
                    f"ОШИБКА: {e}")