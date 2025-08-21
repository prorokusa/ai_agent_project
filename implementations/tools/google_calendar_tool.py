import os
import asyncio
import pickle
from typing import Any, List, Dict, Optional

from interfaces.tool import Tool

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError

# Объединенные области доступа для всех Google API, которые вы можете использовать.
SCOPES = [
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/tasks'
]

class GoogleCalendarTool(Tool):
    """
    Инструмент для создания событий в Google Календаре.
    ТРЕБУЕТ НАСТРОЙКИ АВТОРИЗАЦИИ GOOGLE API (client_id и client_secret в .env).
    """
    def __init__(self):
        super().__init__(
            name="google_calendar_tool",
            description="Создает новое событие в Google Календаре. Принимает 'summary' (название события), 'start_time' (время начала в формате ISO 8601, например, '2025-08-21T10:00:00+03:00' для события 21 августа 2025 года в 10:00 по московскому времени), 'end_time' (время окончания в формате ISO 8601), опционально 'description' (подробное описание), 'location' (место проведения) и 'attendees' (список email-адресов участников)."
        )
        self.service = None 
        self.is_enabled = os.getenv("GOOGLE_CALENDAR_ENABLED", "false").lower() == "true"
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

        if not self.is_enabled:
             print("ВНИМАНИЕ: GoogleCalendarTool не активен. Установите GOOGLE_CALENDAR_ENABLED=true в .env для его использования.")
        elif not self.client_id or not self.client_secret:
             print("ВНИМАНИЕ: GOOGLE_CLIENT_ID или GOOGLE_CLIENT_SECRET не найдены в .env. GoogleCalendarTool будет нефункционален.")
             self.is_enabled = False # Отключаем, если нет учетных данных
        else:
             print("GoogleCalendarTool активирован. Для его работы потребуется аутентификация.")

    # Синхронный метод для аутентификации, который будет вызываться через run_in_executor
    def _authenticate(self):
        creds = None
        if os.path.exists('token.pickle'):
            try:
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
            except Exception as e:
                print(f"Ошибка при загрузке token.pickle: {e}. Попробуем получить новые учетные данные.")
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except RefreshError as e:
                    print(f"Ошибка при обновлении токена Google Calendar: {e}. Требуется повторная авторизация.")
                    return "Error: Could not refresh Google Calendar token. Re-authorization required."
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
                        "redirect_uris": ["http://localhost"] # 'http://localhost' still works for CLI flow
                    }
                }
                
                print("\n--- Google Авторизация (WSL/CLI) ---")
                print("Пожалуйста, откройте следующую ссылку в ВАШЕМ БРАУЗЕРЕ (на Windows):")
                flow = InstalledAppFlow.from_client_secrets_dict(client_config, SCOPES)
                
                # --- ИЗМЕНЕНИЕ: ИСПОЛЬЗУЕМ run_console() ВМЕСТО run_local_server() ---
                creds = flow.run_console() 
                print("--- Авторизация завершена ---")
            
            try:
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            except Exception as e:
                print(f"Предупреждение: Не удалось сохранить token.pickle: {e}")

        self.service = build('calendar', 'v3', credentials=creds)
        return "Authentication successful."

    async def execute(self, summary: str, start_time: str, end_time: str,
                      description: Optional[str] = None, location: Optional[str] = None,
                      attendees: Optional[List[str]] = None) -> str:
        
        if not self.is_enabled:
             return "Ошибка: Интеграция с Google Календарем отключена. Включите ее в настройках (`GOOGLE_CALENDAR_ENABLED=true`)."
        
        if not self.service:
            loop = asyncio.get_running_loop()
            auth_result = await loop.run_in_executor(None, self._authenticate)
            if "Error" in auth_result:
                return auth_result
        
        event = {
            'summary': summary,
            'location': location,
            'description': description,
            'start': {'dateTime': start_time, 'timeZone': 'Europe/Moscow'},
            'end': {'dateTime': end_time, 'timeZone': 'Europe/Moscow'},
            'attendees': [{'email': email} for email in attendees] if attendees else [],
            'reminders': {'useDefault': True},
        }

        try:
            loop = asyncio.get_running_loop()
            event = await loop.run_in_executor(
                None,
                lambda: self.service.events().insert(calendarId='primary', body=event).execute()
            )
            return f"Событие '{summary}' успешно создано в Google Календаре. Ссылка: {event.get('htmlLink')}"
        except Exception as e:
            return f"Ошибка при создании события в Google Календаре: {e}"