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

# SCOPES определены в google_calendar_tool.py, чтобы использовать один token.pickle
SCOPES = [
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/tasks'
]

class GoogleTasksTool(Tool):
    """
    Инструмент для создания задач в Google Tasks.
    ТРЕБУЕТ НАСТРОЙКИ АВТОРИЗАЦИИ GOOGLE API (client_id и client_secret в .env).
    """
    def __init__(self):
        super().__init__(
            name="google_tasks_tool",
            description="Создает новую задачу в Google Tasks. Принимает 'title' (название задачи), опционально 'due_date' (срок выполнения задачи в формате ISO 8601 даты, например, '2025-08-21'), 'notes' (подробное описание задачи) и 'task_list_id' (ID списка задач, куда добавить задачу; по умолчанию используется основной список)."
        )
        self.service = None 
        self.is_enabled = os.getenv("GOOGLE_TASKS_ENABLED", "false").lower() == "true"
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

        if not self.is_enabled:
             print("ВНИМАНИЕ: GoogleTasksTool не активен. Установите GOOGLE_TASKS_ENABLED=true в .env для его использования.")
        elif not self.client_id or not self.client_secret:
             print("ВНИМАНИЕ: GOOGLE_CLIENT_ID или GOOGLE_CLIENT_SECRET не найдены в .env. GoogleTasksTool будет нефункционален.")
             self.is_enabled = False
        else:
            print("GoogleTasksTool активирован. Для его работы потребуется аутентификация.")

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
                    print(f"Ошибка при обновлении токена Google Tasks: {e}. Требуется повторная авторизация.")
                    return "Error: Could not refresh Google Tasks token. Re-authorization required."
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

        self.service = build('tasks', 'v1', credentials=creds)
        return "Authentication successful."

    async def execute(self, title: str, due_date: Optional[str] = None,
                      notes: Optional[str] = None, task_list_id: Optional[str] = None) -> str:
        
        if not self.is_enabled:
             return "Ошибка: Интеграция с Google Tasks отключена. Включите ее в настройках (`GOOGLE_TASKS_ENABLED=true`)."

        if not self.service:
            loop = asyncio.get_running_loop()
            auth_result = await loop.run_in_executor(None, self._authenticate)
            if "Error" in auth_result:
                return auth_result

        task_body = {'title': title}
        if notes:
            task_body['notes'] = notes
        if due_date:
            task_body['due'] = f"{due_date}T00:00:00Z" 
        
        try:
            loop = asyncio.get_running_loop()
            
            if not task_list_id:
                lists_result = await loop.run_in_executor(None, lambda: self.service.tasklists().list().execute())
                default_list = next((item for item in lists_result.get('items', []) if item.get('title') == 'My Tasks'), None)
                if default_list:
                    task_list_id = default_list['id']
                elif lists_result.get('items'):
                    task_list_id = lists_result['items'][0]['id']
                else:
                    return "Ошибка: Не удалось найти ни одного списка задач в Google Tasks и не был предоставлен task_list_id."

            result = await loop.run_in_executor(
                None, lambda: self.service.tasks().insert(tasklist=task_list_id, body=task_body).execute()
            )
            return f"Задача '{title}' успешно создана в Google Tasks. Ссылка: https://tasks.google.com/tasks/a/default/all/{result.get('id')}"
        except Exception as e:
            return f"Ошибка при создании задачи в Google Tasks: {e}"