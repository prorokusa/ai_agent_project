import os
import requests
import json
from typing import Any, Optional
from interfaces.tool import Tool
from dotenv import load_dotenv

load_dotenv()

class GoogleCSESearchTool(Tool):
    """
    Инструмент для выполнения поиска в интернете через Google Custom Search JSON API.
    Требует GOOGLE_CSE_API_KEY и GOOGLE_CSE_ID (Search Engine ID).
    Имеет бесплатный лимит 100 запросов в день.
    """
    def __init__(self):
        super().__init__(name="google_cse_search", description="Выполняет поиск информации в интернете с помощью Google Custom Search. Принимает 'query' (str). Возвращает релевантные результаты веб-поиска в виде сниппетов (кратких описаний).")
        self.api_key = os.getenv("GOOGLE_CSE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if not self.api_key or not self.cse_id:
            print("Предупреждение: GOOGLE_CSE_API_KEY или GOOGLE_CSE_ID не установлены. GoogleCSESearchTool будет использовать фиктивные данные.")

    def execute(self, query: str) -> str:
        if not self.api_key or not self.cse_id:
            # Фиктивный ответ, если API ключ не настроен
            if "столица франции" in query.lower():
                return "Столица Франции - Париж. (Фиктивный ответ, т.к. API ключи Google CSE не настроены)"
            elif "самая высокая гора" in query.lower():
                return "Самая высокая гора в мире - Эверест. (Фиктивный ответ, т.к. API ключи Google CSE не настроены)"
            elif "квантовые компьютеры" in query.lower():
                return "Квантовые компьютеры используют принципы квантовой механики для выполнения вычислений, что позволяет решать некоторые задачи гораздо быстрее классических компьютеров. (Фиктивный ответ)"
            else:
                return f"Результат фиктивного поиска по запросу '{query}': Информация найдена, но не детализирована в этом примере без реального API."

        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": 3 # Количество результатов (макс. 10 за запрос)
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status() # Вызывает HTTPError для плохих ответов (4xx или 5xx)
            data = response.json()
            
            if data and 'items' in data and data['items']:
                snippets = []
                for item in data['items']:
                    title = item.get('title', 'Без названия')
                    snippet = item.get('snippet', 'Без описания')
                    link = item.get('link', '#')
                    snippets.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")
                return "\n---\n".join(snippets)
            elif 'error' in data:
                return f"Google CSE Error: {data['error'].get('message', 'Неизвестная ошибка')}"
            else:
                return f"Google CSE: Не найдено результатов по запросу '{query}'."
        except requests.exceptions.RequestException as e:
            return f"Ошибка при выполнении Google CSE API запроса: {e}"
        except json.JSONDecodeError:
            return "Ошибка: Ответ Google CSE API не является валидным JSON."
        except Exception as e:
            return f"Произошла непредвиденная ошибка при поиске: {e}"