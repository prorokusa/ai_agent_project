# implementations/tools/text_extractor_tool.py
import os
import httpx # Для асинхронных HTTP-запросов
import asyncio
import logging
import json # Импорт для логирования JSON
from typing import Any, Union, Optional
from interfaces.tool import Tool
from dotenv import load_dotenv

load_dotenv() # Загружаем переменные окружения

logger = logging.getLogger(__name__)

class TextExtractorTool(Tool):
    """
    Инструмент для извлечения текстового содержимого из различных типов файлов
    с использованием LlamaIndex Cloud API для текстовых документов и Mistral AI OCR API для изображений и PDF.
    
    Принимает 'file_path' (строка) - путь к файлу, и опционально 'language' (строка) для OCR.
    Возвращает извлеченный текст в формате Markdown.
    """
    def __init__(self):
        super().__init__(
            name="text_extractor",
            description="Извлекает текстовое содержимое из различных типов файлов. Для документов (DOC/DOCX/TXT/XLS/XLSX) использует LlamaIndex Cloud API. Для изображений (JPG/PNG/JPEG) и PDF-файлов использует Mistral AI OCR API. Принимает 'file_path' (строка) - путь к файлу, и опционально 'language' (строка, например 'en' или 'ru') для улучшения распознавания текста на изображениях и PDF."
        )
        # Получаем API ключи из переменных окружения
        self.llamaindex_api_key = os.getenv("LLAMAINDEX_CLOUD_API_KEY")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY") 

        if not self.llamaindex_api_key:
            logger.warning("LLAMAINDEX_CLOUD_API_KEY не установлен. Функции извлечения текста из документов (DOC/DOCX/TXT/XLS/XLSX) будут недоступны.")
        if not self.mistral_api_key:
            logger.warning("MISTRAL_API_KEY не установлен. Функции OCR для изображений и PDF будут недоступны.")

        self.llamaindex_base_url = "https://api.cloud.llamaindex.ai/api/v1/parsing"
        self.mistral_base_url = "https://api.mistral.ai/v1"

        # --- ОБНОВЛЕННЫЕ СПИСКИ ПОДДЕРЖИВАЕМЫХ РАСШИРЕНИЙ ---
        # Расширения, которые обрабатываются LlamaIndex (текстовые/офисные)
        self.supported_llamaindex_extensions = ['.doc', '.docx', '.txt', '.xls', '.xlsx']
        # Расширения, которые обрабатываются Mistral AI OCR (изображения и PDF)
        self.supported_mistral_ocr_extensions = ['.pdf', '.jpg', '.png', '.jpeg']

    async def _llamaindex_extract(self, file_path: str) -> str:
        """
        Извлекает текст из документа с использованием LlamaIndex Cloud API.
        """
        if not self.llamaindex_api_key:
            return "Ошибка: LLAMAINDEX_CLOUD_API_KEY не настроен для извлечения документов."

        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.llamaindex_api_key}"
            }

            # 1. Upload file
            logger.info(f"Загружаю файл {file_path} в LlamaIndex (для DOCX/TXT/XLSX)...")
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (os.path.basename(file_path), f.read(), "application/octet-stream")}
                    upload_response = await client.post(
                        f"{self.llamaindex_base_url}/upload",
                        headers=headers,
                        files=files
                    )
                upload_response.raise_for_status()
                job_id = upload_response.json().get("id")
                if not job_id:
                    return f"Ошибка загрузки файла в LlamaIndex: Не получен Job ID. Ответ: {upload_response.text}"
                logger.info(f"Файл загружен, Job ID: {job_id}. Ожидаю обработки...")
            except httpx.HTTPStatusError as e:
                return f"Ошибка HTTP при загрузке файла в LlamaIndex: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"Ошибка при загрузке файла в LlamaIndex: {e}"

            # 2. Poll for status
            status = "PENDING"
            max_retries = 30 
            retries = 0
            while status in ["PENDING", "RUNNING"] and retries < max_retries:
                await asyncio.sleep(10) # Ожидание
                retries += 1
                logger.info(f"Проверка статуса обработки Job ID {job_id} (попытка {retries}/{max_retries})...")
                try:
                    status_response = await client.get(
                        f"{self.llamaindex_base_url}/job/{job_id}",
                        headers=headers
                    )
                    status_response.raise_for_status()
                    status = status_response.json().get("status")
                    logger.info(f"Текущий статус: {status}")
                except httpx.HTTPStatusError as e:
                    return f"Ошибка HTTP при получении статуса LlamaIndex Job ID {job_id}: {e.response.status_code} - {e.response.text}"
                except Exception as e:
                    return f"Ошибка при получении статуса LlamaIndex Job ID {job_id}: {e}"

            if status != "SUCCESS":
                return f"Ошибка: Обработка файла LlamaIndex завершилась со статусом '{status}' после {retries} попыток или не удалось получить окончательный статус."

            # 3. Get result
            logger.info(f"Получаю результат обработки для Job ID {job_id}...")
            try:
                result_response = await client.get(
                    f"{self.llamaindex_base_url}/job/{job_id}/result/markdown",
                    headers=headers
                )
                result_response.raise_for_status()
                result_data = result_response.json()
                
                markdown_text = result_data.get("markdown")
                if markdown_text is None and "pages" in result_data and isinstance(result_data["pages"], list):
                    markdown_text = "\n\n---\n\n".join([p.get("markdown", "") for p in result_data["pages"]])
                
                if markdown_text is None:
                    return f"Ошибка: Не получен извлеченный текст из LlamaIndex. Ответ: {result_response.text}"
                
                logger.info(f"Текст успешно извлечен из файла {file_path} (LlamaIndex).")
                return markdown_text
            except httpx.HTTPStatusError as e:
                return f"Ошибка HTTP при получении результата LlamaIndex Job ID {job_id}: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"Ошибка при получении результата LlamaIndex Job ID {job_id}: {e}"

    async def _mistral_ocr_extract(self, file_path: str, language: Optional[str] = None) -> str:
        """
        Извлекает текст из изображения или PDF с использованием Mistral AI OCR API.
        Добавлен параметр language для настройки языка OCR (но его использование может зависеть от версии API).
        """
        if not self.mistral_api_key:
            return "Ошибка: MISTRAL_API_KEY не настроен для OCR."

        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}"
            }

            # 1. Upload to Mistral (for OCR purpose)
            logger.info(f"Загружаю файл {file_path} в Mistral AI для OCR (изображение/PDF)...")
            try:
                with open(file_path, "rb") as f:
                    files = {
                        "file": (os.path.basename(file_path), f.read(), "application/octet-stream"),
                        "purpose": (None, "ocr") 
                    }
                    upload_response = await client.post(
                        f"{self.mistral_base_url}/files",
                        headers=headers,
                        files=files
                    )
                upload_response.raise_for_status()
                file_id = upload_response.json().get("id")
                if not file_id:
                    return f"Ошибка загрузки файла в Mistral AI: Не получен File ID. Ответ: {upload_response.text}"
                logger.info(f"Файл загружен в Mistral AI, File ID: {file_id}.")
            except httpx.HTTPStatusError as e:
                return f"Ошибка HTTP при загрузке файла в Mistral AI: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"Ошибка при загрузке файла в Mistral AI: {e}"

            # 2. Get signed URL
            logger.info(f"Получаю подписанный URL для File ID {file_id}...")
            try:
                signed_url_response = await client.get(
                    f"{self.mistral_base_url}/files/{file_id}/url",
                    headers=headers,
                    params={"expiry": 24} 
                )
                signed_url_response.raise_for_status()
                signed_url = signed_url_response.json().get("url")
                if not signed_url:
                    return f"Ошибка получения подписанного URL Mistral AI: Не получен URL. Ответ: {signed_url_response.text}"
                logger.info(f"Подписанный URL получен: {signed_url[:50]}...")
            except httpx.HTTPStatusError as e:
                return f"Ошибка HTTP при получении подписанного URL Mistral AI: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"Ошибка при получении подписанного URL Mistral AI: {e}"

            # 3. Get OCR result
            logger.info("Отправляю запрос на OCR в Mistral AI...")
            try:
                ocr_payload = {
                    "model": "mistral-ocr-latest",
                    "document": {
                        "type": "document_url",
                        "document_url": signed_url
                    },
                    "include_image_base64": False 
                }
                # --- ИСПРАВЛЕНИЕ: Закомментируем добавление параметра языка ---
                # Согласно сообщению об ошибке Mistral AI, параметр 'language' в теле запроса
                # для 'mistral-ocr-latest' не разрешен ("Extra inputs are not permitted").
                # Модель, вероятно, автоматически определяет язык.
                # if language:
                #     ocr_payload["language"] = language
                #     logger.info(f"Использую язык OCR: {language}")
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                ocr_response = await client.post(
                    f"{self.mistral_base_url}/ocr",
                    headers=headers,
                    json=ocr_payload
                )
                ocr_response.raise_for_status()
                ocr_result = ocr_response.json()
                
                # --- ЛОГИРОВАНИЕ СЫРОГО ОТВЕТА ДЛЯ ОТЛАДКИ ---
                logger.info(f"Сырой ответ Mistral OCR для '{os.path.basename(file_path)}': {json.dumps(ocr_result, indent=2, ensure_ascii=False)}")

                extracted_text = ""
                # Mistral OCR может возвращать текст в разных структурах, подстраиваемся
                if "pages" in ocr_result and isinstance(ocr_result["pages"], list):
                    extracted_text = "\n\n---\n\n".join([page.get("text", "") for page in ocr_result["pages"]])
                elif "text" in ocr_result.get("result", {}): # Запасной вариант для прямого текстового поля
                    extracted_text = ocr_result["result"]["text"]
                
                if not extracted_text.strip(): # Проверяем на пустой или только пробелы
                    logger.warning(f"Mistral OCR вернул пустой или нераспознанный текст для файла {file_path}. Проверьте содержимое файла.")
                    return f"Извлечение текста не дало результатов. Возможно, файл пуст или не содержит распознаваемого текста."
                
                logger.info(f"Текст успешно извлечен из файла {file_path} (Mistral OCR).")
                return extracted_text

            except httpx.HTTPStatusError as e:
                return f"Ошибка HTTP при получении результата OCR от Mistral AI: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"Ошибка при получении результата OCR от Mistral AI: {e}"

    # --- ИЗМЕНЕНИЕ В МЕТОДЕ EXECUTE ---
    # Теперь он принимает опциональный аргумент language.
    async def execute(self, file_path: str, language: Optional[str] = None) -> str:
        """
        Основной метод выполнения инструмента TextExtractor.
        Определяет тип файла и вызывает соответствующую функцию извлечения.
        """
        if not os.path.exists(file_path):
            return f"Ошибка: Файл не найден по пути '{file_path}'."
        if not os.path.isfile(file_path):
            return f"Ошибка: Указанный путь '{file_path}' не является файлом."

        file_extension = os.path.splitext(file_path)[1].lower()

        # --- ОБНОВЛЕННАЯ ЛОГИКА МАРШРУТИЗАЦИИ ---
        if file_extension in self.supported_mistral_ocr_extensions:
            # PDF и изображения идут в Mistral OCR
            # Обратите внимание, что аргумент 'language' по-прежнему передается сюда,
            # но внутри _mistral_ocr_extract он теперь игнорируется для 'mistral-ocr-latest'.
            # Это позволяет сохранить гибкость, если Mistral AI добавит поддержку языка позже.
            return await self._mistral_ocr_extract(file_path, language)
        elif file_extension in self.supported_llamaindex_extensions:
            # DOC/DOCX/TXT/XLS/XLSX идут в LlamaIndex
            return await self._llamaindex_extract(file_path)
        else:
            return (f"Ошибка: Формат файла '{file_extension}' не поддерживается инструментом TextExtractor. "
                    f"Поддерживаемые форматы документов: {', '.join(self.supported_llamaindex_extensions)}. "
                    f"Поддерживаемые форматы изображений/PDF для OCR: {', '.join(self.supported_mistral_ocr_extensions)}.")