# implementations/tools/structured_data_extractor_tool.py
import os
import json
import logging
from typing import Any, Union, Optional, List, Dict

from interfaces.tool import Tool
from interfaces.llm import AbstractLLM
from interfaces.vector_store import VectorStore # Нужен доступ к VectorStore

logger = logging.getLogger(__name__)

class StructuredDataExtractorTool(Tool):
    """
    Инструмент для извлечения структурированных данных (полей) из текстовых документов.
    Использует TextExtractorTool для получения сырого текста и LLM для парсинга.
    Может работать с большими файлами, используя RAG-подход внутри себя.
    """
    def __init__(self, text_extractor_tool: Tool, llm_for_parsing: AbstractLLM, vector_store: Optional[VectorStore], predefined_field_sets: Dict[str, str]):
        super().__init__(
            name="structured_data_extractor",
            description=(
                "Извлекает структурированные данные (например, ФИО, адреса, номера) из содержимого файла "
                "по заданному названию набора полей (например, 'набор 1'). "
                "Принимает 'file_path' (строка) - путь к файлу, и 'field_set_name' (строка) - "
                "название предопределенного набора полей для извлечения. "
                "Возвращает извлеченные данные в формате 'Название поля: Значение'. "
                "Если данные для поля отсутствуют, оно не выводится. "
                "Если файл большой, его содержимое будет автоматически проиндексировано для поиска перед извлечением."
            )
        )
        self.text_extractor_tool = text_extractor_tool
        self.llm_for_parsing = llm_for_parsing
        self.vector_store = vector_store
        self.predefined_field_sets = predefined_field_sets # Словарь с наборами полей

    # Вспомогательная функция для чанкинга текста (копируем из agent.py)
    def _chunk_text(self, text: str, max_chunk_size: int = 1500, overlap_size: int = 200) -> List[str]:
        if not text:
            return []
        chunks = []
        text_length = len(text)
        current_pos = 0
        while current_pos < text_length:
            end_pos = min(current_pos + max_chunk_size, text_length)
            chunk = text[current_pos:end_pos]
            chunks.append(chunk)
            current_pos += (max_chunk_size - overlap_size)
            if current_pos >= text_length:
                break 
        return chunks

    async def execute(self, file_path: str, field_set_name: str, language: Optional[str] = None) -> str:
        if not os.path.exists(file_path):
            return f"Ошибка: Файл не найден по пути '{file_path}'."
        if not os.path.isfile(file_path):
            return f"Ошибка: Указанный путь '{file_path}' не является файлом."

        # Разрешаем имя набора полей в актуальный список полей
        fields_prompt = self.predefined_field_sets.get(field_set_name)
        if not fields_prompt:
            return f"Ошибка: Набор полей с названием '{field_set_name}' не найден. Доступные наборы: {', '.join(self.predefined_field_sets.keys())}."

        logger.info(f"StructuredDataExtractor: Извлечение текста из '{file_path}' для анализа по набору '{field_set_name}'.")
        extracted_text = await self.text_extractor_tool.execute(file_path=file_path, language=language)

        if "Ошибка:" in extracted_text or "Извлечение текста не дало результатов" in extracted_text:
            return f"Ошибка при извлечении текста из файла: {extracted_text}"

        logger.info(f"StructuredDataExtractor: Текст извлечен. Размер: {len(extracted_text)} символов. Приступаю к извлечению полей.")

        # Определяем промпт для LLM для извлечения данных
        extraction_instruction_prompt = (
            f"Используя следующий текст, извлеки информацию по указанным полям. "
            f"Выводи только те поля, для которых ты смог найти данные. "
            f"Сформируй вывод в виде списка 'Название поля: Значение'. "
            f"Если данные для поля отсутствуют, не выводи это поле. "
            f"Список полей для извлечения:\n{fields_prompt}\n\n"
            f"Текст для анализа:\n"
        )
        
        # Порог для прямого анализа LLM (в символах, можно настроить)
        LLM_DIRECT_PARSE_THRESHOLD = 5000 # ~1200 токенов для gpt-4o-mini
        
        final_extraction_content_for_llm = extracted_text

        if len(extracted_text) > LLM_DIRECT_PARSE_THRESHOLD:
            # Если текст большой, индексируем его и делаем RAG-запрос
            if not self.vector_store:
                return "Ошибка: Для анализа больших файлов необходимо инициализировать векторное хранилище в StructuredDataExtractorTool."
            
            logger.info(f"StructuredDataExtractor: Текст большой ({len(extracted_text)} символов). Разделяю на чанки и индексирую в векторном хранилище для RAG-извлечения.")
            chunks = self._chunk_text(extracted_text)
            
            # Добавляем метаданные, чтобы можно было идентифицировать чанки позже, если потребуется.
            # Для надежности, можно добавить уникальный ID файла.
            file_unique_id = f"file_{os.path.basename(file_path).replace('.', '_')}_{os.urandom(4).hex()}"
            metadatas = [
                {"source_file": file_path, "chunk_index": i, "extraction_id": file_unique_id} 
                for i in range(len(chunks))
            ]
            await self.vector_store.add_documents(chunks, metadatas=metadatas)
            logger.info(f"StructuredDataExtractor: Файл '{file_path}' проиндексирован с ID '{file_unique_id}'.")

            # Теперь выполняем RAG-поиск по проиндексированным чанкам
            # Запрос для RAG должен включать запрос на поля
            rag_query = f"Извлеки следующие поля: {fields_prompt} из информации о файле: {os.path.basename(file_path)}"
            retrieved_chunks = await self.vector_store.similarity_search(rag_query, k=5) # Извлекаем 5 самых релевантных чанков
            
            if not retrieved_chunks:
                logger.warning(f"StructuredDataExtractor: RAG поиск не дал результатов для файла '{file_path}' по запросу '{field_set_name}'.")
                return f"Не удалось найти релевантную информацию в проиндексированном файле '{os.path.basename(file_path)}' для извлечения полей по набору '{field_set_name}'."
            
            # Формируем контент для LLM из извлеченных чанков
            final_extraction_content_for_llm = "\n---\n".join(retrieved_chunks)
            logger.info(f"StructuredDataExtractor: Извлечено {len(retrieved_chunks)} релевантных чанков для LLM-анализа.")

        # Вызываем LLM для извлечения данных из (возможно, укороченного) текста
        try:
            # Системный промпт для LLM, чтобы она строго следовала формату
            llm_system_prompt = (
                "Ты высокоточный инструмент для извлечения информации из текста. "
                "Твоя задача - найти и вывести только запрашиваемые поля и их значения из предоставленного текста, "
                "строго следуя формату 'Название поля: Значение'. "
                "Не добавляй никаких других пояснений, предисловий или заключений. "
                "Если данных для поля нет, не выводи это поле. "
                "Если текста недостаточно, укажи это."
            )

            response_from_llm = await self.llm_for_parsing.generate_response(
                prompt=extraction_instruction_prompt + final_extraction_content_for_llm,
                system_prompt=llm_system_prompt
            )
            
            if isinstance(response_from_llm, str):
                # Форматируем ответ для удобного копирования
                formatted_output = "Извлеченные данные:\n" + response_from_llm
                return formatted_output
            else: 
                return f"Ошибка: LLM попыталась вызвать инструмент вместо извлечения данных. Ответ LLM: {json.dumps(response_from_llm, ensure_ascii=False)}"

        except Exception as e:
            logger.error(f"StructuredDataExtractor: Ошибка при извлечении данных LLM: {e}")
            return f"Ошибка при извлечении структурированных данных с помощью LLM: {e}"