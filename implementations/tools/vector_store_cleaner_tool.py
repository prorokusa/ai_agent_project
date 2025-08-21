# implementations/tools/vector_store_cleaner_tool.py
import logging
from typing import Optional
from interfaces.tool import Tool
from interfaces.vector_store import VectorStore

logger = logging.getLogger(__name__)

class VectorStoreCleanerTool(Tool):
    """
    Инструмент для очистки (удаления всех данных) из векторного хранилища.
    Используйте с осторожностью!
    """
    def __init__(self, vector_store: VectorStore):
        super().__init__(
            name="vector_store_cleaner",
            description="Очищает (удаляет все данные) из векторного хранилища агента. Используйте с осторожностью! Принимает опциональный аргумент 'confirm' (boolean) - если True, выполнит очистку. Использование: 'очисти временный раг подтвердив: true'."
        )
        self.vector_store = vector_store

    async def execute(self, confirm: bool = False) -> str:
        if not self.vector_store:
            return "Ошибка: Векторное хранилище не инициализировано, невозможно выполнить очистку."
        
        if not confirm:
            return "Пожалуйста, подтвердите очистку векторного хранилища, добавив 'confirm: true' в запрос к инструменту. Например: 'очисти временный раг подтвердив: true'."
        
        try:
            logger.info("Начинаю очистку векторного хранилища...")
            await self.vector_store.clear()
            logger.info("Векторное хранилище успешно очищено.")
            return "Векторное хранилище успешно очищено от всех данных."
        except Exception as e:
            logger.error(f"Ошибка при очистке векторного хранилища: {e}")
            return f"Произошла ошибка при очистке векторного хранилища: {e}"