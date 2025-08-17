from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class Memory(ABC):
    """
    Абстрактный базовый класс для управления памятью (историей чата).
    Предназначен для хранения *динамической* истории (пользователь, ассистент, инструмент).
    Системный промпт управляется отдельно агентом.
    """
    @abstractmethod
    def add_message(self, role: str, content: str):
        """
        Добавляет сообщение в историю памяти.
        Args:
            role (str): Роль отправителя сообщения (например, "user", "assistant", "tool").
            content (str): Содержание сообщения.
        """
        pass

    @abstractmethod
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Извлекает историю чата.
        Args:
            limit (Optional[int]): Максимальное количество последних сообщений для извлечения.
        Returns:
            List[Dict[str, str]]: Список сообщений в формате [{"role": "...", "content": "..."}].
        """
        pass

    @abstractmethod
    def clear(self):
        """Очищает всю историю памяти."""
        pass