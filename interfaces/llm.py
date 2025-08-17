from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union, Awaitable

class AbstractLLM(ABC):
    """
    Абстрактный базовый класс для подключения различных Больших Языковых Моделей (LLM).
    """
    @abstractmethod
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Генерирует ответ от LLM.
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]: # <--- ИЗМЕНЕНИЕ: Метод get_embedding теперь синхронный
        """
        Генерирует векторное представление (embedding) для заданного текста.
        """
        pass