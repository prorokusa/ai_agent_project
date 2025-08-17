from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

class AbstractLLM(ABC):
    """
    Абстрактный базовый класс для подключения различных Больших Языковых Моделей (LLM).
    Пользователь будет реализовывать этот класс для каждой LLM.
    """
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        """
        Генерирует ответ от LLM на основе заданного промпта, системного промпта и истории.
        Args:
            prompt (str): Входной промпт для LLM (текущее сообщение пользователя).
            system_prompt (Optional[str]): Системный промпт для настройки поведения LLM.
            history (Optional[List[Dict[str, str]]]): Динамическая история чата (user/assistant/tool messages).
                                                       Формат: [{"role": "user", "content": "bla"}, {"role": "assistant", "content": "bla"}]
            **kwargs: Дополнительные аргументы, специфичные для LLM (например, temperature, max_tokens).
        Returns:
            str: Сгенерированный LLM ответ.
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Генерирует векторное представление (embedding) для заданного текста.
        Args:
            text (str): Текст для эмбеддинга.
        Returns:
            List[float]: Векторное представление текста.
        """
        pass