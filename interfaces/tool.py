from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):
    """
    Абстрактный базовый класс для инструментов, которыми может управлять агент.
    """
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Выполняет логику инструмента.
        Args:
            **kwargs: Аргументы, необходимые для выполнения инструмента.
        Returns:
            Any: Результат выполнения инструмента.
        """
        pass