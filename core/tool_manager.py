from typing import List, Dict, Any, Optional
from interfaces.tool import Tool

class ToolManager:
    """
    Управляет доступными инструментами для агента.
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """Регистрирует инструмент."""
        if tool.name in self._tools:
            print(f"Предупреждение: Инструмент с именем '{tool.name}' уже зарегистрирован и будет перезаписан.")
        self._tools[tool.name] = tool
        print(f"Инструмент '{tool.name}' зарегистрирован.")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Возвращает инструмент по имени."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """Возвращает список всех зарегистрированных инструментов с их описаниями."""
        return [{"name": tool.name, "description": tool.description} for tool in self._tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Выполняет инструмент по имени.
        Args:
            tool_name (str): Имя инструмента для выполнения.
            **kwargs: Аргументы для инструмента.
        Returns:
            Any: Результат выполнения инструмента.
        Raises:
            ValueError: Если инструмент не найден.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Инструмент '{tool_name}' не найден.")
        print(f"Выполнение инструмента '{tool_name}' с аргументами: {kwargs}")
        return tool.execute(**kwargs)