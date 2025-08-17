from typing import Union, Any
from interfaces.tool import Tool

class CalculatorTool(Tool):
    """
    Пример инструмента: простой калькулятор.
    """
    def __init__(self):
        super().__init__(name="calculator", description="Выполняет базовые математические операции: сложение, вычитание, умножение, деление. Принимает 'expression' (str).")

    def execute(self, expression: str) -> Union[int, float, str]:
        try:
            # ОСТОРОЖНО: eval() может быть небезопасным при использовании с недоверенным вводом!
            # В продакшене используйте более безопасные парсеры или библиотеки (например, asteval, numexpr).
            result = eval(expression) 
            return result
        except Exception as e:
            return f"Ошибка при вычислении: {e}"