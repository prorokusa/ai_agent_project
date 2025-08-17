from typing import List, Dict, Optional
from interfaces.memory import Memory

class ChatHistoryMemory(Memory):
    """
    Простая реализация памяти в оперативной памяти для динамической истории чата.
    """
    def __init__(self):
        self._messages: List[Dict[str, str]] = []
        print("Инициализирована ChatHistoryMemory.")

    def add_message(self, role: str, content: str):
        # Важно: системный промпт не добавляется сюда. Только динамические сообщения.
        if role not in ["user", "assistant", "tool", "file_content", "system_error"]: # Добавьте сюда другие роли, которые вы хотите хранить
            print(f"Предупреждение: Роль '{role}' не является стандартной для хранения в динамической памяти.")
        self._messages.append({"role": role, "content": content})
        print(f"Добавлено сообщение в память: {role}: {content[:50]}...")

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        if limit is None:
            return self._messages[:]
        return self._messages[-limit:] # Возвращаем последние 'limit' сообщений

    def clear(self):
        self._messages = []
        print("История памяти очищена.")