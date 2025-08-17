from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

class VectorStore(ABC):
    """
    Абстрактный базовый класс для векторного хранилища.
    """
    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """
        Добавляет документы в векторное хранилище, генерируя их эмбеддинги.
        Args:
            documents (List[str]): Список текстовых документов.
            metadatas (Optional[List[Dict]]): Список метаданных, соответствующих документам.
        Returns:
            List[str]: Список ID добавленных документов.
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        """
        Выполняет поиск по схожести в векторном хранилище.
        Args:
            query (str): Запрос для поиска.
            k (int): Количество наиболее похожих документов для возврата.
        Returns:
            List[str]: Список наиболее похожих документов.
        """
        pass

    @abstractmethod
    def clear(self):
        """Очищает векторное хранилище."""
        pass