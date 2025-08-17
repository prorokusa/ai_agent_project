from typing import List, Dict, Optional, Any
from interfaces.vector_store import VectorStore
from interfaces.llm import AbstractLLM

class SimpleVectorStore(VectorStore):
    """
    Простая реализация векторного хранилища в оперативной памяти.
    В реальном приложении здесь будет интеграция с Pinecone, Chroma, FAISS и т.д.
    """
    def __init__(self, llm_for_embedding: AbstractLLM):
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._llm = llm_for_embedding # LLM для генерации эмбеддингов
        self._next_id = 0
        print("Инициализировано SimpleVectorStore.")

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        added_ids = []
        for i, doc in enumerate(documents):
            doc_id = f"doc_{self._next_id}"
            embedding = self._llm.get_embedding(doc)
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            self._storage[doc_id] = {"content": doc, "embedding": embedding, "metadata": metadata}
            added_ids.append(doc_id)
            self._next_id += 1
        print(f"Добавлено {len(documents)} документов в векторное хранилище.")
        return added_ids

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        if not self._storage:
            print("Векторное хранилище пусто.")
            return []

        query_embedding = self._llm.get_embedding(query)
        if not query_embedding:
            print("Не удалось получить эмбеддинг для запроса.")
            return []

        # Простая косинусная схожесть для примера
        # Для упрощения: чем ближе числа, тем "похожее".
        # Здесь просто демонстрация, что мы что-то сравниваем.
        similarities = []
        for doc_id, data in self._storage.items():
            doc_embedding = data["embedding"]
            if doc_embedding and len(query_embedding) == len(doc_embedding):
                # Расчет простой L1-нормы расстояния (сумма абсолютных разностей)
                # Меньше значение - выше схожесть (для этого примера)
                distance = sum(abs(q - d) for q, d in zip(query_embedding, doc_embedding))
                similarities.append((distance, data["content"]))
        
        # Сортируем по "схожести" (для этого примера - по наименьшему расстоянию)
        similarities.sort(key=lambda x: x[0])
        
        results = [content for score, content in similarities[:k]]
        print(f"Найдено {len(results)} похожих документов для запроса '{query}'.")
        return results

    def clear(self):
        self._storage = {}
        self._next_id = 0
        print("Векторное хранилище очищено.")