import os
import shutil
from typing import List, Dict, Optional, Any, Callable

from interfaces.vector_store import VectorStore
from interfaces.llm import AbstractLLM

import chromadb
# Импортируем типы для EmbeddingFunction из ChromaDB
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings # Добавлены Documents, Embeddings

# Определим размерность эмбеддинга для OpenAI text-embedding-ada-002
# Если вы используете SimpleInferenceLLM, измените на 128.
OPENAI_EMBEDDING_DIM = 1536 

class CustomLLMEmbeddingFunction(EmbeddingFunction):
    """
    Класс-обертка для использования нашей AbstractLLM как функции эмбеддинга ChromaDB.
    """
    def __init__(self, llm: AbstractLLM, embedding_dimension: int = OPENAI_EMBEDDING_DIM):
        self._llm = llm
        self._embedding_dimension = embedding_dimension
        self._name = f"custom_llm_embedding_function_dim_{embedding_dimension}"

    # Обязательный метод для соответствия интерфейсу EmbeddingFunction
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            emb = self._llm.get_embedding(text)
            if emb and len(emb) == self._embedding_dimension:
                embeddings.append(emb)
            else:
                print(f"Предупреждение: Не удалось получить валидный эмбеддинг для текста: '{text[:50]}...'. Использование нулевого вектора размерностью {self._embedding_dimension}.")
                embeddings.append([0.0] * self._embedding_dimension)
        return embeddings

    # Метод 'name' требуется ChromaDB для валидации
    def name(self) -> str:
        return self._name

class ChromaDBStore(VectorStore):
    """
    Реализация векторного хранилища с использованием ChromaDB.
    Может работать как in-memory, так и с persistent хранилищем.
    """
    def __init__(self, llm_for_embedding: AbstractLLM, collection_name: str = "agent_documents", persist_directory: Optional[str] = None):
        self._llm = llm_for_embedding
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        if self.persist_directory:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            print(f"ChromaDB Persistent Client инициализирован в: {self.persist_directory}")
            os.makedirs(self.persist_directory, exist_ok=True)
        else:
            self.client = chromadb.Client() # In-memory client
            print("ChromaDB In-Memory Client инициализирован.")
        
        # Передаем экземпляр нашей новой обертки CustomLLMEmbeddingFunction
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=CustomLLMEmbeddingFunction(self._llm) 
        )
        print(f"Коллекция ChromaDB '{self.collection_name}' готова.")

    # Метод _get_embedding_function_for_chroma больше не нужен,
    # так как мы передаем экземпляр класса CustomLLMEmbeddingFunction напрямую.
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        ids = [f"doc_{self.collection.count() + i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{}] * len(documents)

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Добавлено {len(documents)} документов в ChromaDB.")
        return ids

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        if self.collection.count() == 0:
            print("ChromaDB коллекция пуста.")
            return []
            
        results = self.collection.query(
            query_texts=[query], 
            n_results=k
        )
        
        if results and results['documents'] and results['documents'][0]:
            found_docs = results['documents'][0]
            print(f"Найдено {len(found_docs)} похожих документов в ChromaDB.")
            return found_docs
        print("Не найдено похожих документов в ChromaDB.")
        return []

    def clear(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            if self.persist_directory and os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                print(f"ChromaDB Persistent Directory очищена: {self.persist_directory}")
            print(f"Коллекция ChromaDB '{self.collection_name}' очищена.")
        except Exception as e:
            print(f"Ошибка при очистке ChromaDB коллекции: {e}")