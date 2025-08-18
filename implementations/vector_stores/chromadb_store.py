# implementations/vector_stores/chromadb_store.py
import os
import shutil
# import time # Удаляем, так как это не помогает решить проблему блокировки
from typing import List, Dict, Optional, Any, Callable

from interfaces.vector_store import VectorStore
from interfaces.llm import AbstractLLM

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings 
from chromadb.errors import NotFoundError 

OPENAI_EMBEDDING_DIM = 1536 

class CustomLLMEmbeddingFunction(EmbeddingFunction):
    # ... (остается без изменений) ...
    def __init__(self, llm: AbstractLLM, embedding_dimension: int = OPENAI_EMBEDDING_DIM):
        self._llm = llm
        self._embedding_dimension = embedding_dimension
        self._name = f"custom_llm_embedding_function_dim_{embedding_dimension}"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                emb = self._llm.get_embedding(text) 
                if emb and len(emb) == self._embedding_dimension:
                    embeddings.append(emb)
                else:
                    print(f"Предупреждение: Не удалось получить валидный эмбеддинг для текста: '{text[:50]}...'. Использование нулевого вектора размерностью {self._embedding_dimension}.")
                    embeddings.append([0.0] * self._embedding_dimension)
            except Exception as e:
                print(f"Ошибка при получении эмбеддинга в CustomLLMEmbeddingFunction: {e}. Использование нулевого вектора.")
                embeddings.append([0.0] * self._embedding_dimension)
        return embeddings

    def name(self) -> str:
        return self._name

class ChromaDBStore(VectorStore):
    def __init__(self, llm_for_embedding: AbstractLLM, collection_name: str = "agent_documents", persist_directory: Optional[str] = None):
        self._llm = llm_for_embedding
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            print(f"ChromaDB Persistent Client инициализирован в: {self.persist_directory}")
        else:
            self.client = chromadb.Client() 
            print("ChromaDB In-Memory Client инициализирован.")
        
        self._create_or_get_collection()

        print(f"Коллекция ChromaDB '{self.collection_name}' готова.")

    def _create_or_get_collection(self):
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=CustomLLMEmbeddingFunction(self._llm)
            )
            print(f"Коллекция ChromaDB '{self.collection_name}' получена.")
        except NotFoundError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=CustomLLMEmbeddingFunction(self._llm)
            )
            print(f"Коллекция ChromaDB '{self.collection_name}' создана.")
        except Exception as e:
            print(f"Ошибка при инициализации/получении коллекции ChromaDB: {e}")
            raise 

    async def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]: 
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

    async def similarity_search(self, query: str, k: int = 4) -> List[str]: 
        if self.collection.count() == 0:
            print(f"ChromaDB коллекция '{self.collection_name}' пуста.")
            return []
            
        results = self.collection.query(
            query_texts=[query], 
            n_results=k
        )
        
        if results and results['documents'] and results['documents'][0]:
            found_docs = results['documents'][0]
            print(f"Найдено {len(found_docs)} похожих документов в ChromaDB.")
            return found_docs
        print(f"Не найдено похожих документов в ChromaDB коллекции '{self.collection_name}'.")
        return []

    async def clear(self): 
        try:
            # Сначала удаляем коллекцию
            self.client.delete_collection(name=self.collection_name)
            print(f"Коллекция ChromaDB '{self.collection_name}' удалена.")
            
            # Затем пытаемся удалить директорию, если она персистентная
            if self.persist_directory and os.path.exists(self.persist_directory):
                print(f"Попытка удалить директорию: {self.persist_directory}")
                try:
                    # ПОЛНОСТЬЮ УБИРАЕМ time.sleep(), т.к. это не решает проблему блокировки
                    shutil.rmtree(self.persist_directory) 
                    print(f"ChromaDB Persistent Directory очищена: {self.persist_directory}")
                except OSError as e:
                    # --- ИЗМЕНЕНИЕ ЗДЕСЬ: ЛОВИМ OSError И ПРОСТО ПРЕДУПРЕЖДАЕМ ---
                    print(f"Предупреждение: Не удалось полностью очистить директорию '{self.persist_directory}': {e}. Файлы могут быть заблокированы. Продолжаем работу, но старые файлы могут остаться.")
            
            # После удаления, мы должны ПЕРЕИНИЦИАЛИЗИРОВАТЬ self.collection
            # чтобы он ссылался на НОВУЮ, ПУСТУЮ коллекцию.
            self._create_or_get_collection() 
            print(f"Коллекция ChromaDB '{self.collection_name}' пересоздана и готова к использованию.")

        except NotFoundError:
            print(f"Предупреждение: Коллекция ChromaDB '{self.collection_name}' не существовала при попытке очистки.")
            self._create_or_get_collection() 
            print(f"Коллекция ChromaDB '{self.collection_name}' создана (после попытки очистки несуществующей).")
        except Exception as e:
            # Если это не NotFoundError и не OSError при rmtree, то это действительно непредвиденная ошибка
            print(f"Непредвиденная ошибка при очистке ChromaDB коллекции '{self.collection_name}': {e}")