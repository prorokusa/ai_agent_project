import os
from typing import List, Dict, Optional, Any, Tuple, Callable 
from interfaces.vector_store import VectorStore
from interfaces.llm import AbstractLLM

from supabase.client import create_client, Client 

OPENAI_EMBEDDING_DIM = 1536 

class SupabaseVectorStore(VectorStore):
    """
    Реализация векторного хранилища с использованием Supabase (PostgreSQL + pgvector).
    """
    def __init__(self, llm_for_embedding: AbstractLLM, table_name: str = "documents"):
        self._llm = llm_for_embedding
        self.table_name = table_name

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL или SUPABASE_ANON_KEY не установлены в .env. Невозможно инициализировать SupabaseVectorStore.")
        
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            print(f"SupabaseVectorStore инициализирован для таблицы '{self.table_name}'.")
        except Exception as e:
            raise ConnectionError(f"Не удалось подключиться к Supabase: {e}")

    async def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]: 
        if not documents:
            return []

        data_to_insert = []

        for i, doc_content in enumerate(documents):
            embedding = self._llm.get_embedding(doc_content) 
            if not embedding or len(embedding) != OPENAI_EMBEDDING_DIM:
                print(f"Предупреждение: Не удалось получить валидный эмбеддинг для документа: '{doc_content[:50]}...'. Пропуск документа.")
                continue
            
            entry = {
                "content": doc_content,
                "embedding": embedding,
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else {} 
            }
            data_to_insert.append(entry)

        if not data_to_insert:
            return []

        try:
            # --- ИЗМЕНЕНИЕ: УДАЛЕН await ---
            response = self.client.from_(self.table_name).insert(data_to_insert).execute() 
            inserted_ids = [item['id'] for item in response.data] if response.data else [] 
            print(f"Добавлено {len(inserted_ids)} документов в Supabase таблицу '{self.table_name}'.")
            return inserted_ids
        except Exception as e:
            print(f"Ошибка при добавлении документов в Supabase: {e}")
            return []

    async def similarity_search(self, query: str, k: int = 4, filters: Optional[Dict] = None) -> List[str]: 
        query_embedding = self._llm.get_embedding(query) 
        if not query_embedding or len(query_embedding) != OPENAI_EMBEDDING_DIM:
            print("Не удалось получить валидный эмбеддинг для запроса. Поиск невозможен.")
            return []

        try:
            # --- ИЗМЕНЕНИЕ: УДАЛЕН await ---
            response = self.client.rpc( 
                'match_documents', 
                {
                    'query_embedding': query_embedding,
                    'match_count': k,
                    'filter': filters if filters else {} 
                }
            ).execute()
            
            if response.data:
                found_docs = [item['content'] for item in response.data]
                print(f"Найдено {len(found_docs)} похожих документов в Supabase.")
                return found_docs
            print("Не найдено похожих документов в Supabase.")
            return []
        except Exception as e:
            print(f"Ошибка при поиске по схожести в Supabase: {e}")
            return []

    async def clear(self): # <--- ИЗМЕНЕНИЕ: Метод clear() остается асинхронным, но await внутри удален
        try:
            # --- ИЗМЕНЕНИЕ: УДАЛЕН await ---
            response = self.client.from_(self.table_name).delete().gt("id", 0).execute() 
            
            print(f"Очищена таблица Supabase '{self.table_name}'.")
        except Exception as e:
            print(f"Ошибка при очистке таблицы Supabase '{self.table_name}': {e}")