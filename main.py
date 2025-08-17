import os
import json 
from dotenv import load_dotenv

# Импортируем все необходимые классы
from implementations.llms.simple_inference_llm import SimpleInferenceLLM
from implementations.llms.openai_llm import OpenAI_LLM 

from implementations.tools.calculator_tool import CalculatorTool
from implementations.tools.web_search_tool import GoogleCSESearchTool 

from implementations.memory.chat_history_memory import ChatHistoryMemory

from implementations.vector_stores.chromadb_store import ChromaDBStore 
from implementations.vector_stores.supabase_store import SupabaseVectorStore # <--- НОВЫЙ ИМПОРТ

from core.agent import AIAgent

if __name__ == "__main__":
    load_dotenv() 

    # 1. Инициализация компонентов
    my_llm = OpenAI_LLM(model_name="gpt-3.5-turbo") 
    # my_llm = SimpleInferenceLLM(model_name="GPT-Dummy-3.5") 

    my_memory = ChatHistoryMemory()
    
    # --- Выбор векторного хранилища ---
    # Раскомментируйте один из следующих вариантов:

    # Вариант 1: Использовать ChromaDB (in-memory или persistent)
    # my_vector_store = ChromaDBStore(llm_for_embedding=my_llm) 
    # my_vector_store = ChromaDBStore(llm_for_embedding=my_llm, persist_directory="./chroma_data")

    # Вариант 2: Использовать Supabase (требует настройки .env и создания таблицы в Supabase)
    my_vector_store = SupabaseVectorStore(llm_for_embedding=my_llm) 
    # -----------------------------------

    # 2. Определение системного промпта
    system_prompt = (
        "Ты очень полезный, дружелюбный и компетентный AI ассистент. "
        "Твоя задача - точно отвечать на вопросы, использовать предоставленные инструменты, когда это уместно, "
        "и извлекать информацию из векторного хранилища для ответов на вопросы о личных данных или контексте. "
        "Всегда старайся быть вежливым и кратким, если это возможно. "
        "Если пользователь задает вопрос, на который может ответить инструмент, предложи его использовать или используй его напрямую, если это очевидно. "
        "Если вопрос касается твоей собственной информации или контекста, используй поиск по векторному хранилищу. "
        "После выполнения инструмента или поиска в векторном хранилище, обобщи результаты и предоставь связный ответ."
        "Будь готов использовать функцию `google_cse_search` для поиска актуальной информации в интернете, когда это необходимо." 
        "Особенно используй `google_cse_search`, если вопрос касается текущих событий, статистических данных или общих фактов, отсутствующих в твоей памяти."
        "Если пользователь просит проанализировать файл, ожидай, что содержание файла будет передано тебе как часть user-сообщения, и ты должен его проанализировать."
    )

    # 3. Создание агента с системным промптом
    agent = AIAgent(llm=my_llm, memory=my_memory, vector_store=my_vector_store, system_prompt=system_prompt)

    # 4. Регистрация инструментов
    agent.register_tool(CalculatorTool())
    agent.register_tool(GoogleCSESearchTool()) 

    # 5. Добавление документов в векторное хранилище для демонстрации RAG 
    # Внимание: если вы переключаетесь между ChromaDB и Supabase,
    # и у вас persistent ChromaDB, то данные будут сохраняться.
    # Для Supabase, данные будут сохраняться в вашей БД, пока вы их не очистите вручную
    # или через my_vector_store.clear().
    
    # Рекомендуется очищать хранилище при запуске для тестов
    my_vector_store.clear() # Очистка данных перед добавлением для чистоты эксперимента
    
    agent.vector_store.add_documents([
        "Мое любимое хобби - это чтение книг по истории и философия.",
        "Я живу в городе Москва, работаю инженером в IT-компании, мой адрес: ул. Пушкина, 10.",
        "Последний раз мы обсуждали планы на отпуск в горах Кавказа в июле 2024 года.",
        "Имя моего питомца - Барсик, он рыжий кот и очень любит играть с лазерной указкой.",
        "Мои предпочтения в еде включают итальянскую кухню, особенно пиццу и пасту."
    ], metadatas=[
        {"type": "hobby"},
        {"type": "personal_info", "city": "Москва"},
        {"type": "past_discussion", "year": 2024},
        {"type": "pet_info"},
        {"type": "food_prefs"}
    ])

    print("\n--- Начало интерактивного диалога с Агентом ---")
    print("Введите 'exit' для завершения.")
    print("Чтобы отправить файл, введите 'file:<путь_к_файлу.txt>' (например, 'file:temp_doc.txt').")

    while True:
        user_input = input("\nВы: ")
        if user_input.lower() == 'exit':
            print("Завершение диалога.")
            break
        
        if user_input.lower().startswith("file:"):
            file_path = user_input[len("file:"):].strip()
            if os.path.exists(file_path):
                response = agent.process_message(file_input=file_path)
                print(f"Агент (обработка файла): {response}")
            else:
                print(f"Ошибка: Файл не найден по пути '{file_path}'.")
        else:
            response = agent.process_message(text_input=user_input)
            print(f"Агент: {response}")

    print("\n--- Полная история чата ---")
    for msg in agent.memory.get_history():
        if msg["role"] == "assistant" and isinstance(msg["content"], str) and "tool_calls" in msg["content"]:
            try:
                content_obj = json.loads(msg["content"])
                if "tool_calls" in content_obj:
                    print(f"assistant (вызов инструмента): {json.dumps(content_obj['tool_calls'], indent=2, ensure_ascii=False)}")
                else: 
                    print(f"{msg['role']}: {msg['content']}")
            except json.JSONDecodeError:
                print(f"{msg['role']}: {msg['content']}")
        elif msg["role"] == "tool" and isinstance(msg["content"], str):
             try:
                content_obj = json.loads(msg["content"])
                print(f"tool (результат): Tool ID: {content_obj.get('tool_call_id', 'N/A')}, Output: {json.dumps(content_obj.get('output', ''), indent=2, ensure_ascii=False)}")
             except json.JSONDecodeError:
                print(f"{msg['role']}: {msg['content']}")
        else:
            print(f"{msg['role']}: {msg['content']}")

    # Очистка ChromaDB / Supabase данных при завершении (раскомментируйте, если хотите очищать данные при каждом выходе)
    my_vector_store.clear()