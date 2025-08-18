import os
import json
import asyncio
import logging # Добавим импорт для логирования
from dotenv import load_dotenv

# Импортируем все необходимые классы
from implementations.llms.simple_inference_llm import SimpleInferenceLLM
from implementations.llms.openai_llm import OpenAI_LLM
from implementations.llms.openrouter_llm import OpenRouter_LLM

from implementations.tools.calculator_tool import CalculatorTool
from implementations.tools.web_search_tool import GoogleCSESearchTool

from implementations.memory.chat_history_memory import ChatHistoryMemory

from implementations.vector_stores.chromadb_store import ChromaDBStore
from implementations.vector_stores.supabase_store import SupabaseVectorStore

from core.agent import AIAgent, AgentContext
from core.agent import AgentProcessor

# ИМПОРТИРУЕМ ВАШ НОВЫЙ МОДУЛЬ ДЛЯ ТРАНСКРИБАЦИИ
from utils.audio_transcriber import AudioTranscriber 

# Настройка базового логирования для всего приложения
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Пример пользовательского плагина (middleware) ---
async def my_custom_pre_llm_processor(agent: AIAgent, context: AgentContext):
    print(f"\n[PLUGIN: pre_llm] Выполняется перед запросом LLM...")
    import datetime
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    context.current_prompt_for_llm = f"Сегодня {current_date}. " + context.current_prompt_for_llm
    context.metadata["current_date"] = current_date
    print(f"[PLUGIN: pre_llm] Добавлена дата в промпт: {context.current_prompt_for_llm[:50]}...")

async def my_custom_final_response_processor(agent: AIAgent, context: AgentContext):
    print(f"\n[PLUGIN: final_response] Получен окончательный ответ агента.")
    if context.processed_successfully:
        print(f"[PLUGIN: final_response] Ответ успешный. Сообщение: {context.final_response[:50]}...")
    else:
        print(f"[PLUGIN: final_response] Ответ содержит ошибку. Ошибка: {context.error_message}")
    print(f"[PLUGIN: final_response] Метаданные контекста: {context.metadata}")


async def main():
    load_dotenv() # Загружаем переменные окружения из .env

    # --- Инициализация AudioTranscriber ---
    # Инициализируем транскрибатор один раз при запуске main().
    # Это может выбросить ValueError, если API-ключ не найден.
    audio_transcriber = None
    try:
        audio_transcriber = AudioTranscriber()
        logger.info("AudioTranscriber успешно инициализирован.")
    except ValueError as e:
        logger.error(f"Ошибка при инициализации AudioTranscriber: {e}")
        logger.error("Функции транскрибации аудио будут недоступны.")
    except Exception as e:
        logger.error(f"Неизвестная ошибка при инициализации AudioTranscriber: {e}")
        logger.error("Функции транскрибации аудио будут недоступны.")


    # my_llm = OpenRouter_LLM(model_name="qwen/qwen3-coder:free")
    my_llm = OpenAI_LLM(model_name="gpt-3.5-turbo")

    my_memory = ChatHistoryMemory()
    
    # my_vector_store = ChromaDBStore(llm_for_embedding=my_llm)
    # my_vector_store = ChromaDBStore(llm_for_embedding=my_llm, persist_directory="./chroma_data")
    my_vector_store = SupabaseVectorStore(llm_for_embedding=my_llm)

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
        "Ты используешь модель OpenRouter для генерации ответов."
        "После выполнения инструмента, **всегда** включай его результат (если он является ответом на вопрос пользователя) в свой финальный ответ. Формулируй ответ пользователю ясно, кратко и дружелюбно, прямо отвечая на его исходный вопрос, используя полученные данные."
    )

    agent = AIAgent(llm=my_llm, memory=my_memory, vector_store=my_vector_store, system_prompt=system_prompt)

    # agent.register_processor('pre_llm', my_custom_pre_llm_processor)
    # agent.register_processor('final_response', my_custom_final_response_processor)

    agent.register_tool(CalculatorTool())
    agent.register_tool(GoogleCSESearchTool())

    await my_vector_store.clear()
    
    await agent.vector_store.add_documents([
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
    print("Чтобы отправить текстовый файл, введите 'file:<путь_к_файлу.txt>' (например, 'file:temp_doc.txt').")
    print("Чтобы отправить аудиофайл, введите 'audio:<путь_к_файлу.mp3>' (например, 'audio:voice_memo.mp3').")


    while True:
        user_input = input("\nВы: ")
        if user_input.lower() == 'exit':
            print("Завершение диалога.")
            break
        
        if user_input.lower().startswith("file:"):
            file_path = user_input[len("file:"):].strip()
            if os.path.exists(file_path):
                response = await agent.process_message(file_input=file_path)
                print(f"Агент (обработка текстового файла): {response}")
            else:
                print(f"Ошибка: Файл не найден по пути '{file_path}'.")
        elif user_input.lower().startswith("audio:"): # НОВАЯ ЛОГИКА ДЛЯ АУДИО
            audio_path = user_input[len("audio:"):].strip()
            if audio_transcriber: # Проверяем, что транскрибатор инициализирован
                if os.path.exists(audio_path):
                    print(f"Начинаю транскрибацию аудиофайла: {audio_path}...")
                    transcribed_text = audio_transcriber.transcribe_audio(audio_path)
                    if transcribed_text:
                        print(f"Аудио транскрибировано. Передаю текст агенту: {transcribed_text[:100]}...")
                        response = await agent.process_message(text_input=transcribed_text)
                        print(f"Агент (из аудио): {response}")
                    else:
                        print("Не удалось транскрибировать аудио.")
                else:
                    print(f"Ошибка: Аудиофайл не найден по пути '{audio_path}'.")
            else:
                print("Модуль транскрибации аудио не инициализирован. Невозможно обработать аудио.")
        else:
            response = await agent.process_message(text_input=user_input)
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

if __name__ == "__main__":
    asyncio.run(main())