# utils/audio_transcriber.py
import os
import logging
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIStatusError
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError # ИСПРАВЛЕНО: 'CouldNotDecodeError' на 'CouldntDecodeError'

# Настройка логирования для модуля
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Загружаем переменные окружения один раз при загрузке модуля
load_dotenv()

class AudioTranscriber:
    """
    Модуль для транскрибации аудиофайлов в текст с использованием OpenAI Whisper API.
    Поддерживает автоматическую конвертацию в поддерживаемые форматы при необходимости.
    """
    # Поддерживаемые форматы OpenAI Whisper API
    SUPPORTED_FORMATS = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
    DEFAULT_CONVERT_FORMAT = "mp3" # Формат, в который будем конвертировать по умолчанию

    def __init__(self):
        """
        Инициализирует транскрибатор.
        Требует наличия переменной окружения OPENAI_API_KEY.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY не найдена в переменных окружения. Убедитесь, что она установлена и .env файл загружен.")
            raise ValueError("OPENAI_API_KEY не установлена. Транскрибация через API невозможна.")
        
        self.client = OpenAI(api_key=api_key)
        logger.info("AudioTranscriber (OpenAI API) успешно инициализирован.")

    def _convert_audio_to_supported_format(self, input_filepath: str, target_format: str = DEFAULT_CONVERT_FORMAT) -> str | None:
        """
        Конвертирует аудиофайл в поддерживаемый формат.

        :param input_filepath: Путь к исходному аудиофайлу.
        :param target_format: Целевой формат (например, "mp3", "wav").
        :return: Путь к конвертированному файлу или None, если конвертация не удалась.
        """
        try:
            # Создаем временный файл для конвертированного аудио
            temp_dir = tempfile.gettempdir()
            temp_filename = os.path.join(temp_dir, f"temp_converted_audio_{os.getpid()}_{os.urandom(8).hex()}.{target_format}")

            logger.info(f"Попытка конвертации '{input_filepath}' в '{target_format}'...")
            audio = AudioSegment.from_file(input_filepath)
            audio.export(temp_filename, format=target_format)
            logger.info(f"Файл успешно конвертирован во временный файл: {temp_filename}")
            return temp_filename
        except FileNotFoundError:
            logger.error("Ошибка: FFmpeg не найден. Пожалуйста, убедитесь, что FFmpeg установлен и доступен в PATH.")
            return None
        except CouldntDecodeError as e: # ИСПРАВЛЕНО ЗДЕСЬ
            logger.error(f"Ошибка конвертации: Не удалось декодировать аудиофайл '{input_filepath}'. "
                         f"Возможно, файл поврежден или FFmpeg не может его прочитать. Детали: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при конвертации '{input_filepath}': {e}")
            return None

    def transcribe_audio(self, audio_filepath: str) -> str:
        """
        Транскрибирует аудиофайл в текст с использованием OpenAI Whisper API.
        Автоматически конвертирует файл, если исходный формат не поддерживается.

        :param audio_filepath: Полный путь к аудиофайлу.
        :return: Транскрибированный текст. Возвращает пустую строку, если транскрибация не удалась.
        """
        if not os.path.exists(audio_filepath):
            logger.error(f"Ошибка: Файл не найден по пути '{audio_filepath}'")
            return ""
        
        if not os.path.isfile(audio_filepath):
            logger.error(f"Ошибка: Указанный путь '{audio_filepath}' не является файлом.")
            return ""

        # Проверка размера файла (API имеет ограничение в 25 МБ)
        if os.path.getsize(audio_filepath) > 25 * 1024 * 1024: # 25 MB
            logger.error(f"Файл '{audio_filepath}' слишком большой для OpenAI Whisper API (> 25 МБ).")
            return ""

        current_audio_file = audio_filepath
        temp_file_created = False

        try:
            logger.info(f"Начало транскрибации файла '{current_audio_file}' через OpenAI Whisper API...")
            with open(current_audio_file, "rb") as audio_file_obj:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file_obj,
                    response_format="text"
                )
            transcribed_text = transcript
            logger.info(f"Транскрибация завершена для {current_audio_file}.")
            return transcribed_text
        except APIStatusError as e:
            if e.status_code == 400 and "Invalid file format" in str(e):
                logger.warning(f"Ошибка формата файла при транскрибации '{audio_filepath}'. Попытка конвертации...")
                
                converted_filepath = self._convert_audio_to_supported_format(audio_filepath)
                
                if converted_filepath:
                    current_audio_file = converted_filepath
                    temp_file_created = True
                    
                    if os.path.getsize(current_audio_file) > 25 * 1024 * 1024:
                        logger.error(f"Конвертированный файл '{current_audio_file}' также слишком большой (> 25 МБ).")
                        return ""

                    try:
                        logger.info(f"Повторная попытка транскрибации конвертированного файла '{current_audio_file}'...")
                        with open(current_audio_file, "rb") as audio_file_obj:
                            transcript = self.client.audio.transcriptions.create(
                                model="whisper-1", 
                                file=audio_file_obj,
                                response_format="text"
                            )
                        transcribed_text = transcript
                        logger.info(f"Транскрибация завершена для конвертированного файла {current_audio_file}.")
                        return transcribed_text
                    except Exception as retry_e:
                        logger.error(f"Ошибка при повторной транскрибации конвертированного файла '{current_audio_file}' через OpenAI API: {retry_e}")
                        logger.error("Проверьте ваш API-ключ, подключение к интернету и формат/размер файла.")
                        return ""
                else:
                    logger.error("Конвертация файла не удалась, невозможно продолжить транскрибацию.")
                    return ""
            else:
                logger.error(f"Ошибка при транскрибации файла '{audio_filepath}' через OpenAI API: {e}")
                logger.error("Проверьте ваш API-ключ, подключение к интернету.")
                return ""
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при транскрибации файла '{audio_filepath}': {e}")
            logger.error("Проверьте доступность файла и его целостность.")
            return ""
        finally:
            if temp_file_created and os.path.exists(current_audio_file):
                try:
                    os.remove(current_audio_file)
                    logger.info(f"Временный файл '{current_audio_file}' удален.")
                except Exception as cleanup_e:
                    logger.warning(f"Не удалось удалить временный файл '{current_audio_file}': {cleanup_e}")


# Пример использования модуля для тестирования
if __name__ == "__main__":
    test_audio_path_awb = "audio_to_convert.awb"
    test_audio_path_mp3 = "audio_ready.mp3"

    if not os.path.exists(test_audio_path_awb) and not os.path.exists(test_audio_path_mp3):
        print(f"Ни '{test_audio_path_awb}', ни '{test_audio_path_mp3}' не найдены.")
        print("Создаю фиктивный MP3-файл для тестирования.")
        try:
            from pydub import AudioSegment
            empty_audio = AudioSegment.silent(duration=2000)
            empty_audio.export(test_audio_path_mp3, format="mp3")
            print(f"Фиктивный файл '{test_audio_path_mp3}' создан.")
            actual_test_path = test_audio_path_mp3
        except Exception as e:
            print(f"Не удалось создать фиктивный аудиофайл: {e}")
            print("Пожалуйста, убедитесь, что у вас есть реальный аудиофайл для тестирования.")
            exit()
    elif os.path.exists(test_audio_path_awb):
        actual_test_path = test_audio_path_awb
    else:
        actual_test_path = test_audio_path_mp3


    if not os.path.exists(actual_test_path):
        print(f"Файл для тестирования не найден по пути: {actual_test_path}")
        print("Пожалуйста, замените его на существующий аудиофайл для проверки.")
        print("И убедитесь, что у вас есть OPENAI_API_KEY в файле .env")
    else:
        print(f"Попытка транскрибации тестового файла: {actual_test_path} через OpenAI API (с автоконвертацией)")
        try:
            transcriber = AudioTranscriber()
            text = transcriber.transcribe_audio(actual_test_path)

            if text:
                print("\n--- Транскрибированный текст ---")
                print(text)
            else:
                print("Не удалось транскрибировать аудио.")
        except ValueError as e:
            print(f"Ошибка инициализации транскрибатора: {e}")
        except Exception as e:
            print(f"Непредвиденная ошибка при тестировании: {e}")
        finally:
            if 'actual_test_path' in locals() and actual_test_path == test_audio_path_mp3 and os.path.exists(test_audio_path_mp3):
                try:
                    os.remove(test_audio_path_mp3)
                    print(f"Фиктивный тестовый файл '{test_audio_path_mp3}' удален.")
                except Exception as cleanup_e:
                    print(f"Не удалось удалить фиктивный тестовый файл: {cleanup_e}")