import os
import asyncio
import logging
import tempfile
import aioftp
import time
from typing import Any, List, Dict, Optional # <-- ДОБАВЛЕНО: Dict

from interfaces.tool import Tool
from utils.audio_transcriber import AudioTranscriber
import config # Импорт настроек из config.py

logger = logging.getLogger(__name__)

class FtpAudioProcessorTool(Tool):
    """
    Инструмент для отслеживания новых аудиофайлов на FTP-сервере,
    их загрузки, транскрибации и возврата текста.
    """
    def __init__(self):
        super().__init__(
            name="ftp_audio_processor",
            description="Сканирует указанную папку на FTP-сервере на наличие новых аудиофайлов, скачивает их, транскрибирует с помощью инструмента транскрибации аудио и возвращает транскрибированный текст. Может опционально очищать удаленные файлы после обработки. Требует наличия локальной директории для загрузки."
        )
        self.transcriber = AudioTranscriber()
        self.processed_files_history = {} 
        self.default_audio_extensions = [f".{ext}" for ext in AudioTranscriber.SUPPORTED_FORMATS + ['awb', 'amr']] 
        logger.info("FtpAudioProcessorTool инициализирован.")

    async def _list_ftp_files(self, client: aioftp.Client, remote_path: str) -> List[Dict[str, Any]]:
        """
        Вспомогательная функция для получения списка файлов на FTP с их размерами.
        Возвращает список словарей: [{"name": "file.mp3", "size": 12345}]
        """
        files_info = []
        try:
            async for path, info in client.list(remote_path):
                file_name = path.name
                if file_name and file_name != "." and file_name != "..":
                    if info.is_file:
                        files_info.append({"name": file_name, "size": info.size})
        except aioftp.StatusCodeError as e:
            logger.error(f"Ошибка FTP при получении списка файлов '{remote_path}': {e}")
            raise
        except Exception as e:
            logger.error(f"Неизвестная ошибка при получении списка файлов '{remote_path}': {e}")
            raise
        return files_info

    async def execute(self, ftp_host: str, ftp_user: str, ftp_password: str,
                      remote_path: str, local_download_dir: str,
                      allowed_audio_extensions: Optional[List[str]] = None,
                      clear_remote_after_processing: bool = False) -> str:
        
        os.makedirs(local_download_dir, exist_ok=True)

        if allowed_audio_extensions is None:
            final_allowed_extensions = self.default_audio_extensions
        else:
            final_allowed_extensions = [f".{ext.lstrip('.')}" for ext in allowed_audio_extensions] 

        ftp_folder_key = f"{ftp_host}:{remote_path}"
        if ftp_folder_key not in self.processed_files_history:
            self.processed_files_history[ftp_folder_key] = {} 

        transcriptions = []
        files_to_delete_from_ftp = []

        try:
            async with aioftp.Client(ftp_host, ftp_user, ftp_password) as client:
                logger.info(f"Подключен к FTP '{ftp_host}', директория: '{remote_path}'")

                current_remote_files_info = await self._list_ftp_files(client, remote_path)
                
                new_files_to_process = []
                for file_info in current_remote_files_info:
                    file_name = file_info["name"]
                    file_size = file_info["size"]
                    file_extension = os.path.splitext(file_name)[1].lower()

                    is_new = file_name not in self.processed_files_history[ftp_folder_key]
                    has_changed = (not is_new and 
                                   (self.processed_files_history[ftp_folder_key][file_name]["size"] != file_size)) 
                                   # Здесь намеренно убрал time, так как aioftp.Client.list() не всегда возвращает надежное время модификации
                                   # Если нужна более надежная проверка, рассмотрите mtime в info.time.
                                   # Но для FTP это может быть не всегда консистентно.
                                   # Для текущего асинхронного FTP-клиента info.time может быть None или неточно.
                                   # Сравнение по размеру - более надежный первичный признак для большинства случаев.

                    if (is_new or has_changed) and file_extension in final_allowed_extensions:
                        new_files_to_process.append(file_name)
                        self.processed_files_history[ftp_folder_key][file_name] = {"size": file_size, "timestamp": time.time()}

                if not new_files_to_process:
                    return f"На FTP-сервере '{ftp_host}' в папке '{remote_path}' новые или измененные аудиофайлы не обнаружены."

                logger.info(f"Обнаружено {len(new_files_to_process)} новых/измененных аудиофайлов для обработки.")

                for file_name in new_files_to_process:
                    remote_filepath = os.path.join(remote_path, file_name)
                    local_filepath = os.path.join(local_download_dir, file_name)
                    
                    try:
                        logger.info(f"Скачиваю '{remote_filepath}' в '{local_filepath}'...")
                        await client.download(remote_filepath, local_filepath)
                        logger.info(f"Файл '{file_name}' успешно скачан.")
                        
                        loop = asyncio.get_running_loop()
                        transcribed_text = await loop.run_in_executor(
                            None,
                            self.transcriber.transcribe_audio,
                            local_filepath
                        )
                        
                        if transcribed_text:
                            transcriptions.append(f"Транскрибация файла '{file_name}':\n```\n{transcribed_text}\n```")
                            if clear_remote_after_processing:
                                files_to_delete_from_ftp.append(remote_filepath)
                        else:
                            transcriptions.append(f"Не удалось транскрибировать файл '{file_name}'.")

                    except aioftp.StatusCodeError as e:
                        transcriptions.append(f"Ошибка FTP при скачивании '{file_name}': {e}. Пропуск.")
                        logger.error(f"Ошибка FTP при скачивании '{file_name}': {e}")
                    except Exception as e:
                        transcriptions.append(f"Неожиданная ошибка при обработке файла '{file_name}': {e}. Пропуск.")
                        logger.error(f"Неожиданная ошибка при обработке файла '{file_name}': {e}")
                    finally:
                        if os.path.exists(local_filepath):
                            try:
                                os.remove(local_filepath)
                                logger.info(f"Временный локальный файл '{local_filepath}' удален.")
                            except Exception as e:
                                logger.warning(f"Не удалось удалить временный файл '{local_filepath}': {e}")

                if clear_remote_after_processing and files_to_delete_from_ftp:
                    logger.info(f"Удаляю {len(files_to_delete_from_ftp)} обработанных файлов с FTP...")
                    for file_path_to_delete in files_to_delete_from_ftp:
                        try:
                            await client.remove(file_path_to_delete)
                            logger.info(f"Файл '{file_path_to_delete}' удален с FTP.")
                        except aioftp.StatusCodeError as e:
                            logger.error(f"Ошибка FTP при удалении '{file_path_to_delete}': {e}")
                            transcriptions.append(f"Ошибка при удалении файла '{file_path_to_delete}' с FTP: {e}")
                        except Exception as e:
                            logger.error(f"Неожиданная ошибка при удалении файла '{file_path_to_delete}' с FTP: {e}")
                            transcriptions.append(f"Неожиданная ошибка при удалении файла '{file_path_to_delete}' с FTP: {e}")

        except aioftp.StatusCodeError as e:
            return f"Ошибка подключения к FTP-серверу: {e}. Проверьте хост, имя пользователя, пароль и путь."
        except Exception as e:
            return f"Произошла непредвиденная ошибка при работе с FTP: {e}."
        
        if transcriptions:
            return "Результаты обработки аудиофайлов с FTP:\n\n" + "\n\n".join(transcriptions)
        else:
            return "Обработка завершена, но транскрибированный текст не был получен ни для одного файла."