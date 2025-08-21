import os
import asyncio
import logging
import tempfile
import ftplib
import time
from typing import List, Dict, Any, Optional
from datetime import datetime # <-- Убедитесь, что это единственный импорт datetime.
from concurrent.futures import ThreadPoolExecutor

from utils.audio_transcriber import AudioTranscriber
from core.agent import AIAgent
import config

logger = logging.getLogger(__name__)

class FtpMonitor:
    """
    Класс для фонового мониторинга FTP-сервера на наличие новых аудиофайлов,
    их транскрибации и передачи результатов AIAgent'у.
    """
    def __init__(self, agent: AIAgent):
        self.agent = agent
        self.transcriber = AudioTranscriber()
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        self.agent_owner_name = os.getenv("AGENT_OWNER_NAME", "Пользователь")
        self.enabled = config.FTP_MONITOR_ENABLED
        self.interval = config.FTP_MONITOR_INTERVAL_SECONDS
        self.ftp_host = os.getenv("FTP_HOST")
        self.ftp_user = os.getenv("FTP_USER")
        self.ftp_password = os.getenv("FTP_PASSWORD") 
        self.remote_path = config.FTP_MONITOR_REMOTE_PATH
        self.local_download_dir = config.FTP_MONITOR_LOCAL_DOWNLOAD_DIR
        self.allowed_extensions = config.FTP_MONITOR_ALLOWED_EXTENSIONS
        self.clear_remote_after_processing = config.FTP_MONITOR_CLEAR_REMOTE_AFTER_PROCESSING

        if not self.enabled:
            logger.info("FTP-мониторинг отключен в config.py.")
            return

        if not self.ftp_host or not self.ftp_user or not self.ftp_password:
            logger.error("FTP-учетные данные (FTP_HOST, FTP_USER, FTP_PASSWORD) не найдены в .env. FTP-мониторинг будет нефункционален.")
            self.enabled = False
            return

        os.makedirs(self.local_download_dir, exist_ok=True)
        self.processed_files_history = {} 
        
        self._monitoring_task: Optional[asyncio.Task] = None
        logger.info(f"FTP-мониторинг инициализирован для '{self.ftp_host}{self.remote_path}' "
                    f"с интервалом {self.interval}с. Локальная папка: '{self.local_download_dir}'")
        if self.clear_remote_after_processing:
            logger.warning("ВНИМАНИЕ: Настроено удаление файлов с FTP после обработки.")

    def _parse_custom_date_format(self, date_str: str) -> float:
        """
        Парсит кастомный формат даты 'MM-DD-YY HH:MMAM/PM' в timestamp.
        Пример: '08-15-25 02:37PM' -> timestamp
        """
        try:
            formats_to_try = [
                '%m-%d-%y %I:%M%p',   # 08-15-25 02:37PM
                '%m-%d-%y %I:%M %p',  # 08-15-25 02:37 PM
                '%m/%d/%y %I:%M%p',   # 08/15/25 02:37PM
                '%m/%d/%y %I:%M %p',  # 08/15/25 02:37 PM
            ]
            
            for fmt in formats_to_try:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return time.mktime(dt.timetuple())
                except ValueError:
                    continue
            
            logger.warning(f"Не удалось распарсить дату: {date_str}, использую текущее время")
            return time.time()
            
        except Exception as e:
            logger.warning(f"Ошибка парсинга даты '{date_str}': {e}, использую текущее время")
            return time.time()

    def _create_ftp_connection(self):
        """
        Создает и возвращает подключение к FTP.
        """
        try:
            # Пробуем разные кодировки для подключения
            # Начинаем с cp1251, так как это наиболее частая проблема для русскоязычных FTP
            encodings_to_try = ['cp1251', 'utf-8', 'latin-1', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    # ftplib.FTP может принимать host, user, password, acct, timeout, port, source_address, encoding, ssl_context, *, keyfile, certfile, cert_reqs, check_hostname
                    # Передаем encoding в конструктор FTP
                    ftp = ftplib.FTP(self.ftp_host, encoding=encoding)
                    
                    ftp.login(self.ftp_user, self.ftp_password)
                    ftp.cwd(self.remote_path)
                    logger.debug(f"Успешное подключение к FTP с кодировкой: {encoding}")
                    return ftp
                    
                except Exception as e:
                    logger.debug(f"Кодировка {encoding} не сработала для подключения или CWD: {e}")
                    # Если ошибка, закрываем ftp-соединение, чтобы попробовать следующую кодировку
                    if 'ftp' in locals() and ftp:
                        try: ftp.quit()
                        except: pass
                    continue
            
            # Если ни одна кодировка не подошла
            raise Exception("Не удалось подключиться к FTP ни с одной из известных кодировок.")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к FTP: {e}. Проверьте host, user, password, path и кодировки.")
            raise

    def _list_ftp_files_sync(self) -> List[Dict[str, Any]]:
        """
        Синхронная функция для получения списка файлов на FTP.
        Теперь использует RETRBINARY для LIST и ручное декодирование с перебором кодировок.
        """
        files_info = []
        ftp = None
        try:
            ftp = self._create_ftp_connection()
            
            # Получаем сырые байты из LIST
            line_bytes_list = []
            ftp.retrbinary('LIST', line_bytes_list.append)
            
            # Декодируем байты в строки, перебирая кодировки
            candidate_lines = []
            encodings_for_list = ['cp1251', 'utf-8', 'latin-1', 'iso-8859-1']
            
            for line_bytes in line_bytes_list:
                decoded_line = None
                for enc in encodings_for_list:
                    try:
                        decoded_line = line_bytes.decode(enc, errors='strict') # errors='strict' чтобы поймать ошибку и попробовать другую
                        break # Если декодирование успешно, используем эту кодировку для строки
                    except UnicodeDecodeError:
                        continue # Пробуем следующую кодировку
                
                if decoded_line is None:
                    # Если ни одна строгая кодировка не сработала, делаем последнюю попытку с ignore
                    decoded_line = line_bytes.decode('utf-8', errors='ignore') 
                    logger.warning(f"Не удалось строго декодировать строку LIST. Использован errors='ignore'. Сырая строка: {line_bytes}")
                candidate_lines.append(decoded_line)
            
            # Парсим декодированные строки
            for line in candidate_lines:
                if not line.strip():
                    continue
                    
                try:
                    parts = line.split()
                    
                    if len(parts) < 4:
                        continue
                    
                    is_windows_format = False
                    is_unix_format = False

                    # Проверка на Unix-подобный формат (permissions links owner group size month day time_or_year filename)
                    if len(parts) >= 9 and parts[0].startswith(('-', 'd', 'l', 'c', 'b')): # - (file), d (dir), l (symlink), c (char device), b (block device)
                        is_unix_format = True
                    # Проверка на Windows-подобный формат (MM-DD-YY HH:MMPM <DIR> size filename)
                    elif len(parts) >= 3 and '-' in parts[0] and ':' in parts[1]: # Basic date/time pattern
                        is_windows_format = True

                    file_name = ''
                    file_size = 0
                    file_time = time.time()
                    
                    if is_unix_format:
                        if parts[0].startswith('d'): # Пропускаем директории
                            continue
                        
                        size_str = parts[4]
                        # Убедимся, что размер - число
                        if not size_str.isdigit():
                            logger.warning(f"Unix-like LIST: Invalid size '{size_str}' in line: {line}")
                            continue

                        size = int(size_str)
                        month = parts[5]
                        day = parts[6]
                        time_or_year = parts[7] 
                        name = ' '.join(parts[8:]) # Имя файла может содержать пробелы

                        current_year = datetime.now().year 
                        
                        # Парсинг времени/года
                        if ':' in time_or_year: # HH:MM -> текущий год
                            date_str_full = f"{month} {day} {current_year} {time_or_year}"
                            dt_obj = datetime.strptime(date_str_full, "%b %d %Y %H:%M")
                        elif len(time_or_year) == 4 and time_or_year.isdigit(): # YYYY
                            date_str_full = f"{month} {day} {time_or_year}"
                            dt_obj = datetime.strptime(date_str_full, "%b %d %Y")
                        else:
                            logger.warning(f"Unix-like LIST: Unknown time/year format '{time_or_year}' in line: {line}")
                            dt_obj = datetime.fromtimestamp(time.time()) # Fallback to current time
                            
                        file_time = time.mktime(dt_obj.timetuple())

                        files_info.append({"name": name, "size": size, "time": file_time})
                        logger.debug(f"Найден Unix-like файл: {name}, size: {size}, time: {file_time}")
                    
                    elif is_windows_format:
                        date_part = parts[0]
                        time_part = parts[1]
                        
                        # Обработка <DIR>
                        if len(parts) > 2 and parts[2] == '<DIR>':
                            # Это директория, пропускаем
                            continue
                        
                        size_str = parts[2]
                        if not size_str.isdigit():
                            logger.warning(f"Windows-like LIST: Invalid size '{size_str}' in line: {line}")
                            continue

                        size = int(size_str)
                        name = ' '.join(parts[3:]) # Имя файла может содержать пробелы
                        
                        date_time_str = f"{date_part} {time_part}"
                        file_time = self._parse_custom_date_format(date_time_str)
                            
                        files_info.append({"name": name, "size": size, "time": file_time})
                        logger.debug(f"Найден Windows-like файл: {name}, size: {size}, time: {file_time}")

                except (ValueError, IndexError, KeyError) as parse_e:
                    logger.warning(f"Не удалось распарсить строку LIST '{line}': {parse_e}")
                    continue
            
        except ftplib.all_errors as e:
            logger.error(f"Ошибка при получении списка файлов (LIST): {e}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при _list_ftp_files_sync: {e}")
        finally:
            if ftp:
                try:
                    ftp.quit()
                except:
                    pass
        
        return files_info

    def _download_file_sync(self, file_name: str) -> Optional[str]:
        """
        Синхронная функция для скачивания файла с FTP.
        """
        local_filepath = os.path.join(self.local_download_dir, file_name)
        ftp = None
        try:
            ftp = self._create_ftp_connection()
            
            logger.info(f"Скачиваю '{file_name}' в '{local_filepath}'...")
            with open(local_filepath, 'wb') as f:
                ftp.retrbinary(f'RETR {file_name}', f.write)
            
            logger.info(f"Файл '{file_name}' успешно скачан.")
            return local_filepath
            
        except Exception as e:
            logger.error(f"Ошибка при скачивании файла '{file_name}': {e}")
            return None
        finally:
            if ftp:
                try:
                    ftp.quit()
                except Exception:
                    pass

    def _delete_file_sync(self, file_name: str) -> bool:
        """
        Синхронная функция для удаления файла с FTP.
        """
        ftp = None
        try:
            ftp = self._create_ftp_connection()
            ftp.delete(file_name)
            logger.info(f"Файл '{file_name}' удален с FTP.")
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении файла '{file_name}' с FTP: {e}")
            return False
        finally:
            if ftp:
                try:
                    ftp.quit()
                except Exception:
                    pass

    async def _load_initial_ftp_files(self):
        """
        Загружает список файлов, существующих на FTP при старте.
        """
        if not self.enabled:
            return

        logger.info("Загружаю начальный список файлов с FTP...")
        try:
            loop = asyncio.get_running_loop()
            current_remote_files_info = await loop.run_in_executor(
                self.thread_pool, self._list_ftp_files_sync
            )
            
            for file_info in current_remote_files_info:
                self.processed_files_history[file_info["name"]] = {
                    "size": file_info["size"], 
                    "time": file_info["time"]
                }
            
            logger.info(f"Загружено {len(self.processed_files_history)} существующих файлов с FTP при инициализации.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке начальных файлов с FTP: {e}")

    async def _check_for_new_files(self):
        """
        Проверяет FTP-папку на наличие новых или измененных файлов.
        """
        if not self.enabled:
            return

        logger.info(f"Проверка FTP-сервера на наличие новых файлов в '{self.remote_path}'...")
        
        try:
            loop = asyncio.get_running_loop()
            current_remote_files_info = await loop.run_in_executor(
                self.thread_pool, self._list_ftp_files_sync
            )
            
            logger.debug(f"Найдено файлов: {len(current_remote_files_info)}")

            new_files_to_process = []
            for file_info in current_remote_files_info:
                file_name = file_info["name"]
                file_size = file_info["size"]
                file_time = file_info["time"]
                file_extension = os.path.splitext(file_name)[1].lower()

                is_new = file_name not in self.processed_files_history
                
                has_changed = False
                if not is_new:
                    old_info = self.processed_files_history[file_name]
                    if old_info["size"] != file_size:
                        has_changed = True
                        logger.debug(f"File {file_name} changed size (old: {old_info['size']}, new: {file_size})")
                    if old_info["time"] < file_time:
                        has_changed = True
                        logger.debug(f"File {file_name} changed time (old: {old_info['time']}, new: {file_time})")

                is_allowed_extension = file_extension in self.allowed_extensions
                logger.debug(f"File {file_name}: is_new={is_new}, has_changed={has_changed}, allowed_ext={is_allowed_extension}")

                if (is_new or has_changed) and is_allowed_extension:
                    new_files_to_process.append(file_info)
                    self.processed_files_history[file_name] = {"size": file_size, "time": file_time}

            if new_files_to_process:
                logger.info(f"Обнаружено {len(new_files_to_process)} новых/измененных аудиофайлов на FTP.")
                for file_info in new_files_to_process:
                    await self._process_detected_file(file_info)
            else:
                logger.info(f"Новых или измененных аудиофайлов в '{self.remote_path}' не обнаружено.")

        except Exception as e:
            logger.error(f"Ошибка при проверке новых файлов на FTP: {e}")

    async def _process_detected_file(self, file_info: Dict[str, Any]):
        """
        Обрабатывает один обнаруженный файл.
        """
        file_name = file_info["name"]
        
        try:
            loop = asyncio.get_running_loop()
            local_filepath = await loop.run_in_executor(
                self.thread_pool, self._download_file_sync, file_name
            )
            
            if not local_filepath:
                logger.error(f"Не удалось скачать файл '{file_name}'")
                return
            
            transcribed_text = await loop.run_in_executor(
                self.thread_pool,
                self.transcriber.transcribe_audio,
                local_filepath
            )
            
            if transcribed_text:
                logger.info(f"Файл '{file_name}' транскрибирован. Передаю агенту.")
                
                current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
                
                # --- ЭТОТ БЛОК ФОРМИРУЕТ ПРОМПТ ДЛЯ LLM ---
                agent_prompt = (
                    f"Текущая дата и время: {current_datetime_str}\n"
                    f"Разговор вел(а) {self.agent_owner_name}.\n\n" # <--- ИМЯ ВЛАДЕЛЬЦА ЗДЕСЬ
                    f"Новый аудиофайл '{file_name}' был обнаружен на FTP-сервере и успешно транскрибирован. "
                    f"Пожалуйста, проанализируйте следующий текст и выполните необходимые действия. "
                    f"Транскрибированный текст:\n\n```\n{transcribed_text}\n```" # <--- ВЕСЬ ТЕКСТ ТРАНСКРИПЦИИ ЗДЕСЬ
                )
                await self.agent.process_message(text_input=agent_prompt)

                if self.clear_remote_after_processing:
                    await loop.run_in_executor(
                        self.thread_pool, self._delete_file_sync, file_name
                    )
            else:
                logger.warning(f"Не удалось транскрибировать файл '{file_name}'.")

        except Exception as e:
            logger.error(f"Ошибка при обработке файла '{file_name}': {e}")
        finally:
            if local_filepath and os.path.exists(local_filepath):
                try:
                    os.remove(local_filepath)
                    logger.info(f"Временный локальный файл '{local_filepath}' удален.")
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл '{local_filepath}': {e}")

    async def start_monitoring(self):
        """
        Запускает фоновый цикл мониторинга FTP.
        """
        if not self.enabled:
            logger.info("FTP-мониторинг не будет запущен, так как он отключен или не настроен.")
            return

        logger.info("Запуск фонового FTP-мониторинга...")
        while True:
            try:
                await self._check_for_new_files()
            except asyncio.CancelledError:
                logger.info("Задача FTP-мониторинга отменена.")
                break
            except Exception as e:
                logger.error(f"Критическая ошибка в цикле FTP-мониторинга: {e}")
            await asyncio.sleep(self.interval)

    def run_as_task(self):
        """
        Создает и запускает мониторинг как asyncio.Task.
        """
        if not self.enabled:
            return None
        self._monitoring_task = asyncio.create_task(self.start_monitoring())
        return self._monitoring_task

    def stop_monitoring(self):
        """
        Останавливает фоновый мониторинг.
        """
        if self._monitoring_task:
            self._monitoring_task.cancel()
            logger.info("Запрошена остановка FTP-мониторинга.")
        
        self.thread_pool.shutdown(wait=False)

    def __del__(self):
        """Очистка ресурсов."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)