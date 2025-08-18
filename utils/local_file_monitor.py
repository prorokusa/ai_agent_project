# utils/local_file_monitor.py
import os
import asyncio
import logging
import time # Для демонстрационного теста

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Добавляем StreamHandler только если его еще нет, чтобы избежать дублирования логов
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class LocalFileMonitor:
    """
    Модуль для отслеживания появления новых файлов в указанной локальной папке.
    Использует метод периодического опроса.
    """
    def __init__(self, directory_path: str, interval: int = 5, callback_func=None, allowed_extensions: list = None):
        """
        Инициализирует монитор файлов.

        :param directory_path: Путь к папке для отслеживания.
        :param interval: Интервал проверки в секундах.
        :param callback_func: Асинхронная функция, которая будет вызвана для каждого нового файла.
                              Принимает один аргумент: полный путь к новому файлу.
                              Если None, то новые файлы будут только логироваться.
        :param allowed_extensions: Список расширений файлов для отслеживания (например, ['.mp3', '.wav', '.awb']).
                                   Если None или пустой, отслеживаются все файлы. Расширения должны быть в нижнем регистре.
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Указанный путь '{directory_path}' не является существующей директорией.")

        self.directory_path = directory_path
        self.interval = interval
        self.callback_func = callback_func
        # Убедимся, что расширения в нижнем регистре для корректного сравнения
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions] if allowed_extensions else []
        
        self.known_files = set() # Множество для хранения путей к уже известным файлам
        self._load_initial_files() # Загружаем файлы, которые есть в директории при старте
        self._running = False
        self._monitor_task = None # Задача asyncio для фонового мониторинга
        
        logger.info(f"LocalFileMonitor инициализирован для папки: '{self.directory_path}' с интервалом {self.interval}с.")
        if self.allowed_extensions:
            logger.info(f"Отслеживаемые расширения: {', '.join(self.allowed_extensions)}")
        else:
            logger.info("Отслеживаются все файлы (без фильтрации по расширению).")


    def _load_initial_files(self):
        """
        Загружает список файлов, существующих при старте,
        чтобы не обрабатывать их как новые при первом запуске.
        """
        try:
            for item in os.listdir(self.directory_path):
                full_path = os.path.join(self.directory_path, item)
                # Убедимся, что это файл и что его расширение разрешено
                if os.path.isfile(full_path) and \
                   (not self.allowed_extensions or os.path.splitext(item)[1].lower() in self.allowed_extensions):
                    self.known_files.add(full_path)
            logger.info(f"Загружено {len(self.known_files)} существующих файлов при инициализации.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке начальных файлов из '{self.directory_path}': {e}")

    async def _check_for_new_files(self):
        """
        Проверяет папку на наличие новых файлов, сравнивая текущий список
        с ранее известным.
        """
        try:
            current_files = set()
            for item in os.listdir(self.directory_path):
                full_path = os.path.join(self.directory_path, item)
                # Убедимся, что это файл и что его расширение разрешено
                if os.path.isfile(full_path) and \
                   (not self.allowed_extensions or os.path.splitext(item)[1].lower() in self.allowed_extensions):
                    current_files.add(full_path)
            
            # Находим файлы, которые есть сейчас, но не было раньше
            new_files = current_files - self.known_files
            
            if new_files:
                logger.info(f"Обнаружены новые файлы: {len(new_files)}")
                for new_file_path in new_files:
                    logger.info(f"Новый файл обнаружен: {new_file_path}")
                    if self.callback_func:
                        # Вызываем асинхронную callback-функцию
                        await self.callback_func(new_file_path)
                    else:
                        logger.warning(f"Новый файл '{new_file_path}' обнаружен, но функция обратного вызова не задана.")
                self.known_files.update(new_files) # Добавляем новые файлы в список известных
            # else:
            #     logger.debug("Новых файлов не обнаружено.") # Закомментировано для уменьшения логов

        except Exception as e:
            logger.error(f"Ошибка при проверке новых файлов в '{self.directory_path}': {e}")

    async def start_monitoring(self):
        """
        Начинает фоновый мониторинг папки в асинхронном режиме.
        Эта функция должна быть запущена как asyncio.Task.
        """
        if self._running:
            logger.warning("Мониторинг уже запущен.")
            return

        self._running = True
        logger.info("Запуск мониторинга файлов...")
        while self._running:
            await self._check_for_new_files()
            await asyncio.sleep(self.interval)
        logger.info("Мониторинг файлов остановлен.")

    def stop_monitoring(self):
        """Останавливает фоновый мониторинг папки."""
        if self._running:
            self._running = False
            # Если задача мониторинга запущена, пытаемся ее отменить.
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                logger.info("Задача мониторинга отменена.")
            logger.info("Запрошена остановка мониторинга.")
        else:
            logger.warning("Мониторинг не запущен.")

    # Добавим метод для запуска как asyncio.Task
    def run_as_task(self):
        """Создает и запускает мониторинг как asyncio.Task."""
        self._monitor_task = asyncio.create_task(self.start_monitoring())
        return self._monitor_task

# Для тестирования модуля (запускается, если этот файл запускается напрямую)
if __name__ == "__main__":
    async def dummy_callback(file_path: str):
        """Пример асинхронной callback-функции."""
        print(f"\n[CALLBACK]: Получен новый файл: {file_path}")
        # Здесь могла бы быть логика транскрибации/обработки
        await asyncio.sleep(2) # Имитация длительной операции
        print(f"[CALLBACK]: Обработка файла '{file_path}' завершена.")

    test_monitor_dir = "./monitor_test_folder"
    os.makedirs(test_monitor_dir, exist_ok=True) # Создаем тестовую директорию

    print(f"Создана директория для тестирования: {test_monitor_dir}")
    print("Чтобы протестировать:")
    print(f"1. Запустите этот скрипт: python {os.path.basename(__file__)}")
    print(f"2. Добавляйте файлы (.mp3, .wav, .awb) в папку '{test_monitor_dir}'.")
    print("   Мониторинг будет проверять каждые 3 секунды.")
    print("   Нажмите Ctrl+C для завершения.")

    # Мониторим только аудиофайлы, которые может обработать наш транскрибатор
    allowed_audio_extensions = [
        '.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm', # Поддерживаемые OpenAI API
        '.awb', '.amr' # Те, что pydub может конвертировать
    ]

    # Инициализируем монитор
    monitor = LocalFileMonitor(
        directory_path=test_monitor_dir,
        interval=3, # Проверяем каждые 3 секунды
        callback_func=dummy_callback,
        allowed_extensions=allowed_audio_extensions
    )

    async def run_test_monitor():
        """Основная функция для запуска теста монитора."""
        monitor_task = monitor.run_as_task() # Запускаем монитор как задачу asyncio
        try:
            # Держим основной event loop активным
            await asyncio.Future() # Это будет ждать бесконечно, пока задача не будет отменена
        except asyncio.CancelledError:
            print("Основной тест отменен.")
        finally:
            monitor.stop_monitoring()
            # Ждем, пока задача мониторинга завершится после отмены
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass # Ожидаемо, что задача будет отменена
            print("Тестирование монитора завершено.")

    try:
        asyncio.run(run_test_monitor())
    except KeyboardInterrupt:
        print("\nПолучено прерывание (Ctrl+C). Завершение...")
        # При KeyboardInterrupt asyncio.run() сам поднимает CancelledError,
        # что приводит к отмене run_test_monitor() и его финализации.