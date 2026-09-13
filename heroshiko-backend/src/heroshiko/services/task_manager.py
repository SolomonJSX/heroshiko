import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from heroshiko.core.schemas import GenerateRequestParams, TaskResponse, TaskStatus
from heroshiko.services.generation_service import HeroshikoService

logger = logging.getLogger("heroshiko.tasks")


class TaskManager:
    _instance: "TaskManager | None" = None
    MAX_STORED_TASKS = 200  # Защита от утечки памяти истории задач

    def __init__(self, storage_dir: Path | str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.outputs_dir = self.storage_dir / "outputs"
        self.uploads_dir = self.storage_dir / "uploads"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self._tasks: dict[str, TaskResponse] = {}
        self._start_times: dict[str, float] = {}
        # Храним в очереди только ID задачи и путь к файлу на диске (экономия RAM)
        self._queue: asyncio.Queue[tuple[str, Path, GenerateRequestParams]] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    @classmethod
    def get_instance(cls, storage_dir: Path | str = "storage") -> "TaskManager":
        if cls._instance is None:
            cls._instance = cls(storage_dir)
        return cls._instance

    def start_worker(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_queue())
            logger.info("Воркер задач генерации Heroshiko успешно запущен.")

    def stop_worker(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            logger.info("Воркер задач остановлен.")

    def _cleanup_old_tasks_if_needed(self) -> None:
        """Очищает устаревшие завершенные задачи при превышении лимита MAX_STORED_TASKS."""
        if len(self._tasks) > self.MAX_STORED_TASKS:
            # Сортируем задачи по времени создания и удаляем старейшие завершенные
            finished_keys = [
                tid for tid, t in self._tasks.items()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            ]
            num_to_delete = len(self._tasks) - self.MAX_STORED_TASKS
            for tid in finished_keys[:num_to_delete]:
                self._tasks.pop(tid, None)
                self._start_times.pop(tid, None)

    async def enqueue_task(
        self,
        image: Image.Image,
        params: GenerateRequestParams,
    ) -> TaskResponse:
        self._cleanup_old_tasks_if_needed()

        task_id = uuid.uuid4().hex[:12]
        now_str = datetime.now().isoformat()

        # Сохраняем исходное фото сразу на диск для экономии RAM (гарантируем RGB для JPEG)
        input_path = self.uploads_dir / f"{task_id}_input.jpg"
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(input_path, format="JPEG", quality=95)

        task_info = TaskResponse(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            progress=0,
            preset_id=params.preset_id,
            created_at=now_str,
        )

        self._tasks[task_id] = task_info
        self._start_times[task_id] = time.time()

        # В очередь передается путь к файлу вместо тяжелого объекта в памяти
        await self._queue.put((task_id, input_path, params))
        logger.info(f"Задача [{task_id}] добавлена в очередь (тема: {params.preset_id}).")
        return task_info

    def get_task(self, task_id: str) -> TaskResponse | None:
        return self._tasks.get(task_id)

    async def _process_queue(self) -> None:
        service = HeroshikoService.get_instance()

        while True:
            try:
                task_id, input_path, params = await self._queue.get()
                logger.info(f"Начало обработки задачи [{task_id}]...")

                task = self._tasks[task_id]
                task.status = TaskStatus.PROCESSING
                task.progress = 15

                start_t = self._start_times.get(task_id, time.time())

                try:
                    # Чтение входного фото с диска
                    input_image = Image.open(input_path).convert("RGB")

                    # Запуск генерации в ThreadPoolExecutor для избежания блокировки asyncio loop
                    loop = asyncio.get_running_loop()
                    task.progress = 30

                    final_img, face_preview, meta = await loop.run_in_executor(
                        None,
                        service.generate_photoshoot,
                        input_image,
                        params,
                    )

                    task.progress = 85

                    # Сохранение результатов на диск
                    out_filename = f"{task_id}.jpg"
                    out_path = self.outputs_dir / out_filename
                    final_img.save(out_path, format="JPEG", quality=95)

                    face_url = None
                    if face_preview is not None:
                        face_filename = f"{task_id}_face.jpg"
                        face_path = self.outputs_dir / face_filename
                        face_preview.save(face_path, format="JPEG", quality=90)
                        face_url = f"/api/v1/images/{face_filename}"

                    elapsed = round(time.time() - start_t, 2)
                    task.status = TaskStatus.COMPLETED
                    task.progress = 100
                    task.result_url = f"/api/v1/images/{out_filename}"
                    task.preview_face_url = face_url
                    task.elapsed_seconds = elapsed
                    task.metadata = meta

                    logger.info(f"Задача [{task_id}] завершена за {elapsed} сек!")

                except Exception as ex:
                    elapsed = round(time.time() - start_t, 2)
                    logger.exception(f"Ошибка при обработке задачи [{task_id}]: {ex}")
                    task.status = TaskStatus.FAILED
                    task.progress = 100
                    task.error_message = str(ex)
                    task.elapsed_seconds = elapsed

                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Непредвиденная ошибка в воркере задач: {e}")
                await asyncio.sleep(1)
