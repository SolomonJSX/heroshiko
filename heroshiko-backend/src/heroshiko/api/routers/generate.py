import io
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image

from heroshiko.core.schemas import (
    GenerateRequestParams,
    TaskResponse,
    TaskStatus,
)
from heroshiko.services.generation_service import HeroshikoService
from heroshiko.services.task_manager import TaskManager

logger = logging.getLogger("heroshiko.api.generate")
router = APIRouter(tags=["Generation & Tasks"])


def _load_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Невозможно декодировать изображение: {e}")


@router.post(
    "/generate",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Создать задачу генерации фотосессии (Асинхронно)",
)
async def generate_async(
    file: UploadFile = File(..., description="Селфи пользователя"),
    preset_id: str = Form(..., description="ID пресета (dubai_luxury, paris_cafe, kpop_idol, kpop_group)"),
    hair_mode: str = Form("keep", description="'keep' (сохранить мою), 'change' (сменить), 'preset' (стиль пресета)"),
    hair_style: Optional[str] = Form(None, description="Стиль прически при смене"),
    hair_color: Optional[str] = Form(None, description="Цвет прически при смене"),
    ethnicity: str = Form("auto", description="Этнотип: auto, central_asian, east_asian, european, middle_eastern, latino"),
    gender: str = Form("auto", description="Пол: auto, man, woman"),
    body_build: str = Form("auto", description="Комплекция: auto, slender, athletic, plus_size"),
    face_strength: float = Form(0.40, description="Сила влияния лица в диффузии (реком: 0.35-0.45)"),
    steps: int = Form(25, description="Шаги инференса диффузии DPM++ 2M Karras (реком: 25)"),
    guidance_scale: float = Form(5.5, description="CFG Scale для чистого естественного света (реком: 5.0-6.0)"),
    apply_pre_enhance: bool = Form(False, description="Предварительное улучшение качества селфи (Pre-Enhance)"),
    apply_face_swap: bool = Form(True, description="100% перенос биометрии INSwapper"),
    apply_face_restore: bool = Form(True, description="HD-реставрация микродеталей и пор кожи CodeFormer"),
    fidelity_weight: float = Form(0.80, description="Точность CodeFormer (реком: 0.80)"),
    apply_film_grain: bool = Form(True, description="Кинематографичное зерно Film Grain"),
):
    """
    Создает фоновую задачу фотосессии.
    Возвращает `task_id`, по которому клиент опрашивает статус через `GET /tasks/{task_id}`.
    """
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Файл селфи пуст.")

    pil_img = _load_image(content)

    params = GenerateRequestParams(
        preset_id=preset_id,
        hair_mode=hair_mode,
        hair_style=hair_style,
        hair_color=hair_color,
        ethnicity=ethnicity,
        gender=gender,
        body_build=body_build,
        face_strength=face_strength,
        steps=steps,
        guidance_scale=guidance_scale,
        apply_pre_enhance=apply_pre_enhance,
        apply_face_swap=apply_face_swap,
        apply_face_restore=apply_face_restore,
        fidelity_weight=fidelity_weight,
        apply_film_grain=apply_film_grain,
    )

    task_mgr = TaskManager.get_instance()
    task = await task_mgr.enqueue_task(pil_img, params)
    return task


@router.get("/tasks/{task_id}", response_model=TaskResponse, summary="Получить статус задачи генерации")
async def get_task_status(task_id: str):
    """
    Возвращает текущий прогресс и статус задачи (`queued`, `processing`, `completed`, `failed`).
    При завершении содержит ссылку `result_url` на скачивание фото.
    """
    task_mgr = TaskManager.get_instance()
    task = task_mgr.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Задача [{task_id}] не найдена.")
    return task


@router.post("/generate/sync", response_model=TaskResponse, summary="Синхронная генерация фотосессии")
async def generate_sync(
    file: UploadFile = File(..., description="Селфи пользователя"),
    preset_id: str = Form(..., description="ID пресета"),
    hair_mode: str = Form("keep"),
    hair_style: Optional[str] = Form(None),
    hair_color: Optional[str] = Form(None),
    ethnicity: str = Form("auto"),
    body_build: str = Form("auto"),
    face_strength: float = Form(0.40),
    steps: int = Form(25),
    guidance_scale: float = Form(5.5),
    apply_pre_enhance: bool = Form(False),
    apply_face_swap: bool = Form(True),
    apply_face_restore: bool = Form(True),
    fidelity_weight: float = Form(0.80),
    apply_film_grain: bool = Form(True),
):
    """
    Блокирующий синхронный вызов генерации. Рекомендуется для локальных тестов и отладки через Swagger UI.
    """
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Файл селфи пуст.")

    pil_img = _load_image(content)

    params = GenerateRequestParams(
        preset_id=preset_id,
        hair_mode=hair_mode,
        hair_style=hair_style,
        hair_color=hair_color,
        ethnicity=ethnicity,
        body_build=body_build,
        face_strength=face_strength,
        steps=steps,
        guidance_scale=guidance_scale,
        apply_pre_enhance=apply_pre_enhance,
        apply_face_swap=apply_face_swap,
        apply_face_restore=apply_face_restore,
        fidelity_weight=fidelity_weight,
        apply_film_grain=apply_film_grain,
    )

    service = HeroshikoService.get_instance()
    task_id = uuid.uuid4().hex[:12]
    start_t = time.time()

    try:
        final_img, face_preview, meta = service.generate_photoshoot(pil_img, params)

        outputs_dir = Path("storage/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)

        out_filename = f"{task_id}.jpg"
        final_img.save(outputs_dir / out_filename, format="JPEG", quality=95)

        face_url = None
        if face_preview is not None:
            face_filename = f"{task_id}_face.jpg"
            face_preview.save(outputs_dir / face_filename, format="JPEG", quality=90)
            face_url = f"/api/v1/images/{face_filename}"

        elapsed = round(time.time() - start_t, 2)
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            preset_id=preset_id,
            result_url=f"/api/v1/images/{out_filename}",
            preview_face_url=face_url,
            elapsed_seconds=elapsed,
            created_at=datetime.now().isoformat(),
            metadata=meta,
        )
    except Exception as e:
        elapsed = round(time.time() - start_t, 2)
        logger.exception(f"Ошибка при синхронной генерации: {e}")
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            progress=100,
            preset_id=preset_id,
            error_message=str(e),
            elapsed_seconds=elapsed,
            created_at=datetime.now().isoformat(),
        )
