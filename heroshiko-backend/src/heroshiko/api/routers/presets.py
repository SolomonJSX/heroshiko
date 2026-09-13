from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from heroshiko.core.schemas import PresetListResponse, PresetResponse
from heroshiko.services.generation_service import HeroshikoService

router = APIRouter(prefix="/presets", tags=["Presets & Styles"])
COVERS_DIR = Path("assets/covers")


def _get_preset_cover_path(preset_id: str, cover_image: str | None = None) -> Path | None:
    if cover_image:
        p = COVERS_DIR / Path(cover_image).name
        if p.exists():
            return p

    # Поиск по id пресета
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = COVERS_DIR / f"{preset_id}{ext}"
        if candidate.exists():
            return candidate
    return None


@router.get("", response_model=PresetListResponse, summary="Получить список всех пресетов фотосессий")
async def list_presets():
    """
    Возвращает каталог всех доступных стилей фотосессий с постерами-обложками (cover_url).
    """
    service = HeroshikoService.get_instance()
    all_presets = service.list_presets()

    items = []
    for p in all_presets:
        cover_path = _get_preset_cover_path(p.id, p.cover_image)
        cover_url = f"/api/v1/presets/{p.id}/cover" if cover_path else None

        items.append(
            PresetResponse(
                id=p.id,
                title=p.title,
                prompt=p.prompt.strip(),
                negative_prompt=p.negative_prompt.strip(),
                control_strength=p.control_strength,
                denoising_strength=p.denoising_strength,
                cover_image=p.cover_image,
                cover_url=cover_url,
                preview_url=cover_url,
                tags=["group"] if "group" in p.id else ["portrait", "lifestyle"],
            )
        )

    return PresetListResponse(presets=items, total=len(items))


@router.get("/{preset_id}", response_model=PresetResponse, summary="Получить информацию о конкретном пресете")
async def get_preset(preset_id: str):
    service = HeroshikoService.get_instance()
    p = service.get_preset(preset_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Пресет '{preset_id}' не найден.")

    cover_path = _get_preset_cover_path(p.id, p.cover_image)
    cover_url = f"/api/v1/presets/{p.id}/cover" if cover_path else None

    return PresetResponse(
        id=p.id,
        title=p.title,
        prompt=p.prompt.strip(),
        negative_prompt=p.negative_prompt.strip(),
        control_strength=p.control_strength,
        denoising_strength=p.denoising_strength,
        cover_image=p.cover_image,
        cover_url=cover_url,
        preview_url=cover_url,
        tags=["group"] if "group" in p.id else ["portrait", "lifestyle"],
    )


@router.get("/{preset_id}/cover", summary="Скачать изображение-обложку (постер) пресета")
async def get_preset_cover(preset_id: str):
    """
    Возвращает файл постера/обложки пресета для отображения карточки стиля в UI приложения.
    """
    service = HeroshikoService.get_instance()
    p = service.get_preset(preset_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Пресет '{preset_id}' не найден.")

    cover_path = _get_preset_cover_path(preset_id, p.cover_image)
    if not cover_path or not cover_path.exists():
        raise HTTPException(status_code=404, detail=f"Постер для пресета '{preset_id}' не найден.")

    ext = cover_path.suffix.lower().replace(".", "")
    media_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/jpeg")

    return FileResponse(
        path=str(cover_path),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.post("/{preset_id}/cover", summary="Загрузить/обновить обложку (постер) пресета")
async def upload_preset_cover(
    preset_id: str,
    file: UploadFile = File(..., description="Изображение постера (JPG/PNG/WEBP)"),
):
    """
    Позволяет обновить обложку темы/пресета прямо через API.
    """
    service = HeroshikoService.get_instance()
    p = service.get_preset(preset_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Пресет '{preset_id}' не найден.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Файл пуст.")

    COVERS_DIR.mkdir(parents=True, exist_ok=True)
    ext = file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else "jpg"
    target_filename = f"{preset_id}.{ext}"
    target_path = COVERS_DIR / target_filename

    with open(target_path, "wb") as f:
        f.write(content)

    return {
        "status": "ok",
        "preset_id": preset_id,
        "cover_url": f"/api/v1/presets/{preset_id}/cover",
        "message": "Обложка успешно обновлена!",
    }
