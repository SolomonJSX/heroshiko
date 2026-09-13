from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/images", tags=["Images & Storage"])

STORAGE_DIR = Path("storage/outputs")


@router.get("/{filename}", summary="Скачать сгенерированное изображение")
async def get_image(filename: str):
    # Защита от path traversal
    safe_name = Path(filename).name
    file_path = STORAGE_DIR / safe_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Изображение не найдено.")

    ext = safe_name.split(".")[-1].lower()
    media_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/jpeg")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )
