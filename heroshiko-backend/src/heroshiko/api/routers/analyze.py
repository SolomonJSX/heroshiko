import base64
import io
import logging
from PIL import Image
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from heroshiko.core.schemas import FaceAnalysisResponse
from heroshiko.services.generation_service import HeroshikoService

logger = logging.getLogger("heroshiko.api.analyze")
router = APIRouter(prefix="/analyze", tags=["Face & Biometrics Analysis"])


class Base64AnalyzeRequest(BaseModel):
    image_base64: str


def _load_image_from_bytes(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Невозможно декодировать изображение: {e}")


@router.post("", response_model=FaceAnalysisResponse, summary="Быстрый биометрический анализ селфи")
async def analyze_face_upload(
    file: UploadFile = File(..., description="Файл фотографии лица (JPEG, PNG, WEBP)"),
):
    """
    Выполняет мгновенный анализ загруженного селфи:
    - Детекция лица (InsightFace)
    - Определение биологического пола и возраста
    - Анализ цвета, длины прически и наличия бороды
    - Оценка комплекции тела (BodyBuildEstimator)
    - Координаты рамки лица (bbox)

    Используется в мобильном/веб приложении для автоматического предзаполнения настроек стиля.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загружаемый файл должен быть изображением (image/*)")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Файл пуст.")

    pil_image = _load_image_from_bytes(content)
    service = HeroshikoService.get_instance()
    return service.analyze_face(pil_image)


@router.post("/base64", response_model=FaceAnalysisResponse, summary="Биометрический анализ селфи через Base64")
async def analyze_face_base64(payload: Base64AnalyzeRequest):
    """
    Альтернативный эндпоинт для отправки изображения в формате Base64 data-URI или raw base64.
    """
    raw_str = payload.image_base64
    if "," in raw_str:
        raw_str = raw_str.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Невалидный base64: {e}")

    pil_image = _load_image_from_bytes(image_bytes)
    service = HeroshikoService.get_instance()
    return service.analyze_face(pil_image)
