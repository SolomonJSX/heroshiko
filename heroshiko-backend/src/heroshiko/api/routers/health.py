from fastapi import APIRouter
from heroshiko.core.schemas import HealthResponse
from heroshiko.services.generation_service import HeroshikoService

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse, summary="Проверка работоспособности и VRAM")
async def get_health():
    """
    Возвращает статус сервера, состояние видеопамяти (VRAM) на GPU и список загруженных нейросетей.
    """
    service = HeroshikoService.get_instance()
    info = service.get_vram_info()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        cuda_available=info["cuda_available"],
        device_name=info["device_name"],
        vram_allocated_mb=info.get("vram_allocated_mb"),
        vram_reserved_mb=info.get("vram_reserved_mb"),
        vram_total_mb=info.get("vram_total_mb"),
        models_loaded=info["models_loaded"],
    )
