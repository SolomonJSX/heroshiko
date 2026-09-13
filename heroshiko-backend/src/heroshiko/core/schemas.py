from enum import Enum
from typing import Any
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Пресеты ---
class PresetResponse(BaseModel):
    id: str
    title: str
    prompt: str
    negative_prompt: str
    control_strength: float = 0.75
    denoising_strength: float = 0.65
    cover_image: str | None = None
    cover_url: str | None = None
    preview_url: str | None = None
    tags: list[str] = []


class PresetListResponse(BaseModel):
    presets: list[PresetResponse]
    total: int


# --- Биометрия и Анализ лица ---
class FaceAnalysisResponse(BaseModel):
    face_detected: bool
    multiple_faces_detected: bool = False
    total_faces_count: int = 1
    gender: str | None = None
    age: int | None = None
    hair_color: str | None = None
    hair_length: str | None = None
    has_beard: bool = False
    hair_prompt: str | None = None
    build_type: str | None = None
    build_label_ru: str | None = None
    bbox: list[int] | None = None
    message: str = "Успешно"


# --- Параметры генерации ---
class GenerateRequestParams(BaseModel):
    preset_id: str = Field(..., description="ID пресета (например: 'dubai_luxury', 'paris_cafe', 'kpop_idol', 'kpop_group')")
    hair_mode: str = Field("keep", description="Режим прически: 'keep' (сохранить мою), 'change' (сменить прическу), 'preset' (стиль пресета)")
    hair_style: str | None = Field(None, description="Желаемый стиль прически (если hair_mode='change')")
    hair_color: str | None = Field(None, description="Желаемый цвет волос (если hair_mode='change')")
    ethnicity: str = Field("auto", description="Этнотип: 'auto', 'central_asian', 'east_asian', 'european', 'middle_eastern', 'latino'")
    gender: str = Field("auto", description="Пол: 'auto', 'man', 'woman'")
    body_build: str = Field("auto", description="Комплекция тела: 'auto', 'slender', 'athletic', 'plus_size'")
    face_strength: float = Field(0.40, ge=0.1, le=1.5, description="Сила влияния лица в диффузии (IP-Adapter, реком. 0.35-0.45)")
    steps: int = Field(25, ge=10, le=60, description="Количество шагов диффузии (DPM++ 2M Karras, реком. 25)")
    guidance_scale: float = Field(5.5, ge=1.0, le=15.0, description="Guidance scale (CFG, реком. 5.0-6.0)")
    apply_pre_enhance: bool = Field(False, description="Предварительное улучшение качества селфи (Pre-Enhance)")
    apply_face_swap: bool = Field(True, description="100% перенос лица через INSwapper")
    apply_face_restore: bool = Field(True, description="HD-реставрация микродеталей и пор кожи (CodeFormer)")
    fidelity_weight: float = Field(0.80, ge=0.1, le=1.0, description="Точность CodeFormer (реком. 0.80)")
    apply_film_grain: bool = Field(True, description="Кинематографичное зерно (Film Grain)")


# --- Статус фоновой задачи генерации ---
class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int = Field(0, ge=0, le=100)
    preset_id: str
    result_url: str | None = None
    preview_face_url: str | None = None
    error_message: str | None = None
    elapsed_seconds: float | None = None
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Статус сервера и видеопамяти ---
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    cuda_available: bool
    device_name: str | None = None
    vram_allocated_mb: float | None = None
    vram_reserved_mb: float | None = None
    vram_total_mb: float | None = None
    models_loaded: list[str] = []


# --- Внутренние схемы конвейера ---
class GenerationRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image
    prompt: str
    negative_prompt: str = "bad anatomy, blurry, distorted face, extra limbs, low quality, artifact"
    character_preset: str | None = None
    face_strength: float = 0.85
    pose_strength: float = 0.80
    guidance_scale: float = 7.0
    num_steps: int = 30


class GenerationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_image: Image.Image
    pose_preview: Image.Image | None = None
    mask_preview: Image.Image | None = None
