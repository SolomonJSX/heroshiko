from PIL import Image
from pydantic import BaseModel, ConfigDict


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
