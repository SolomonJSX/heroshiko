from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEVICE: str = "cuda"
    ENABLE_CPU_OFFLOAD: bool = True
    ENABLE_VRAM_ATTENTION_SLICING: bool = True

    # Модель SDXL Inpainting по умолчанию.
    # Для максимального фотореализма кожи и тканей:
    # - "OzzyGT/RealVisXL_V4.0_inpainting" (отличная текстура кожи, естественные тени)
    # - "RunDiffusion/Juggernaut-XL-v9" (кинематографичный свет, реализм)
    SDXL_BASE_ID: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    CONTROLNET_OPENPOSE_ID: str = "thibaud/controlnet-openpose-sdxl-1.0"
    IP_ADAPTER_REPO: str = "h94/IP-Adapter-FaceID"
    IP_ADAPTER_WEIGHT_NAME: str = "ip-adapter-faceid-plusv2_sdxl.bin"
    IMAGE_ENCODER_ID: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    WEIGHTS_DIR: Path = BASE_DIR / "weights"


settings = Settings()
