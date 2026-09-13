from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEVICE: str = "cuda"
    CHECKPOINT_PATH: str = "weights/checkpoints/Realistic_Vision_V5.1_fp16-no-ema.safetensors"
    IP_ADAPTER_PATH: str = "weights/ip_adapter/ip-adapter-faceid_sd15.bin"
    INSIGHTFACE_MODEL: str = "buffalo_l"

    # Конфигурация API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_IMAGE_DIM: int = 1920
    MAX_STORED_TASKS: int = 200

    # Пути директорий
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    CONFIGS_DIR: Path = BASE_DIR / "configs"


settings = Settings()
