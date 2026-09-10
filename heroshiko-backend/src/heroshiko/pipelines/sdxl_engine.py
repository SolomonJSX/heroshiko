import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
)
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from heroshiko.core.settings import settings
from heroshiko.cv.face import FaceIdentity


def calculate_sdxl_dimensions(
    orig_w: int,
    orig_h: int,
    max_area: int = 1024 * 1024,
    multiple_of: int = 64,
    min_dim: int = 512,
    max_dim: int = 1536,
) -> tuple[int, int]:
    """
    Вычисляет оптимальные размеры (width, height) для SDXL с сохранением
    исходного соотношения сторон (aspect ratio).

    Результат кратен multiple_of (по умолчанию 64) и сохраняет общую площадь
    в районе ~1 мегапикселя (1024x1024), под которую натренирован SDXL.
    """
    aspect = orig_w / orig_h
    w = int(round((max_area * aspect) ** 0.5))
    h = int(round((max_area / aspect) ** 0.5))

    w = int(round(w / multiple_of) * multiple_of)
    h = int(round(h / multiple_of) * multiple_of)

    w = max(min_dim, min(max_dim, w))
    h = max(min_dim, min(max_dim, h))
    return w, h


class HeroshikoEngine:
    def __init__(
        self,
        base_model_id: str = settings.SDXL_BASE_ID,
        controlnet_id: str = settings.CONTROLNET_OPENPOSE_ID,
        ip_adapter_repo: str = settings.IP_ADAPTER_REPO,
        ip_adapter_weight_name: str = settings.IP_ADAPTER_WEIGHT_NAME,
        image_encoder_id: str = settings.IMAGE_ENCODER_ID,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32

        print("1/4 Загрузка ControlNet (OpenPose SDXL)...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=self.torch_dtype,
        )

        print("2/4 Загрузка фикс-VAE (madebyollin)...")
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.torch_dtype,
        )

        print("3/4 Инициализация SDXL Inpaint пайплайна...")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model_id,
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
        )

        print("4/4 Подключение IP-Adapter FaceID PlusV2...")
        # Image encoder для визуального кропа лица (PlusV2)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_id,
            torch_dtype=self.torch_dtype,
        )
        self.clip_processor = CLIPImageProcessor.from_pretrained(image_encoder_id)

        # Регистрируем Image Encoder в пайплайне для корректной работы с IP-Adapter и CPU offload
        self.pipe.register_modules(image_encoder=self.image_encoder)

        # Загрузка весов IP-Adapter прямо в слои диффузии
        self.pipe.load_ip_adapter(
            ip_adapter_repo,
            subfolder="",
            weight_name=ip_adapter_weight_name,
            image_encoder_folder=None,
        )

        # Оптимизации VRAM под GPU (RTX 3060 12GB):
        # enable_model_cpu_offload автоматически переносит неактивные модули в RAM
        if self.device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_tiling()
        else:
            self.pipe.to(self.device)

    def generate(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        pose_image: Image.Image,
        face_info: FaceIdentity | None,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.75,
        face_strength: float = 0.85,
        strength: float = 0.98,
        target_size: tuple[int, int] | None = None,
        restore_original_size: bool = False,
    ) -> Image.Image:
        """
        Запуск генерации с полной интеграцией:
        - Inpaint маска (одежда + фон)
        - ControlNet (скелет позы)
        - IP-Adapter FaceID (вектор лица + выровненный кроп)
        - Адаптивный расчет разрешения под пропорции (Aspect Ratio) исходного фото
        """
        # Динамический расчет размеров под SDXL с сохранением пропорций
        if target_size is None:
            target_w, target_h = calculate_sdxl_dimensions(image.width, image.height)
        else:
            target_w, target_h = target_size

        init_img = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        mask_img = mask_image.resize((target_w, target_h), Image.Resampling.NEAREST)
        pose_img = pose_image.resize((target_w, target_h), Image.Resampling.NEAREST)

        # Настраиваем силу влияния IP-Adapter на генерацию лица
        self.pipe.set_ip_adapter_scale(face_strength)

        # Подготовка данных лица для FaceID PlusV2
        ip_adapter_image = None
        if face_info is not None:
            # Превращаем кроп лица в тензор CLIP
            clip_inputs = self.clip_processor(
                images=face_info.aligned_face_chip, return_tensors="pt"
            ).pixel_values.to(self.device, dtype=self.torch_dtype)

            # Эмбеддинг InsightFace нормализуем под тип устройства
            face_embeds = face_info.embedding.to(self.device, dtype=self.torch_dtype)

            # Формат FaceID PlusV2: [face_embeds, clip_visual_tokens]
            ip_adapter_image = [[face_embeds, clip_inputs]]

        # Инференс
        generator = torch.Generator(device=self.device).manual_seed(42)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_img,
            mask_image=mask_img,
            control_image=pose_img,
            ip_adapter_image=ip_adapter_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images[0]

        if restore_original_size and (output.size != image.size):
            output = output.resize(image.size, Image.Resampling.LANCZOS)

        return output
