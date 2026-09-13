import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image

from heroshiko.cv.face import FaceIdentity
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID


class HeroshikoSD15Engine:
    """Ультра-быстрый фотореалистичный генеративный движок Heroshiko на базе Stable Diffusion 1.5 
    (Realistic Vision V5.1) и адаптера идентичности IP-Adapter-FaceID.
    """

    def __init__(
        self,
        checkpoint_path: str = "weights/checkpoints/Realistic_Vision_V5.1_fp16-no-ema.safetensors",
        ip_adapter_path: str = "weights/ip_adapter/ip-adapter-faceid_sd15.bin",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.ip_adapter_path = Path(ip_adapter_path)

        print(f"[SD1.5 Engine] Загрузка чекпоинта Realistic Vision V5.1 ({self.checkpoint_path.name}) на {self.device}...")
        t0 = time.time()

        self.pipe = StableDiffusionPipeline.from_single_file(
            str(self.checkpoint_path),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )
        self.pipe.to(self.device)

        print(f"[SD1.5 Engine] Инициализация IP-Adapter FaceID ({self.ip_adapter_path.name})...")
        self.ip_model = IPAdapterFaceID(
            self.pipe,
            str(self.ip_adapter_path),
            self.device,
            lora_rank=128,
            num_tokens=4,
        )
        print(f"[SD1.5 Engine] Движок готов к генерации за {time.time() - t0:.2f}s!")

    def _trim_to_token_limit(self, text: str, max_tokens: int = 77) -> str:
        """Обрезает строку по границе токенов CLIP, чтобы избежать предупреждений transformers."""
        if not text:
            return ""
        tokenizer = getattr(self.pipe, "tokenizer", None)
        if tokenizer is None:
            return text
        tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return tokenizer.decode(tokens, skip_special_tokens=True)

    def generate(
        self,
        face_info: Optional[FaceIdentity],
        prompt: str,
        negative_prompt: str = "",
        face_strength: float = 0.60,
        steps: int = 25,
        guidance_scale: float = 5.5,
        width: int = 512,
        height: int = 768,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Генерация кинематографичного фотопортрета с внедрением идентичности лица."""
        if face_info is not None and face_info.embedding is not None:
            if isinstance(face_info.embedding, torch.Tensor):
                faceid_embeds = face_info.embedding.clone().detach()
            else:
                faceid_embeds = torch.from_numpy(np.array(face_info.embedding))
            if faceid_embeds.ndim == 1:
                faceid_embeds = faceid_embeds.unsqueeze(0)
            faceid_embeds = faceid_embeds.to(self.device, dtype=torch.float16)
        else:
            faceid_embeds = torch.zeros((1, 512), device=self.device, dtype=torch.float16)

        # Базовые критические анатомические защиты (должны быть в самом начале, чтобы гарантированно не отсекаться)
        core_safeguards = "(long neck:1.4), (elongated neck:1.4), (distorted neck:1.3), (disproportional neck:1.3), bad anatomy, deformed hands, extra limbs, poorly drawn face, plastic skin, blurry"
        if negative_prompt:
            # Если в negative_prompt уже есть core_safeguards, не дублируем
            if "long neck" in negative_prompt:
                full_negative = negative_prompt
            else:
                full_negative = f"{core_safeguards}, {negative_prompt}"
        else:
            full_negative = f"{core_safeguards}, (nsfw, nude:1.4), (cropped:1.3), cartoon"

        # Безопасное усечение под точный лимит 77 токенов (исключает предупреждения и мусорные хвосты)
        safe_prompt = self._trim_to_token_limit(prompt, max_tokens=77)
        safe_negative = self._trim_to_token_limit(full_negative, max_tokens=77)

        print(f"[SD1.5 Engine] Генерация ({width}x{height}, шагов: {steps}, CFG: {guidance_scale}, FaceID Scale: {face_strength})...")
        t_start = time.time()

        images = self.ip_model.generate(
            faceid_embeds=faceid_embeds,
            prompt=safe_prompt,
            negative_prompt=safe_negative,
            scale=face_strength,
            num_samples=1,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            seed=seed,
        )

        print(f"[SD1.5 Engine] Генерация завершена за {time.time() - t_start:.2f}s!")
        return images[0]
