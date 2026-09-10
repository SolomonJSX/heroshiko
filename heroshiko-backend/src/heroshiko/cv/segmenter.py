import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BackgroundSegmenter:
    def __init__(self, model_id: str = "ZhengPeng7/BiRefNet", device: str = "cuda"):
        self.device = device
        # Загружаем модель сегментации в FP16 для экономии VRAM
        self.model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(self.device).eval().half()

        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def get_masks(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """
        Возвращает:
          1. person_mask (белый человек, черный фон)
          2. background_mask (черный человек, белый фон для Inpainting)
        """
        orig_size = image.size
        rgb_image = image.convert("RGB")

        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device).half()
        preds = self.model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        person_mask = transforms.ToPILImage()(pred).resize(orig_size, Image.Resampling.BILINEAR)
        # Инвертируем маску, чтобы указать SDXL область для дорисовывания (фон)
        background_mask = ImageOps.invert(person_mask)

        return person_mask, background_mask
