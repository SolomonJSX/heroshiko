from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


@dataclass
class HumanParsingResult:
    """Контейнер с результатами сегментации."""

    # Бинарная маска зон для полной перерисовки (фон + вся одежда + обувь + аксессуары)
    inpaint_mask: Image.Image
    # Защищенная маска (лицо, волосы, видимая кожа)
    keep_mask: Image.Image
    # Карта сегментации (числовые метки классов от 0 до 17)
    labels_map: np.ndarray


class ClothingParser:
    def __init__(
        self,
        model_id: str = "mattmdjaga/segformer_b2_clothes",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device).eval()

        # Классы, которые мы СТИРАЕМ и генерируем заново под тему пресета:
        # Фон (0), Головной убор (1), Перчатки (3), Очки (4), Верхняя одежда (5),
        # Юбка (6), Штаны (7), Платье (8), Ремень (9), Обувь (10, 11),
        # Сумки/рюкзаки (17), Шарфы (18)
        self.default_replace_classes = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18}

        # Классы, которые мы СТРОГО защищаем:
        # Волосы (2), Лицо/шея (12)
        self.protected_classes = {2, 12}

    @torch.no_grad()
    def parse(self, image: Image.Image) -> np.ndarray:
        """Инференс SegFormer, возвращает 2D массив меток исходного разрешения."""
        orig_w, orig_h = image.size
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits  # shape: [1, num_classes, H/4, W/4]

        # Интерполяция логитов к исходному размеру фотографии
        upsampled_logits = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return labels

    def get_outfit_and_background_mask(
        self,
        image: Image.Image,
        replace_classes: list[int] | None = None,
        dilate_kernel_size: int = 15,
        blur_radius: int = 5,
    ) -> HumanParsingResult:
        """
        Создает комбинированную маску для Inpainting:
        - 255 (белый) = зоны под генерацию (фон + наряд)
        - 0 (черный) = неприкосновенная зона (лицо и прическа)
        """
        classes_to_replace = (
            set(replace_classes) if replace_classes else self.default_replace_classes
        )
        labels = self.parse(image)

        # 1. Бинарная маска под замену (все указанные классы)
        mask_replace = np.isin(labels, list(classes_to_replace)).astype(np.uint8) * 255

        # 2. Защитная маска (лицо + волосы)
        mask_protected = np.isin(labels, list(self.protected_classes)).astype(np.uint8) * 255

        # 3. Морфологическая дилатация (расширение маски замены)
        # Это критически важно: если не расширить маску на 10-15 пикселей,
        # на плечах и шее останется тонкий край старой футболки/фона.
        if dilate_kernel_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
            )
            mask_replace = cv2.dilate(mask_replace, kernel, iterations=1)

        # 4. Убеждаемся, что расширение не залезло на само лицо
        # Приоритет у защищенной маски лица/волос
        mask_replace = np.where(mask_protected == 255, 0, mask_replace).astype(np.uint8)

        # 5. Легкое размытие по Гауссу для бесшовного блендинга краев
        if blur_radius > 0:
            # Делаем размер ядра нечетным
            ksize = blur_radius if blur_radius % 2 != 0 else blur_radius + 1
            mask_replace = cv2.GaussianBlur(mask_replace, (ksize, ksize), 0)

        inpaint_mask_pil = Image.fromarray(mask_replace).convert("L")
        keep_mask_pil = Image.fromarray(mask_protected).convert("L")

        return HumanParsingResult(
            inpaint_mask=inpaint_mask_pil,
            keep_mask=keep_mask_pil,
            labels_map=labels,
        )
