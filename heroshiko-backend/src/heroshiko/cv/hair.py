from dataclasses import dataclass
import numpy as np
from PIL import Image

from heroshiko.cv.parser import ClothingParser


@dataclass
class HairAttributes:
    color: str
    length: str
    hair_prompt: str
    has_beard: bool = False


class HairAnalyzer:
    """
    Анализатор прически на основе семантической сегментации SegFormer:
    автоматически определяет цвет волос, длину и формирует точные промпты
    для сохранения индивидуальной прически пользователя в SDXL.
    """

    def __init__(self, parser: ClothingParser | None = None, device: str = "cpu"):
        self.parser = parser
        self.device = device

    def _get_parser(self) -> ClothingParser:
        if self.parser is None:
            self.parser = ClothingParser(device=self.device)
        return self.parser

    def analyze(
        self,
        image: Image.Image,
        face_bbox: np.ndarray | None = None,
    ) -> HairAttributes:
        orig_w, orig_h = image.size
        # Масштабируем до 512x512 для ускорения инференса (~0.08 сек)
        scale_size = (512, 512)
        small_img = image.resize(scale_size, Image.Resampling.BILINEAR)

        parser = self._get_parser()
        labels = parser.parse(small_img)

        # Класс 2 = Волосы (Hair) в SegFormer Clothes
        hair_mask = labels == 2

        if not np.any(hair_mask):
            # Фолбэк, если волосы закрыты или очень короткая стрижка
            return HairAttributes(
                color="natural dark",
                length="short",
                hair_prompt="neat short natural haircut, natural hairline",
            )

        # 1. Определение цвета волос по средним значениям RGB и яркости
        img_np = np.array(small_img)
        hair_pixels = img_np[hair_mask]
        mean_rgb = np.mean(hair_pixels, axis=0)
        r, g, b = mean_rgb

        # Вычисление относительной яркости (Perceived luminance)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        if luminance < 50:
            color = "natural black"
        elif luminance < 95:
            color = "dark brown"
        elif luminance < 145:
            if r > g * 1.15 and r > b * 1.3:
                color = "auburn reddish brown"
            else:
                color = "chestnut brown"
        elif luminance < 185:
            color = "blonde"
        else:
            color = "platinum blonde or light grey"

        # 2. Определение длины волос на голове (исключаем бороду и щетину у подбородка)
        hair_y_indices = np.where(hair_mask)[0]

        has_beard = False
        if face_bbox is not None:
            scale_y = 512.0 / orig_h
            y1 = face_bbox[1] * scale_y
            y2 = face_bbox[3] * scale_y
            face_h = max(y2 - y1, 1.0)
            mouth_y = y1 + 0.75 * face_h
            chin_y = y2

            # Разделяем волосы на голове и растительность на лице (бороду/щетину)
            head_hair_y = hair_y_indices[hair_y_indices < mouth_y]
            beard_pixels = hair_y_indices[hair_y_indices >= mouth_y]
            has_beard = len(beard_pixels) > (0.15 * len(hair_y_indices))

            if len(head_hair_y) > 0:
                p90_head = np.percentile(head_hair_y, 90)
                # Волосы спадают заметно ниже подбородка -> длинные
                if p90_head > chin_y + 25:
                    length = "long"
                # Волосы спадают до уровня подбородка/плеч -> средняя длина
                elif p90_head > y1 + 0.92 * face_h:
                    length = "medium-length"
                # Волосы выше уровня мочки уха/подбородка (обычная стрижка) -> короткая
                else:
                    length = "short"
            else:
                length = "short"
        else:
            p80_y = np.percentile(hair_y_indices, 80)
            if p80_y > 350:
                length = "long"
            elif p80_y > 220:
                length = "medium-length"
            else:
                length = "short"

        if length == "short":
            hair_desc = f"neat short {color} haircut, short fade on sides, clean natural hairline"
        elif length == "medium-length":
            hair_desc = f"medium-length {color} hairstyle, natural hair texture"
        else:
            hair_desc = f"long flowing {color} hair, natural volume and texture"

        if has_beard:
            hair_desc += ", matching neat light facial hair"

        return HairAttributes(
            color=color,
            length=length,
            hair_prompt=hair_desc,
            has_beard=has_beard,
        )
