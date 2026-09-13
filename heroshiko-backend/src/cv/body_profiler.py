import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from src.preprocessors.segmentation import SegmentationService


@dataclass
class BodyMetrics:
    """
    Геометрические замеры пропорций тела.
    """
    shoulder_width: float           # Ширина плеч (в пикселях)
    waist_width: float              # Ширина талии (в пикселях)
    hip_width: float                # Ширина бедер (в пикселях)
    body_height: float              # Полный рост силуэта от макушки до стоп
    relative_shoulder: float        # Относительная ширина плеч (W_shoulder / Height)
    relative_waist: float           # Относительная ширина талии (W_waist / Height)
    silhouette_density: float       # Плотность силуэта (площадь человека / площадь bbox)

    @property
    def composite_build_index(self) -> float:
        """Интегральный индекс плотности комплекции (плечи + талия) / рост."""
        return (self.relative_shoulder + self.relative_waist) / 2.0


class BodyProfiler:
    """
    Модуль геометрии тела (3.2).
    Анализирует фигуру человека в полный рост через SegFormer/RMBG,
    рассчитывает Scale Factor и строит адаптивный силуэт (маску) под комплекцию.
    """
    def __init__(self, seg_service: Optional[SegmentationService] = None):
        self.seg_service = seg_service or SegmentationService()

    def analyze_silhouette(self, image: Image.Image) -> Tuple[BodyMetrics, Image.Image]:
        """
        Извлекает силуэт человека и рассчитывает анатомические метрики.
        Возвращает BodyMetrics и бинарную маску силуэта.
        """
        # Извлекаем маску тела человека через RMBG
        human_rgba = self.seg_service.remove_background(image)
        mask_np = np.array(human_rgba.split()[-1])
        mask_binary = (mask_np > 128).astype(np.uint8)

        # Находим координаты контура силуэта
        y_indices, x_indices = np.where(mask_binary > 0)
        if len(y_indices) == 0:
            raise ValueError("[BodyProfiler] Силуэт человека не обнаружен на фото.")

        y_top, y_bottom = int(np.min(y_indices)), int(np.max(y_indices))
        x_left, x_right = int(np.min(x_indices)), int(np.max(x_indices))
        body_h = max(y_bottom - y_top, 1)

        # Анатомические срезы тела (в долях от высоты силуэта H)
        # 1. Плечи: ~16% - 24% от макушки
        y_sh_start, y_sh_end = int(y_top + 0.16 * body_h), int(y_top + 0.24 * body_h)
        # 2. Талия: ~38% - 46% от макушки
        y_w_start, y_w_end = int(y_top + 0.38 * body_h), int(y_top + 0.46 * body_h)
        # 3. Бедра: ~48% - 56% от макушки
        y_h_start, y_h_end = int(y_top + 0.48 * body_h), int(y_top + 0.56 * body_h)

        def measure_slice_width(y1, y2):
            slice_mask = mask_binary[y1:y2, :]
            row_widths = np.sum(slice_mask, axis=1)
            row_widths = row_widths[row_widths > 0]
            return float(np.median(row_widths)) if len(row_widths) > 0 else 0.0

        shoulder_w = measure_slice_width(y_sh_start, y_sh_end)
        waist_w = measure_slice_width(y_w_start, y_w_end)
        hip_w = measure_slice_width(y_h_start, y_h_end)

        # Индекс заполнения bounding box
        bbox_area = max((y_bottom - y_top) * (x_right - x_left), 1)
        density = float(np.sum(mask_binary)) / float(bbox_area)

        metrics = BodyMetrics(
            shoulder_width=shoulder_w,
            waist_width=waist_w,
            hip_width=hip_w,
            body_height=float(body_h),
            relative_shoulder=round(shoulder_w / body_h, 4),
            relative_waist=round(waist_w / body_h, 4),
            silhouette_density=round(density, 4),
        )

        mask_pil = Image.fromarray(mask_binary * 255)
        return metrics, mask_pil

    @staticmethod
    def calculate_scale_factor(user_metrics: BodyMetrics, plate_metrics: BodyMetrics) -> float:
        """
        Формула ТЗ:
        Scale Factor = (Ширина плеч и талии пользователя) / (Ширина плеч и талии модели на плейте)
        С учетом нормировки на рост (инвариантно к масштабу кадра).
        """
        user_idx = user_metrics.composite_build_index
        plate_idx = plate_metrics.composite_build_index

        if plate_idx <= 0:
            return 1.0

        scale_factor = user_idx / plate_idx
        # Ограничиваем диапазон разумными значениями (от 0.75 до 1.40) для предотвращения артефактов
        return float(np.clip(scale_factor, 0.75, 1.40))

    def create_adaptive_mask(
        self,
        plate_mask: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """
        Формирует адаптивный силуэт тела под комплекцию пользователя:
        - Область торса, плеч и талии плавно деформируется на scale_factor.
        - Область головы и стоп остается зафиксированной.
        """
        mask_np = np.array(plate_mask.convert("L"))
        h, w = mask_np.shape
        binary = (mask_np > 128).astype(np.uint8)

        y_indices = np.where(binary > 0)[0]
        if len(y_indices) == 0:
            return plate_mask

        y_top, y_bottom = int(np.min(y_indices)), int(np.max(y_indices))
        body_h = max(y_bottom - y_top, 1)

        result_mask = np.zeros_like(mask_np)

        # Вычисляем деформацию по строкам
        for y in range(y_top, y_bottom):
            row = binary[y, :]
            x_coords = np.where(row > 0)[0]
            if len(x_coords) == 0:
                continue

            x_left = np.min(x_coords)
            x_right = np.max(x_coords)
            x_center = (x_left + x_right) / 2.0
            half_width = (x_right - x_left) / 2.0

            # Доля высоты от макушки
            rel_y = (y - y_top) / body_h

            # Весовой коэффициент адаптации:
            # Голова (0 - 15%): не меняется (weight = 0)
            # Плечи и торс (20% - 60%): максимальное масштабирование (weight = 1)
            # Ноги/стопы (80% - 100%): фиксация (weight = 0)
            if rel_y < 0.15:
                weight = 0.0
            elif rel_y < 0.25:
                weight = (rel_y - 0.15) / 0.10
            elif rel_y < 0.65:
                weight = 1.0
            elif rel_y < 0.85:
                weight = 1.0 - (rel_y - 0.65) / 0.20
            else:
                weight = 0.0

            # Эффективный масштаб строки
            effective_scale = 1.0 + (scale_factor - 1.0) * weight
            new_half_width = half_width * effective_scale

            new_x1 = max(int(np.round(x_center - new_half_width)), 0)
            new_x2 = min(int(np.round(x_center + new_half_width)), w - 1)

            result_mask[y, new_x1:new_x2 + 1] = 255

        # Сглаживание краев деформированной маски
        result_mask = cv2.GaussianBlur(result_mask, (15, 15), 0)
        result_mask = (result_mask > 100).astype(np.uint8) * 255

        return Image.fromarray(result_mask)
