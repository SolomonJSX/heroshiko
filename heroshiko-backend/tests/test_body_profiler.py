import pytest
import numpy as np
from PIL import Image

from src.cv.body_profiler import BodyProfiler, BodyMetrics


def test_body_scale_factor_identical_models():
    """
    Тест: когда комплекция пользователя совпадает с моделью плейта,
    Scale Factor строго равен 1.0.
    """
    metrics = BodyMetrics(
        shoulder_width=120.0,
        waist_width=90.0,
        hip_width=100.0,
        body_height=600.0,
        relative_shoulder=0.20,
        relative_waist=0.15,
        silhouette_density=0.40,
    )
    scale = BodyProfiler.calculate_scale_factor(metrics, metrics)
    assert np.isclose(scale, 1.0, atol=1e-4), f"Ожидался Scale Factor 1.0, получено: {scale}"


def test_body_scale_factor_edge_cases_skinny_vs_plussize():
    """
    Тест пограничных случаев: очень худой vs plus-size.
    Проверяет корректность расчета и безопасные границы (clamping 0.75 - 1.40).
    """
    # Эталонная модель на плейте (стандартное модельное телосложение)
    plate_model = BodyMetrics(
        shoulder_width=120.0,
        waist_width=84.0,
        hip_width=96.0,
        body_height=600.0,
        relative_shoulder=0.20,  # 120 / 600
        relative_waist=0.14,     # 84 / 600
        silhouette_density=0.38,
    )

    # 1. Пограничный случай: ОЧЕНЬ ХУДОЙ клиент (Skinny)
    skinny_user = BodyMetrics(
        shoulder_width=80.0,
        waist_width=50.0,
        hip_width=65.0,
        body_height=600.0,
        relative_shoulder=0.133,
        relative_waist=0.083,
        silhouette_density=0.25,
    )
    scale_skinny = BodyProfiler.calculate_scale_factor(skinny_user, plate_model)
    print(f"Skinny Scale Factor: {scale_skinny}")
    assert scale_skinny < 1.0, "Для худого пользователя масштаб должен быть меньше 1.0"
    assert scale_skinny >= 0.75, "Scale Factor не должен опускаться ниже безопасного лимита 0.75"

    # 2. Пограничный случай: КРУПНЫЙ клиент (Plus-size / Бодибилдер)
    plus_size_user = BodyMetrics(
        shoulder_width=190.0,
        waist_width=160.0,
        hip_width=170.0,
        body_height=600.0,
        relative_shoulder=0.316,
        relative_waist=0.266,
        silhouette_density=0.62,
    )
    scale_plus = BodyProfiler.calculate_scale_factor(plus_size_user, plate_model)
    print(f"Plus-Size Scale Factor: {scale_plus}")
    assert scale_plus > 1.0, "Для крупного пользователя масштаб должен быть больше 1.0"
    assert scale_plus <= 1.40, "Scale Factor не должен превышать максимальный безопасный лимит 1.40"


def test_adaptive_mask_geometry():
    """
    Тест геометрии адаптивной маски:
    При увеличении Scale Factor ширина торса маски расширяется,
    а голова и стопы остаются зафиксированными на своих координатах.
    """
    w, h = 400, 800
    mask_arr = np.zeros((h, w), dtype=np.uint8)
    
    # Прямоугольный силуэт (y: 100..700, x: 160..240)
    # Голова: 100..200 (x: 180..220)
    mask_arr[100:200, 180:220] = 255
    # Торс: 200..500 (x: 150..250, ширина = 100px)
    mask_arr[200:500, 150:250] = 255
    # Ноги: 500..700 (x: 170..230)
    mask_arr[500:700, 170:230] = 255

    orig_mask = Image.fromarray(mask_arr)
    profiler = BodyProfiler(seg_service=None)

    # Применяем увеличение для plus-size (+20%)
    adaptive_mask = profiler.create_adaptive_mask(orig_mask, scale_factor=1.25)
    ad_arr = np.array(adaptive_mask)

    # 1. Замеряем ширину торса (строка y = 350)
    orig_torso_w = np.sum(mask_arr[350, :] > 128)
    new_torso_w = np.sum(ad_arr[350, :] > 128)
    assert new_torso_w > orig_torso_w, f"Ширина торса должна увеличиться (было {orig_torso_w}, стало {new_torso_w})"

    # 2. Замеряем ширину головы (строка y = 130)
    # Голова не должна раздуваться пропорционально торсу
    orig_head_w = np.sum(mask_arr[130, :] > 128)
    new_head_w = np.sum(ad_arr[130, :] > 128)
    assert abs(new_head_w - orig_head_w) <= 4, "Голова обязана оставаться зафиксированной по ширине"
