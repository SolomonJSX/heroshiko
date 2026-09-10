from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dummy_rgb_image() -> Image.Image:
    """Генерирует тестовое RGB изображение 512x512 с простыми фигурами."""
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    # Имитируем условный фон, тело и лицо
    arr[:, :] = [120, 150, 180]  # фон
    arr[200:480, 150:360] = [30, 40, 50]  # тело / одежда
    arr[80:200, 200:310] = [220, 180, 150]  # лицо
    return Image.fromarray(arr)


@pytest.fixture
def dummy_face_mask() -> Image.Image:
    """Бинарная маска лица 512x512."""
    arr = np.zeros((512, 512), dtype=np.uint8)
    arr[80:200, 200:310] = 255
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def sample_presets_yaml(tmp_path: Path) -> Path:
    """Временный валидный YAML-конфиг пресетов."""
    content = """
presets:
  test_theme:
    title: "Тестовая тема"
    prompt: "a photorealistic portrait in test location, 8k"
    negative_prompt: "blurry, low quality"
    control_strength: 0.75
    denoising_strength: 0.65
"""
    config_file = tmp_path / "presets.yaml"
    config_file.write_text(content.strip(), encoding="utf-8")
    return config_file
