import numpy as np
from PIL import Image

from heroshiko.pipelines.postprocess import PostProcessor


def test_reinhard_color_transfer_shape_and_range():
    # Создаем 2 изображения с разной палитрой
    source = np.full((100, 100, 3), fill_value=50, dtype=np.uint8)  # темное
    target = np.full((100, 100, 3), fill_value=200, dtype=np.uint8)  # светлое

    result = PostProcessor.match_color_reinhard(source, target)

    assert result.shape == source.shape
    assert result.dtype == np.uint8
    # Значения должны сместиться в сторону целевого (стать светлее)
    assert np.mean(result) > np.mean(source)


def test_harmonize_lighting_dimensions(dummy_rgb_image: Image.Image, dummy_face_mask: Image.Image):
    processor = PostProcessor.__new__(PostProcessor)
    processor.device = "cpu"

    harmonized = processor.harmonize_lighting(
        generated_img=dummy_rgb_image,
        face_mask=dummy_face_mask,
        blend_factor=0.3,
    )

    assert isinstance(harmonized, Image.Image)
    assert harmonized.size == dummy_rgb_image.size
    assert harmonized.mode == "RGB"
