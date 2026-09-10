import pytest

from heroshiko.pipelines.sdxl_engine import calculate_sdxl_dimensions


@pytest.mark.parametrize(
    "orig_w, orig_h, expected_w, expected_h",
    [
        (1000, 1000, 1024, 1024),  # Square 1:1
        (1080, 1920, 768, 1344),   # Vertical mobile photo 9:16
        (1920, 1080, 1344, 768),   # Landscape 16:9
        (1200, 1600, 896, 1152),   # Portrait 3:4
        (1600, 1200, 1152, 896),   # Landscape 4:3
    ],
)
def test_calculate_sdxl_dimensions_preserves_aspect_ratio(orig_w, orig_h, expected_w, expected_h):
    w, h = calculate_sdxl_dimensions(orig_w, orig_h)
    assert (w, h) == (expected_w, expected_h)
    assert w % 64 == 0
    assert h % 64 == 0
