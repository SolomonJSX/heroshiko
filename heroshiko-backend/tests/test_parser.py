from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from heroshiko.cv.parser import ClothingParser, HumanParsingResult


@patch("heroshiko.cv.parser.AutoImageProcessor.from_pretrained")
@patch("heroshiko.cv.parser.AutoModelForSemanticSegmentation.from_pretrained")
def test_clothing_parser_mask_generation(
    mock_model_cls, mock_processor_cls, dummy_rgb_image: Image.Image
):
    # Мокаем тяжелые веса HF, чтобы тест выполнялся за миллисекунды
    mock_processor = MagicMock()
    mock_processor_cls.return_value = mock_processor

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model_cls.return_value = mock_model

    parser = ClothingParser(device="cpu")

    # Имитируем вывод меток классов (512x512): центр - лицо (12), низ - одежда (5), остальное - фон (0)
    fake_labels = np.zeros((512, 512), dtype=np.uint8)
    fake_labels[100:200, 200:300] = 12  # Face
    fake_labels[200:400, 150:350] = 5  # Upper-clothes
    parser.parse = MagicMock(return_value=fake_labels)

    result = parser.get_outfit_and_background_mask(
        dummy_rgb_image,
        dilate_kernel_size=5,
        blur_radius=3,
    )

    assert isinstance(result, HumanParsingResult)
    assert result.inpaint_mask.size == dummy_rgb_image.size
    assert result.keep_mask.size == dummy_rgb_image.size

    # Лицо в keep_mask должно быть белым (255)
    keep_arr = np.array(result.keep_mask)
    assert keep_arr[150, 250] == 255

    # Одежда и фон в inpaint_mask должны быть под замену (>0)
    inpaint_arr = np.array(result.inpaint_mask)
    assert inpaint_arr[300, 250] > 0  # зона верхней одежды
    assert inpaint_arr[10, 10] > 0  # зона фона
