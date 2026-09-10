from controlnet_aux import OpenposeDetector
from PIL import Image


class PoseExtractor:
    def __init__(self):
        self.detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

    def extract(self, image: Image.Image) -> Image.Image:
        # Возвращает карту скелета в RGB
        return self.detector(image, hand_and_face=True)
