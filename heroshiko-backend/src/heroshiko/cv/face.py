from dataclasses import dataclass

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image


@dataclass
class FaceIdentity:
    """Контейнер данных идентичности лица для IP-Adapter-FaceID."""

    # Нормализованный вектор идентичности (1, 512)
    embedding: torch.Tensor
    # Выровненный кроп лица в PIL для visual-проектора IP-Adapter PlusV2
    aligned_face_chip: Image.Image
    # Ограничивающая рамка (bbox) лица в исходном изображении: [x1, y1, x2, y2]
    bbox: np.ndarray
    # Ключевые точки лица (5 landmarks)
    landmarks: np.ndarray
    # Исходный объект лица InsightFace (для Face Swapper)
    raw_face: object = None
    # Автоматически определенный пол ("man" или "woman")
    gender: str = "man"
    # Автоматически определенный возраст
    age: int = 28
    # Общее количество обнаруженных лиц на фото
    total_faces_count: int = 1


class FaceIdentityExtractor:
    def __init__(
        self,
        name: str = "buffalo_l",
        root_dir: str = "./weights/insightface",
        providers: list | None = None,
        det_size: tuple[int, int] = (640, 640),
    ):
        """
        Инициализирует FaceAnalysis (InsightFace).
        Провайдер по умолчанию: CUDAExecutionProvider для ускорения на RTX 3060.
        """
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.app = FaceAnalysis(name=name, root=root_dir, providers=providers)
        # Включаем модуль детекции (ctx_id=0 для GPU)
        self.app.prepare(ctx_id=0, det_size=det_size)

    def extract(self, image: Image.Image) -> FaceIdentity | None:
        """
        Принимает PIL-изображение, детектирует главное лицо и возвращает FaceIdentity.
        Если на фото несколько людей, выбирается лицо с наибольшей площадью bbox.
        """
        # InsightFace ожидает BGR формат (OpenCV)
        rgb_np = np.array(image.convert("RGB"))
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        faces = self.app.get(bgr_np)
        if not faces:
            return None

        total_faces = len(faces)

        # Выбираем самое крупное лицо на фото (по площади рамки)
        main_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        # 1. Извлекаем и нормализуем эмбеддинг идентичности
        embed_np = main_face.embedding
        # Нормализация L2 обязательна для косинусного сходства в IP-Adapter
        embed_norm = embed_np / np.linalg.norm(embed_np)
        embed_tensor = torch.from_numpy(embed_norm).unsqueeze(0).float()

        # 2. Получаем стандартизированный выровненный кроп лица (112x112 -> масштабируем до 512x512)
        # face_align.norm_crop использует стандартное аффинное выравнивание по 5 точкам
        aligned_bgr = face_align.norm_crop(bgr_np, landmark=main_face.kps, image_size=512)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        aligned_pil = Image.fromarray(aligned_rgb)

        # 3. Извлечение анатомических атрибутов (пол и возраст)
        sex = getattr(main_face, "sex", None)
        gender_num = getattr(main_face, "gender", 1)
        gender_str = "man" if sex == "M" or gender_num == 1 else "woman"
        age_num = int(getattr(main_face, "age", 28))

        return FaceIdentity(
            embedding=embed_tensor,
            aligned_face_chip=aligned_pil,
            bbox=main_face.bbox.astype(int),
            landmarks=main_face.kps,
            raw_face=main_face,
            gender=gender_str,
            age=age_num,
            total_faces_count=total_faces,
        )
