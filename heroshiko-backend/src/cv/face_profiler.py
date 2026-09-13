import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from insightface.app import FaceAnalysis


@dataclass
class FaceProfile:
    """
    Математический слепок лица пользователя.
    """
    embedding: np.ndarray             # Усредненный L2-нормализованный вектор 512-d
    kps: np.ndarray                   # 5 ключевых точек (глаза, нос, углы губ)
    mesh_106: Optional[np.ndarray]    # 106 точек сетки лица для ориентации в пространстве
    bbox: List[float]                 # Ограничивающая рамка [x1, y1, x2, y2]
    num_samples: int                  # Количество обработанных селфи
    gender: Optional[str] = None      # "male" или "female"
    age: Optional[int] = None         # Оценочный возраст

    def to_torch(self, device: str = "cpu", dtype=torch.float16) -> torch.Tensor:
        """Возвращает эмбеддинг в виде PyTorch тензора для InstantID."""
        return torch.from_numpy(self.embedding).unsqueeze(0).to(device=device, dtype=dtype)


class FaceProfiler:
    """
    Модуль биометрии лица (3.1).
    Использует AntelopeV2 и Face Embedding Stacking для исключения бликов и теней.
    """
    def __init__(self, root_dir: str = "."):
        print("[FaceProfiler] Инициализация AntelopeV2 FaceAnalysis...")
        self.app = FaceAnalysis(
            name="antelopev2",
            root=root_dir,
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _extract_face(self, image: Image.Image):
        """Извлекает главное (наиболее крупное) лицо из одного изображения."""
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = self.app.get(cv_img)

        if len(faces) == 0:
            return None

        # Сортируем лица по площади рамки (выбираем ближайшее к камере лицо)
        faces = sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True,
        )
        return faces[0]

    def create_profile(self, images: List[Image.Image]) -> FaceProfile:
        """
        Создает математический слепок внешности на основе 1-N селфи пользователя.
        Выполняет L2-нормализацию и усреднение (Embedding Stacking).
        """
        if not images:
            raise ValueError("[FaceProfiler] Список фотографий пуст.")

        normalized_embeddings = []
        best_face = None
        best_face_area = -1

        for idx, img in enumerate(images):
            face = self._extract_face(img)
            if face is None:
                print(f"[FaceProfiler] Предупреждение: лицо не найдено на фото #{idx + 1}, пропускаем.")
                continue

            # L2-нормализация вектора конкретного фото
            raw_emb = face.embedding
            norm = np.linalg.norm(raw_emb)
            if norm > 0:
                l2_emb = raw_emb / norm
                normalized_embeddings.append(l2_emb)

            area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            if area > best_face_area:
                best_face_area = area
                best_face = face

        if not normalized_embeddings or best_face is None:
            raise ValueError("[FaceProfiler] Ни на одной из предоставленных фотографий лицо не обнаружено.")

        # Усреднение векторов нескольких селфи (Face Embedding Stacking)
        mean_embedding = np.mean(normalized_embeddings, axis=0)
        # Финальная L2-нормализация
        final_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        # Извлечение 106-точечной сетки лица (landmark_2d_106)
        mesh_106 = getattr(best_face, "landmark_2d_106", None)
        if mesh_106 is None:
            # Если 106 точек не найдены, берем стандартные ключевые точки
            mesh_106 = best_face.kps

        gender_str = "male" if getattr(best_face, "gender", 1) == 1 else "female"
        age_val = int(getattr(best_face, "age", 25))

        print(f"[FaceProfiler] Успешно усреднено селфи: {len(normalized_embeddings)} шт. Возраст: ~{age_val}, Пол: {gender_str}")

        return FaceProfile(
            embedding=final_embedding.astype(np.float32),
            kps=best_face.kps,
            mesh_106=mesh_106,
            bbox=best_face.bbox.tolist(),
            num_samples=len(normalized_embeddings),
            gender=gender_str,
            age=age_val,
        )
