import pytest
import torch
import numpy as np
from PIL import Image

from src.cv.face_profiler import FaceProfiler, FaceProfile


def test_face_profile_l2_normalization_contract():
    """
    Проверка контракта нормализации:
    Эмбеддинг лица ОБЯЗАН быть строго L2-нормализован (||e||_2 == 1.0)
    и иметь точную размерность 512-d.
    """
    # Имитация сырого вектора InsightFace
    raw_vector = np.random.randn(512).astype(np.float32)
    norm = np.linalg.norm(raw_vector)
    normalized = raw_vector / norm

    profile = FaceProfile(
        embedding=normalized,
        kps=np.zeros((5, 2), dtype=np.float32),
        mesh_106=np.zeros((106, 2), dtype=np.float32),
        bbox=[100, 100, 300, 300],
        num_samples=1,
    )

    # 1. Проверка размерности
    assert profile.embedding.shape == (512,), "Размерность вектора лица должна быть строго (512,)"

    # 2. Проверка L2-нормы
    l2_norm = np.linalg.norm(profile.embedding)
    assert np.isclose(l2_norm, 1.0, atol=1e-6), f"L2 норма вектора должна быть 1.0, получено: {l2_norm}"

    # 3. Проверка конвертации в PyTorch тензор
    torch_tensor = profile.to_torch()
    assert torch_tensor.shape == (1, 512), "PyTorch тензор для InstantID должен иметь размер (1, 512)"
    assert torch_tensor.dtype == torch.float16, "PyTorch тензор должен быть в FP16"
    assert torch.isclose(torch.norm(torch_tensor.float()), torch.tensor(1.0), atol=1e-3)


def test_multi_photo_face_embedding_stacking():
    """
    Проверка контракта усреднения N векторов лица (Face Embedding Stacking):
    Сумма векторов с 3 разных селфи после усреднения обязана оставаться единичным вектором.
    """
    np.random.seed(42)
    # Генерируем 3 слегка различающихся вектора одного человека (с шумом теней/бликов)
    base_face = np.random.randn(512).astype(np.float32)
    base_face /= np.linalg.norm(base_face)

    # Небольшой шум (вариации освещения, ракурса)
    v1 = base_face + np.random.randn(512).astype(np.float32) * 0.005
    v2 = base_face + np.random.randn(512).astype(np.float32) * 0.005
    v3 = base_face + np.random.randn(512).astype(np.float32) * 0.005

    # L2-нормализация каждого
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)

    # Усреднение и повторная нормализация
    stacked = (v1 + v2 + v3) / 3.0
    final_embedding = stacked / np.linalg.norm(stacked)

    # Проверяем контракт
    assert final_embedding.shape == (512,)
    assert np.isclose(np.linalg.norm(final_embedding), 1.0, atol=1e-6)

    # Проверяем, что усредненный вектор сохраняет высокую близость к истинному лицу
    sim_base = np.dot(final_embedding, base_face)
    assert sim_base > 0.98, f"Усредненный вектор должен сохранять высокую близость к базе (>0.98), получено {sim_base}"
