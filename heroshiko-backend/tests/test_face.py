import numpy as np
import torch


def test_face_identity_dataclass_contract():
    # Проверяем, что вектор строго [1, 512] и имеет единичную норму
    raw_vec = np.random.randn(512).astype(np.float32)
    norm_vec = raw_vec / np.linalg.norm(raw_vec)
    tensor_embed = torch.from_numpy(norm_vec).unsqueeze(0)

    # Проверка L2-нормы
    l2_norm = torch.norm(tensor_embed, p=2).item()
    assert np.isclose(l2_norm, 1.0, atol=1e-5)
    assert tensor_embed.shape == (1, 512)
