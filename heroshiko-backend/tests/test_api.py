import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from heroshiko.api.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _create_test_image() -> bytes:
    """Создает тестовое изображение 200x200 в памяти."""
    img = Image.new("RGB", (200, 200), color=(180, 140, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "cuda_available" in data
    assert "models_loaded" in data


def test_api_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_presets_list(client):
    response = client.get("/api/v1/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert data["total"] >= 4
    preset_ids = [p["id"] for p in data["presets"]]
    assert "dubai_luxury" in preset_ids
    assert "paris_cafe" in preset_ids
    assert "kpop_idol" in preset_ids
    assert "kpop_group" in preset_ids


def test_preset_detail(client):
    response = client.get("/api/v1/presets/dubai_luxury")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "dubai_luxury"
    assert "Дубай" in data["title"]


def test_preset_not_found(client):
    response = client.get("/api/v1/presets/non_existent_preset")
    assert response.status_code == 404


def test_analyze_empty_file(client):
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400


def test_analyze_invalid_mime(client):
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400


def test_analyze_face_no_face_detected(client):
    # Картинка без лица
    img_bytes = _create_test_image()
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["face_detected"] is False


def test_generate_async_enqueue(client):
    img_bytes = _create_test_image()
    files = {"file": ("selfie.jpg", img_bytes, "image/jpeg")}
    data = {
        "preset_id": "dubai_luxury",
        "hair_mode": "keep",
        "steps": "20",
    }
    response = client.post("/api/v1/generate", files=files, data=data)
    assert response.status_code == 202
    res = response.json()
    assert "task_id" in res
    assert res["preset_id"] == "dubai_luxury"
    assert res["status"] in ["queued", "processing"]

    # Проверка получения статуса задачи
    task_id = res["task_id"]
    status_resp = client.get(f"/api/v1/tasks/{task_id}")
    assert status_resp.status_code == 200
    task_data = status_resp.json()
    assert task_data["task_id"] == task_id
