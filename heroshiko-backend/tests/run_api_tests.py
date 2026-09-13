import io
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Добавляем папку src в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PIL import Image
from fastapi.testclient import TestClient
from heroshiko.api.server import app


def _create_test_image() -> bytes:
    img = Image.new("RGB", (200, 200), color=(180, 140, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_all_tests():
    print("\n--- [1/6] Запуск FastAPI TestClient Lifespan ---")
    with TestClient(app) as client:
        print("--- [2/6] Проверка /health и /api/v1/health ---")
        r = client.get("/health")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert data["status"] == "ok"
        print(f"[OK] Health OK: CUDA={data['cuda_available']}, GPU={data['device_name']}")

        print("--- [3/6] Проверка /api/v1/presets и постеров-обложек ---")
        r = client.get("/api/v1/presets")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 4
        preset_ids = [p["id"] for p in data["presets"]]
        print(f"[OK] Presets OK ({data['total']} presets): {preset_ids}")

        # Проверка cover_url для каждого пресета
        for p in data["presets"]:
            assert p["cover_url"] is not None, f"Cover URL missing for preset {p['id']}"
            assert f"/api/v1/presets/{p['id']}/cover" in p["cover_url"]

            # Проверка скачивания обложки
            cover_resp = client.get(p["cover_url"])
            assert cover_resp.status_code == 200, f"Failed to get cover for {p['id']}"
            assert cover_resp.headers["content-type"] in ["image/jpeg", "image/png"]
            assert len(cover_resp.content) > 1000, f"Cover image empty for {p['id']}"
            print(f"[OK] Cover image verified for: {p['id']} ({len(cover_resp.content)} bytes)")

        r = client.get("/api/v1/presets/dubai_luxury")
        assert r.status_code == 200
        assert r.json()["id"] == "dubai_luxury"
        assert r.json()["cover_url"] == "/api/v1/presets/dubai_luxury/cover"

        r = client.get("/api/v1/presets/invalid_preset_id")
        assert r.status_code == 404
        print("[OK] Preset 404 validation OK")

        print("--- [4/6] Проверка /api/v1/analyze (биометрический анализ) ---")
        # Пустой файл -> 400
        r = client.post("/api/v1/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert r.status_code == 400

        # Тестовая картинка без лица -> 200, face_detected=False
        test_img = _create_test_image()
        r = client.post("/api/v1/analyze", files={"file": ("test.jpg", test_img, "image/jpeg")})
        assert r.status_code == 200
        data = r.json()
        assert data["face_detected"] is False
        assert data["multiple_faces_detected"] is False
        assert data["total_faces_count"] == 0
        print("[OK] Analyze endpoint validation OK (face_detected=False, count=0)")

        print("--- [5/6] Проверка /api/v1/generate (постановка в очередь) ---")
        gen_data = {
            "preset_id": "dubai_luxury",
            "hair_mode": "keep",
            "steps": "20",
        }
        r = client.post("/api/v1/generate", files={"file": ("selfie.jpg", test_img, "image/jpeg")}, data=gen_data)
        assert r.status_code == 202
        task = r.json()
        task_id = task["task_id"]
        assert task["preset_id"] == "dubai_luxury"
        assert task["status"] in ["queued", "processing"]
        print(f"[OK] Enqueue task OK: task_id={task_id}, status={task['status']}")

        print("--- [6/6] Проверка /api/v1/tasks/{task_id} (статус задачи) ---")
        r = client.get(f"/api/v1/tasks/{task_id}")
        assert r.status_code == 200
        assert r.json()["task_id"] == task_id
        print(f"[OK] Task polling OK: {r.json()['status']}")

    print("\n>>> ВСЕ ТЕСТЫ БЭКЕНДА УСПЕШНО ПРОЙДЕНЫ! <<<\n")


if __name__ == "__main__":
    run_all_tests()
