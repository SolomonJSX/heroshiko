import argparse
import sys
from pathlib import Path

# Корень проекта и src в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from PIL import Image
from heroshiko.core.schemas import GenerateRequestParams
from heroshiko.services.generation_service import HeroshikoService


def main():
    parser = argparse.ArgumentParser(
        description="Heroshiko AI Photo Studio CLI — Realistic Vision + FaceID + INSwapper"
    )
    parser.add_argument(
        "--selfie",
        type=str,
        required=True,
        help="Путь к фото лица (селфи) пользователя",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="dubai_luxury",
        help="ID темы/пресета: dubai_luxury, paris_cafe, kpop_idol, kpop_group (default: dubai_luxury)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.jpg",
        help="Путь для сохранения итогового изображения (default: output.jpg)",
    )
    parser.add_argument(
        "--hair_mode",
        type=str,
        default="keep",
        choices=["keep", "change", "preset"],
        help="Режим прически: keep (сохранить мою), change (сменить), preset (стиль темы)",
    )
    parser.add_argument(
        "--hair_style",
        type=str,
        default=None,
        help="Желаемый стиль прически при hair_mode=change (каре, пучок, fade, taper и т.д.)",
    )
    parser.add_argument(
        "--hair_color",
        type=str,
        default=None,
        help="Желаемый цвет прически при hair_mode=change (брюнет, блонд, каштановый и т.д.)",
    )
    parser.add_argument(
        "--ethnicity",
        type=str,
        default="auto",
        choices=["auto", "central_asian", "east_asian", "european", "middle_eastern", "latino"],
        help="Этнотип внешности (default: auto)",
    )
    parser.add_argument(
        "--body_build",
        type=str,
        default="auto",
        choices=["auto", "slender", "athletic", "plus_size"],
        help="Комплекция тела (default: auto)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Количество шагов диффузии (default: 25)",
    )
    parser.add_argument(
        "--face_strength",
        type=float,
        default=0.40,
        help="Сила влияния лица IP-Adapter (default: 0.40, реком: 0.35-0.45)",
    )

    args = parser.parse_args()

    selfie_path = Path(args.selfie).resolve()
    if not selfie_path.exists():
        print(f"[Error] Файл селфи не найден: {selfie_path}")
        sys.exit(1)

    print("\n=======================================================")
    print(f"🎬 Heroshiko Studio CLI | Пресет: {args.preset}")
    print(f"👤 Селфи: {selfie_path.name}")
    print(f"🎨 Face Strength: {args.face_strength} | Steps: {args.steps}")
    print("=======================================================\n")

    input_img = Image.open(selfie_path).convert("RGB")
    service = HeroshikoService.get_instance()

    params = GenerateRequestParams(
        preset_id=args.preset,
        hair_mode=args.hair_mode,
        hair_style=args.hair_style,
        hair_color=args.hair_color,
        ethnicity=args.ethnicity,
        body_build=args.body_build,
        face_strength=args.face_strength,
        steps=args.steps,
    )

    final_img, face_preview, meta = service.generate_photoshoot(input_img, params)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(out_path, format="JPEG", quality=95)

    print(f"\n[Done] ✅ Результат успешно сохранен: {out_path}")
    print(f"Метаданные генерации: {meta}")


if __name__ == "__main__":
    main()
