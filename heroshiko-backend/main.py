from pathlib import Path

from PIL import Image

from heroshiko.core.catalog import PresetManager
from heroshiko.cv.face import FaceIdentityExtractor
from heroshiko.cv.parser import ClothingParser
from heroshiko.cv.pose import PoseExtractor
from heroshiko.pipelines.postprocess import PostProcessor
from heroshiko.pipelines.sdxl_engine import HeroshikoEngine


def run_heroshiko(input_path: str, preset_key: str, output_path: str):
    print("Инициализация конвейера heroshiko...")
    pose_extractor = PoseExtractor()
    clothing_parser = ClothingParser()
    face_extractor = FaceIdentityExtractor()
    engine = HeroshikoEngine()
    postprocessor = PostProcessor()

    presets = PresetManager(Path("configs/presets.yaml"))
    preset = presets.get_preset(preset_key)
    if not preset:
        raise ValueError(f"Пресет '{preset_key}' не найден в каталоге.")

    input_img = Image.open(input_path).convert("RGB")

    print("[1/4] Извлечение позы (OpenPose)...")
    pose_img = pose_extractor.extract(input_img)

    print("[2/4] Сегментация наряда и фона (SegFormer)...")
    parsing_res = clothing_parser.get_outfit_and_background_mask(input_img)

    print("[3/4] Извлечение вектора идентичности лица (InsightFace)...")
    face_info = face_extractor.extract(input_img)

    print(f"[4/4] Генерация темы '{preset.title}' на RTX 3060...")
    generated_img = engine.generate(
        image=input_img,
        mask_image=parsing_res.inpaint_mask,
        pose_image=pose_img,
        face_info=face_info,
        prompt=preset.prompt,
        negative_prompt=preset.negative_prompt,
        controlnet_conditioning_scale=preset.control_strength,
        strength=preset.denoising_strength,
    )

    print("[Финал] Постобработка: гармонизация цвета и полировка лица...")
    final_result = postprocessor.process(
        generated_img=generated_img,
        face_mask=parsing_res.keep_mask,
        apply_harmonization=True,
        apply_face_restoration=True,
    )

    final_result.save(output_path, quality=95)
    print(f"Готово! Изображение сохранено: {output_path}")


if __name__ == "__main__":
    run_heroshiko("test.jpg", "dubai_luxury", "output_dubai.jpg")
