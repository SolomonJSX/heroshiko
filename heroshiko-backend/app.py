from pathlib import Path

import gradio as gr
from PIL import Image

from heroshiko.core.catalog import PresetManager
from heroshiko.cv.face import FaceIdentityExtractor
from heroshiko.cv.parser import ClothingParser
from heroshiko.cv.pose import PoseExtractor
from heroshiko.pipelines.postprocess import PostProcessor
from heroshiko.pipelines.sdxl_engine import HeroshikoEngine

# --- Ленивая глобальная загрузка моделей ---
# Инициализируем один раз, чтобы не перезагружать веса при каждом клике
print("Инициализация CV-моделей heroshiko...")
pose_extractor = PoseExtractor()
clothing_parser = ClothingParser()
face_extractor = FaceIdentityExtractor()
postprocessor = PostProcessor()

print("Загрузка каталога пресетов...")
preset_manager = PresetManager(Path("configs/presets.yaml"))
available_presets = {p.title: p.id for p in preset_manager.list_presets()}

# SDXL движок инициализируем по требованию (лениво) при первой генерации
engine: HeroshikoEngine | None = None


def get_engine() -> HeroshikoEngine:
    global engine
    if engine is None:
        print("Инициализация SDXL Engine (RTX 3060)...")
        engine = HeroshikoEngine()
    return engine


def process_image(
    input_image: Image.Image | None,
    preset_title: str,
    face_strength: float,
    control_strength: float,
    steps: int,
    apply_lighting: bool,
    apply_face_restore: bool,
) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None, Image.Image | None, str]:
    if input_image is None:
        return None, None, None, None, "Ошибка: загрузите фотографию!"

    preset_id = available_presets.get(preset_title)
    if not preset_id:
        return None, None, None, None, "Ошибка: пресет не найден."

    preset = preset_manager.get_preset(preset_id)
    sdxl = get_engine()

    rgb_input = input_image.convert("RGB")

    # 1. Извлечение позы
    pose_preview = pose_extractor.extract(rgb_input)

    # 2. Сегментация наряда и фона
    parsing_res = clothing_parser.get_outfit_and_background_mask(rgb_input)
    mask_preview = parsing_res.inpaint_mask

    # 3. Извлечение идентичности лица
    face_info = face_extractor.extract(rgb_input)
    aligned_face_preview = face_info.aligned_face_chip if face_info else None

    # 4. Генерация
    generated = sdxl.generate(
        image=rgb_input,
        mask_image=parsing_res.inpaint_mask,
        pose_image=pose_preview,
        face_info=face_info,
        prompt=preset.prompt,
        negative_prompt=preset.negative_prompt,
        num_inference_steps=int(steps),
        controlnet_conditioning_scale=control_strength,
        face_strength=face_strength,
        strength=preset.denoising_strength,
    )

    # 5. Постобработка
    final_output = postprocessor.process(
        generated_img=generated,
        face_mask=parsing_res.keep_mask,
        apply_harmonization=apply_lighting,
        apply_face_restoration=apply_face_restore,
    )

    status_msg = f"Успешно сгенерировано! Тема: {preset.title}"
    if face_info is None:
        status_msg += " (Внимание: лицо на фото не обнаружено, IP-Adapter пропущен)"

    return final_output, pose_preview, mask_preview, aligned_face_preview, status_msg


# --- Сборка интерфейса Gradio ---
with gr.Blocks(title="Heroshiko Studio") as demo:
    gr.Markdown("# 🎭 Heroshiko AI Studio")
    gr.Markdown("Локальная трансформация образов и замена окружения на RTX 3060.")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Исходное фото")

            preset_dropdown = gr.Dropdown(
                choices=list(available_presets.keys()),
                value=list(available_presets.keys())[0] if available_presets else None,
                label="Выберите тему",
            )

            with gr.Accordion("Тонкие настройки (GPU)", open=False):
                slider_steps = gr.Slider(minimum=20, maximum=40, value=28, step=1, label="Шаги семплера (Steps)")
                slider_face = gr.Slider(minimum=0.5, maximum=1.0, value=0.85, step=0.05, label="Сила сохранения лица (FaceID)")
                slider_control = gr.Slider(minimum=0.4, maximum=1.0, value=0.75, step=0.05, label="Фиксация позы (OpenPose)")
                chk_lighting = gr.Checkbox(value=True, label="Гармонизация освещения (Reinhard Lab)")
                chk_restore = gr.Checkbox(value=True, label="Реставрация лица (CodeFormer)")

            btn_run = gr.Button("Сгенерировать образ", variant="primary")
            status_text = gr.Textbox(label="Статус выполнения", interactive=False)

        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="Финальный результат")

            with gr.Accordion("Промежуточные слои CV", open=False):
                with gr.Row():
                    out_pose = gr.Image(type="pil", label="Скелет (OpenPose)")
                    out_mask = gr.Image(type="pil", label="Маска замены")
                    out_face = gr.Image(type="pil", label="Выровненное лицо")

    btn_run.click(
        fn=process_image,
        inputs=[
            input_img,
            preset_dropdown,
            slider_face,
            slider_control,
            slider_steps,
            chk_lighting,
            chk_restore,
        ],
        outputs=[output_img, out_pose, out_mask, out_face, status_text],
    )

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
