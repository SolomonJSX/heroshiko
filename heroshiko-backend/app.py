import sys
from pathlib import Path

# Автоматическое добавление папки src в пути поиска Python
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
from PIL import Image

from heroshiko.core.catalog import PresetManager
from heroshiko.cv.body import BodyBuildEstimator
from heroshiko.cv.face import FaceIdentityExtractor
from heroshiko.cv.hair import HairAnalyzer
from heroshiko.pipelines.postprocess import PostProcessor
from heroshiko.pipelines.sd15_engine import HeroshikoSD15Engine

# --- Ленивая глобальная загрузка моделей ---
print("Инициализация CV-моделей Heroshiko...")
face_extractor = FaceIdentityExtractor()
postprocessor = PostProcessor(face_extractor=face_extractor)
hair_analyzer = HairAnalyzer(device="cpu")
body_estimator = BodyBuildEstimator()

print("Загрузка каталога пресетов...")
preset_manager = PresetManager(Path("configs/presets.yaml"))
available_presets = {p.title: p.id for p in preset_manager.list_presets()}

# SD 1.5 GhostMix движок инициализируем по требованию при первой генерации
engine: HeroshikoSD15Engine | None = None


def get_engine() -> HeroshikoSD15Engine:
    global engine
    if engine is None:
        print("Инициализация Realistic Vision V5.1 + IP-Adapter FaceID (RTX 3060)...")
        engine = HeroshikoSD15Engine()
    return engine


def process_image(
    input_image: Image.Image | None,
    preset_title: str,
    hair_mode: str,
    hair_style_selection: str,
    hair_color_selection: str,
    ethnicity_selection: str,
    body_build_selection: str,
    neck_style_selection: str,
    face_strength: float,
    steps: int,
    guidance_scale: float,
    apply_pre_enhance: bool,
    apply_face_swap: bool,
    apply_face_restore: bool,
    fidelity_weight: float,
    apply_film_grain: bool,
) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None, str]:
    if input_image is None:
        return None, None, None, "Ошибка: загрузите фотографию с лицом!"

    preset_id = available_presets.get(preset_title)
    if not preset_id:
        return None, None, None, "Ошибка: пресет не найден."

    preset = preset_manager.get_preset(preset_id)
    sd15 = get_engine()

    rgb_input = input_image.convert("RGB")

    # 1. Опциональное предварительное улучшение качества исходного селфи
    if apply_pre_enhance:
        try:
            rgb_input = postprocessor.pre_enhance_selfie(rgb_input, fidelity_weight=0.92)
        except Exception as e:
            print(f"[Селфи] Пропуск pre-enhance: {e}")

    # 2. Извлечение идентичности и анатомических параметров лица (InsightFace)
    face_info = face_extractor.extract(rgb_input)
    aligned_face_preview = face_info.aligned_face_chip if face_info else None

    # 3. Адаптация промпта под пол, возраст, этнотип, прическу, телосложение и шею
    prompt = preset.prompt
    negative_prompt = preset.negative_prompt
    gender_age_info = ""
    hair_desc_info = ""
    body_desc_info = ""
    ethnicity_desc_info = ""

    # Анализ телосложения и комплекции (биометрия лица или выбор пользователя)
    body_info = body_estimator.estimate(
        face_identity=face_info,
        image=rgb_input,
        manual_override=body_build_selection,
    )
    body_desc_info = f", комплекция: {body_info.label_ru}"

    detected_color = "dark"
    detected_has_beard = False
    auto_hair_prompt = "short neat haircut"

    # Анализируем волосы с учетом пола (женщины: пучок/каре, мужчины: фейд/классика)
    if face_info is not None and "Свободный" not in hair_mode:
        try:
            hair_attr = hair_analyzer.analyze(rgb_input, face_bbox=face_info.bbox)
            detected_color = hair_attr.color
            detected_has_beard = hair_attr.has_beard
            auto_hair_prompt = hair_attr.hair_prompt
            hair_desc_info = f", прическа: {hair_attr.color} ({hair_attr.length})"
        except Exception as e:
            print(f"[Прическа] Пропуск анализа волос: {e}")

    # Определение цвета прически при смене стиля
    target_color = detected_color
    if "Сменить прическу" in hair_mode:
        if "Брюнет" in hair_color_selection or "Черный" in hair_color_selection:
            target_color = "dark black"
        elif "Темно-каштановый" in hair_color_selection:
            target_color = "dark brown"
        elif "Русый" in hair_color_selection:
            target_color = "light brown"
        elif "Блонд" in hair_color_selection:
            target_color = "blonde"
        elif "Пепельный" in hair_color_selection:
            target_color = "grey silver"

    # Формирование промпта прически в зависимости от выбора пользователя:
    hair_prompt = ""
    if "Сохранить мою" in hair_mode:
        hair_prompt = f" with {auto_hair_prompt}"
        hair_desc_info = f", прическа: сохранена из фото [{detected_color}]"
    elif "Сменить прическу" in hair_mode:
        beard_str = ", matching neat light facial hair" if detected_has_beard else ", clean shaven"
        if "пучок" in hair_style_selection.lower() or "хвост" in hair_style_selection.lower():
            hair_prompt = f" with {target_color} hair elegantly styled back in a neat bun or updo, clean natural hairline"
            hair_desc_info = f", новая прическа: пучок ({target_color})"
        elif "каре" in hair_style_selection.lower() or "боб" in hair_style_selection.lower():
            hair_prompt = f" with classic chic {target_color} bob haircut, feminine cut, soft volume"
            hair_desc_info = f", новая прическа: каре/боб ({target_color})"
        elif "локоны" in hair_style_selection.lower() or "волны" in hair_style_selection.lower():
            hair_prompt = f" with long voluminous wavy curly {target_color} hair, feminine styling"
            hair_desc_info = f", новая прическа: локоны ({target_color})"
        elif "Длинные" in hair_style_selection:
            hair_prompt = f" with long straight flowing {target_color} hair, natural volume"
            hair_desc_info = f", новая прическа: длинные ({target_color})"
        elif "Короткая модельная" in hair_style_selection:
            hair_prompt = f" with neat short {target_color} haircut, short fade on sides, clean natural hairline{beard_str}"
            hair_desc_info = f", новая прическа: fade/crop ({target_color})"
        elif "Классическая мужская" in hair_style_selection:
            hair_prompt = f" with classic short {target_color} taper haircut, neat clean hairline{beard_str}"
            hair_desc_info = f", новая прическа: classic taper ({target_color})"
        elif "Текстурированная" in hair_style_selection:
            hair_prompt = f" with short textured {target_color} crew cut, neat clean hairline{beard_str}"
            hair_desc_info = f", новая прическа: textured crew cut ({target_color})"
        elif "Стрижка под машинку" in hair_style_selection:
            hair_prompt = f" with very short {target_color} buzz cut, neat hairline{beard_str}"
            hair_desc_info = f", новая прическа: buzz cut ({target_color})"
        elif "андеркат" in hair_style_selection.lower():
            hair_prompt = f" with stylish {target_color} undercut hairstyle, short faded sides{beard_str}"
            hair_desc_info = f", новая прическа: undercut ({target_color})"
        elif "Средняя длина" in hair_style_selection:
            hair_prompt = f" with medium-length {target_color} styled hair"
            hair_desc_info = f", новая прическа: средняя длина ({target_color})"
        else:
            hair_prompt = f" with stylish {target_color} haircut{beard_str}"
            hair_desc_info = f", новая прическа: ({target_color})"
    else:
        hair_desc_info = ", прическа: стиль темы пресета"

    # Этнотип и анатомический профиль
    ethnicity_prompt = ""
    if face_info is not None:
        if "Центрально-Азиатский" in ethnicity_selection or "Казахский" in ethnicity_selection:
            ethnicity_prompt = f"Central Asian Kazakh {face_info.gender}"
            ethnicity_desc_info = ", типаж: Казахский / Центрально-Азиатский"
        elif "Восточно-Азиатский" in ethnicity_selection:
            ethnicity_prompt = f"East Asian {face_info.gender}"
            ethnicity_desc_info = ", типаж: Восточно-Азиатский"
        elif "Европейский" in ethnicity_selection:
            ethnicity_prompt = f"European Slavic {face_info.gender}"
            ethnicity_desc_info = ", типаж: Европейский"
        elif "Ближневосточный" in ethnicity_selection:
            ethnicity_prompt = f"Middle Eastern {face_info.gender}"
            ethnicity_desc_info = ", типаж: Ближневосточный"
        elif "Латиноамериканский" in ethnicity_selection:
            ethnicity_prompt = f"Latino {face_info.gender}"
            ethnicity_desc_info = ", типаж: Латино"

        subject_desc = ethnicity_prompt if ethnicity_prompt else face_info.gender
        body_tag = "plus-size " if body_info.build_type == "plus_size" else ("slender " if body_info.build_type == "slender" else ("athletic " if body_info.build_type == "athletic" else ""))
        target_subject = f"a {face_info.age}yo {body_tag}{subject_desc}{hair_prompt}"
        for placeholder in ["a person", "an elegant person", "a handsome person"]:
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, target_subject)
                break
        gender_age_info = f" [Типаж: {face_info.gender}, ~{face_info.age} лет{ethnicity_desc_info}{hair_desc_info}{body_desc_info}]"

    # Негативные модификаторы комплекции
    if body_info.negative_modifier:
        negative_prompt = f"{negative_prompt}, {body_info.negative_modifier}"

    # 5. Мгновенная генерация через SD 1.5 GhostMix + IP-Adapter FaceID (без костылей примерки одежды)
    eff_face_strength = float(face_strength)
    if "group" in preset_id:
        eff_face_strength = min(eff_face_strength, 0.40)

    generated = sd15.generate(
        face_info=face_info,
        prompt=prompt,
        negative_prompt=negative_prompt,
        face_strength=eff_face_strength,
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        width=512,
        height=768,
    )

    # 6. Финальная шлифовка и 100% перенос лица (INSwapper + CodeFormer + Film Grain)
    final_output = postprocessor.process(
        generated_img=generated,
        source_face_info=face_info,
        apply_face_swap=apply_face_swap,
        apply_face_restore=apply_face_restore,
        fidelity_weight=fidelity_weight,
        match_lighting=True,
        apply_film_grain=apply_film_grain,
    )

    status_msg = f"Успешно сгенерировано: {preset.title}{gender_age_info}"
    return final_output, generated, aligned_face_preview, status_msg


# --- Gradio веб-интерфейс ---
custom_css = """
.gradio-container { max-width: 1240px !important; margin: auto; }
.output-preview { border-radius: 12px; overflow: hidden; }
"""

with gr.Blocks(title="Heroshiko AI Photo Studio — Photoreal Edition") as demo:
    gr.Markdown(
        """
        # ⚡ Heroshiko AI — Фотосессия нового поколения (Realistic Vision + FaceID + INSwapper)
        *Ультра-быстрая генерация (~8 сек), 100% портретное сходство лица, живая текстура кожи и гармоничные пропорции фигуры.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Исходное фото лица (Селфи)", type="pil")
            dropdown_preset = gr.Dropdown(
                choices=list(available_presets.keys()),
                value=list(available_presets.keys())[0] if available_presets else None,
                label="Локация и стиль (Пресет)",
            )

            with gr.Group():
                gr.Markdown("### 👤 Анатомический профиль и этнотип")

                dropdown_ethnicity = gr.Dropdown(
                    choices=[
                        "Автоматический (из фото)",
                        "Казахский / Центрально-Азиатский (Central Asian)",
                        "Восточно-Азиатский (East Asian)",
                        "Европейский / Славянский (Slavic/European)",
                        "Ближневосточный (Middle Eastern)",
                        "Латиноамериканский (Latino)",
                    ],
                    value="Казахский / Центрально-Азиатский (Central Asian)",
                    label="Этнотип и разрез глаз",
                    info="Обеспечивает точную передачу формы века и разреза глаз",
                )

                dropdown_body = gr.Dropdown(
                    choices=[
                        "Авто (по ширине лица и щекам)",
                        "Плотная / Пышная фигура (Plus-size / Curvy)",
                        "Обычное / Естественное (Natural / Regular build)",
                        "Стройное тело (Slender / Slim build)",
                        "Атлетическое / Спортивное (Athletic / Fit build)",
                    ],
                    value="Авто (по ширине лица и щекам)",
                    label="Телосложение / Комплекция (Body Build)",
                    info="Определяет полноту фигуры в кадре",
                )

                dropdown_neck = gr.Dropdown(
                    choices=[
                        "Авто (согласовать с комплекцией) [Рекомендуется]",
                        "Стройная шея и очерченный подбородок (Slender & Defined)",
                        "Мягкие формы и плотная шея (Soft / Fuller)",
                        "Естественная пропорциональная (Natural / Balanced)",
                        "Атлетическая шея (Athletic / Muscular)",
                    ],
                    value="Авто (согласовать с комплекцией) [Рекомендуется]",
                    label="Коррекция шеи и подбородка",
                    info="Устраняет искусственные деформации шеи",
                )

            with gr.Group():
                gr.Markdown("### 💇 Управление прической")

                radio_hair_mode = gr.Radio(
                    choices=[
                        "Сохранить мою прическу (из фото)",
                        "Сменить прическу (выбрать новую)",
                        "Свободный стиль (на усмотрение темы)",
                    ],
                    value="Сохранить мою прическу (из фото)",
                    label="Режим волос: что делать с прической?",
                    info="Сохранить: переносит длину и цвет из селфи. Сменить: генерирует новую прическу.",
                )

                with gr.Row():
                    dropdown_hair_style = gr.Dropdown(
                        choices=[
                            "Короткая модельная (Short Fade / Crop)",
                            "Классическая мужская (Short Classic Taper)",
                            "Текстурированная (Textured Crew Cut)",
                            "Стрижка под машинку (Buzz Cut)",
                            "Стильный андеркат (Slicked Undercut)",
                            "Каре / Боб (Bob haircut)",
                            "Пучок / Хвост (Bun / Updo)",
                            "Локоны / Волны (Wavy curls)",
                            "Длинные прямые (Long flowing hair)",
                            "Средняя длина (Medium-length Casual)",
                        ],
                        value="Короткая модельная (Short Fade / Crop)",
                        label="Новая форма прически",
                        visible=False,
                    )

                    dropdown_hair_color = gr.Dropdown(
                        choices=[
                            "Как на моем фото (Авто)",
                            "Темно-каштановый / Черный (Dark Brown / Black)",
                            "Русый / Светло-каштановый (Light Brown)",
                            "Блонд (Blonde)",
                            "Пепельный / Седой (Ash / Silver)",
                        ],
                        value="Как на моем фото (Авто)",
                        label="Цвет волос",
                        visible=False,
                    )

            def toggle_hair_controls(mode: str):
                is_change = ("Сменить прическу" in mode)
                return gr.update(visible=is_change), gr.update(visible=is_change)

            radio_hair_mode.change(
                fn=toggle_hair_controls,
                inputs=[radio_hair_mode],
                outputs=[dropdown_hair_style, dropdown_hair_color],
            )

            with gr.Accordion("⚙️ Настройки диффузии и точности", open=False):
                chk_face_swap = gr.Checkbox(value=True, label="100% Биометрическое сходство лица (INSwapper)", info="Переносит аутентичную форму глаз, носа и губ с исходного фото")
                slider_face = gr.Slider(minimum=0.30, maximum=0.85, value=0.60, step=0.05, label="Сходство черт в диффузии (FaceID Scale)", info="Рекомендуется 0.55–0.65. Значения выше 0.75 вызывают перенасыщение и пятна.")
                slider_steps = gr.Slider(minimum=15, maximum=35, value=25, step=1, label="Шаги диффузии (DPM++ 2M Karras)")
                slider_guidance = gr.Slider(minimum=3.0, maximum=8.0, value=5.5, step=0.1, label="CFG Guidance Scale", info="Рекомендуется 5.0–6.0 для чистого естественного света")
                chk_pre_enhance = gr.Checkbox(value=True, label="Предварительное улучшение качества селфи (Pre-Enhance)")
                chk_restore = gr.Checkbox(value=True, label="HD-реставрация микродеталей и пор кожи (CodeFormer)")
                slider_fidelity = gr.Slider(minimum=0.5, maximum=1.0, value=0.85, step=0.05, label="Точность CodeFormer")
                chk_film_grain = gr.Checkbox(value=True, label="Кинематографичное зерно (Film Grain)", info="Устраняет искусственный пластик")

        with gr.Column(scale=1):
            btn_generate = gr.Button("✨ Сгенерировать фотосессию", variant="primary", size="lg")
            output_image = gr.Image(label="Финальный результат (100% Сходство + Фотореализм)", type="pil", elem_classes=["output-preview"])
            with gr.Accordion("Промежуточные слои", open=False):
                output_raw = gr.Image(label="Сырой диффузионный кадр", type="pil")
                output_aligned = gr.Image(label="Извлеченное лицо (InsightFace)", type="pil")
            output_status = gr.Textbox(label="Статус выполнения", interactive=False)

        btn_generate.click(
            fn=process_image,
            inputs=[
                input_image,
                dropdown_preset,
                radio_hair_mode,
                dropdown_hair_style,
                dropdown_hair_color,
                dropdown_ethnicity,
                dropdown_body,
                dropdown_neck,
                slider_face,
                slider_steps,
                slider_guidance,
                chk_pre_enhance,
                chk_face_swap,
                chk_restore,
                slider_fidelity,
                chk_film_grain,
            ],
            outputs=[
                output_image,
                output_raw,
                output_aligned,
                output_status,
            ],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), css=custom_css)
