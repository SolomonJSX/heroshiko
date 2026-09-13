import asyncio
import gc
import logging
import threading
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from heroshiko.core.catalog import PresetManager, ThemePreset
from heroshiko.core.schemas import FaceAnalysisResponse, GenerateRequestParams
from heroshiko.cv.body import BodyBuildEstimator
from heroshiko.cv.face import FaceIdentity, FaceIdentityExtractor
from heroshiko.cv.hair import HairAnalyzer
from heroshiko.pipelines.postprocess import PostProcessor
from heroshiko.pipelines.sd15_engine import HeroshikoSD15Engine

logger = logging.getLogger("heroshiko.service")


class HeroshikoService:
    _instance: "HeroshikoService | None" = None
    _lock = threading.Lock()

    def __init__(self, config_path: Path | str = "configs/presets.yaml"):
        self.config_path = Path(config_path)
        logger.info("Инициализация HeroshikoService...")

        # 1. Загрузка каталога пресетов
        self.preset_manager = PresetManager(self.config_path)

        # 2. Инициализация CV-модулей (анализ лица, волос, тела)
        self.face_extractor = FaceIdentityExtractor()
        self.postprocessor = PostProcessor(face_extractor=self.face_extractor)
        self.hair_analyzer = HairAnalyzer(device="cpu")
        self.body_estimator = BodyBuildEstimator()

        # 3. Диффузионный движок (SD 1.5 Realistic Vision) — ленивая инициализация
        self._engine: HeroshikoSD15Engine | None = None
        self._engine_lock = threading.Lock()
        # Потокобезопасный замок GPU для предотвращения параллельного инференса и CUDA OOM
        self._gpu_execution_lock = threading.Lock()

    @classmethod
    def get_instance(cls, config_path: Path | str = "configs/presets.yaml") -> "HeroshikoService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path)
        return cls._instance

    @staticmethod
    def _downscale_if_needed(img: Image.Image, max_dim: int = 1920) -> Image.Image:
        """Ограничивает максимальный размер фото до 1920px для предотвращения перегрузки памяти."""
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    @property
    def engine(self) -> HeroshikoSD15Engine:
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    logger.info("Загрузка диффузионного движка Heroshiko SD 1.5...")
                    self._engine = HeroshikoSD15Engine()
        return self._engine

    def list_presets(self) -> list[ThemePreset]:
        return self.preset_manager.list_presets()

    def get_preset(self, preset_id: str) -> ThemePreset | None:
        return self.preset_manager.get_preset(preset_id)

    def analyze_face(self, image: Image.Image) -> FaceAnalysisResponse:
        """Быстрый анализ биометрии лица без вызова тяжелой диффузии."""
        rgb_img = self._downscale_if_needed(image.convert("RGB"))
        face_info: FaceIdentity | None = self.face_extractor.extract(rgb_img)

        if face_info is None:
            return FaceAnalysisResponse(
                face_detected=False,
                multiple_faces_detected=False,
                total_faces_count=0,
                message="Лицо на фотографии не обнаружено. Пожалуйста, загрузите четкое портретное селфи.",
            )

        # Анализ прически
        hair_color = "dark"
        hair_length = "medium"
        has_beard = False
        hair_prompt = "short neat haircut"

        try:
            hair_attr = self.hair_analyzer.analyze(rgb_img, face_bbox=face_info.bbox)
            hair_color = hair_attr.color
            hair_length = hair_attr.length
            has_beard = hair_attr.has_beard
            hair_prompt = hair_attr.hair_prompt
        except Exception as e:
            logger.warning(f"Ошибка при анализе волос: {e}")

        # Анализ комплекции тела
        try:
            body_info = self.body_estimator.estimate(face_identity=face_info, image=rgb_img)
            build_type = body_info.build_type
            build_label_ru = body_info.label_ru
        except Exception as e:
            logger.warning(f"Ошибка при анализе комплекции: {e}")
            build_type = "athletic"
            build_label_ru = "Стандартная атлетическая"

        bbox_list = [int(v) for v in face_info.bbox] if face_info.bbox is not None else None

        return FaceAnalysisResponse(
            face_detected=True,
            multiple_faces_detected=face_info.total_faces_count > 1,
            total_faces_count=face_info.total_faces_count,
            gender=face_info.gender,
            age=face_info.age,
            hair_color=hair_color,
            hair_length=hair_length,
            has_beard=has_beard,
            hair_prompt=hair_prompt,
            build_type=build_type,
            build_label_ru=build_label_ru,
            bbox=bbox_list,
            message="Лицо успешно распознано" if face_info.total_faces_count == 1 else f"Найдено {face_info.total_faces_count} лиц. Выбрано главное лицо для генерации.",
        )

    def generate_photoshoot(
        self,
        image: Image.Image,
        params: GenerateRequestParams,
    ) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
        """
        Потокобезопасный синхронный инференс генерации фотосессии.
        Возвращает (итоговое_изображение, кроп_лица, метаданные).
        """
        preset = self.get_preset(params.preset_id)
        if not preset:
            raise ValueError(f"Пресет '{params.preset_id}' не найден в каталоге.")

        # Защита памяти: авто-масштабирование огромных селфи со смартфонов
        rgb_input = self._downscale_if_needed(image.convert("RGB"))

        # Блокировка аппаратного доступа к GPU (только 1 поток одновременно выполняет диффузию)
        with self._gpu_execution_lock:
            try:
                # 1. Извлечение идентичности лица (InsightFace) СТРОГО из оригинального селфи!
                # Никаких искажений нейросетью до снятия биометрического вектора!
                face_info = self.face_extractor.extract(rgb_input)
                aligned_face_preview = face_info.aligned_face_chip if face_info else None

                # 2. Адаптация промпта
                prompt = preset.prompt
                negative_prompt = preset.negative_prompt
                meta: dict[str, Any] = {"preset": preset.id, "preset_title": preset.title}

                # Анализ телосложения
                body_info = self.body_estimator.estimate(
                    face_identity=face_info,
                    image=rgb_input,
                    manual_override=params.body_build,
                )
                meta["body_build"] = body_info.build_type

                # Анализ волос
                detected_color = "dark"
                detected_has_beard = False
                auto_hair_prompt = "short neat haircut"

                if face_info is not None and params.hair_mode != "preset":
                    try:
                        hair_attr = self.hair_analyzer.analyze(rgb_input, face_bbox=face_info.bbox)
                        detected_color = hair_attr.color
                        detected_has_beard = hair_attr.has_beard
                        auto_hair_prompt = hair_attr.hair_prompt
                    except Exception as e:
                        logger.warning(f"[Прическа] Пропуск анализа волос: {e}")

                # Определение цвета прически при смене стиля
                target_color = detected_color
                if params.hair_mode == "change" and params.hair_color:
                    color_sel = params.hair_color.lower()
                    if "брюнет" in color_sel or "черн" in color_sel or "black" in color_sel:
                        target_color = "dark black"
                    elif "каштан" in color_sel or "brown" in color_sel:
                        target_color = "dark brown"
                    elif "рус" in color_sel:
                        target_color = "light brown"
                    elif "блонд" in color_sel or "blond" in color_sel:
                        target_color = "blonde"
                    elif "пепельн" in color_sel or "silver" in color_sel or "grey" in color_sel:
                        target_color = "grey silver"

                # Формирование описания прически
                hair_prompt = ""
                if params.hair_mode == "keep":
                    hair_prompt = f" with {auto_hair_prompt}"
                elif params.hair_mode == "change":
                    beard_str = ", matching neat light facial hair" if detected_has_beard else ", clean shaven"
                    style_sel = (params.hair_style or "").lower()

                    if "пучок" in style_sel or "хвост" in style_sel or "bun" in style_sel:
                        hair_prompt = f" with {target_color} hair elegantly styled back in a neat bun or updo, clean natural hairline"
                    elif "каре" in style_sel or "боб" in style_sel or "bob" in style_sel:
                        hair_prompt = f" with classic chic {target_color} bob haircut, feminine cut, soft volume"
                    elif "локон" in style_sel or "волн" in style_sel or "wave" in style_sel:
                        hair_prompt = f" with long voluminous wavy curly {target_color} hair, feminine styling"
                    elif "длинн" in style_sel or "long" in style_sel:
                        hair_prompt = f" with long straight flowing {target_color} hair, natural volume"
                    elif "модельн" in style_sel or "fade" in style_sel:
                        hair_prompt = f" with neat short {target_color} haircut, short fade on sides, clean natural hairline{beard_str}"
                    elif "класс" in style_sel or "taper" in style_sel:
                        hair_prompt = f" with classic short {target_color} taper haircut, neat clean hairline{beard_str}"
                    elif "crew" in style_sel or "текстур" in style_sel:
                        hair_prompt = f" with short textured {target_color} crew cut, neat clean hairline{beard_str}"
                    elif "машинк" in style_sel or "buzz" in style_sel:
                        hair_prompt = f" with very short {target_color} buzz cut, neat hairline{beard_str}"
                    elif "андеркат" in style_sel or "undercut" in style_sel:
                        hair_prompt = f" with stylish {target_color} undercut hairstyle, short faded sides{beard_str}"
                    elif "средн" in style_sel or "medium" in style_sel:
                        hair_prompt = f" with medium-length {target_color} styled hair"
                    else:
                        hair_prompt = f" with stylish {target_color} haircut{beard_str}"

                # Этнотип, пол и анатомический профиль
                gender_param = (getattr(params, "gender", None) or "auto").lower()
                if gender_param in ("man", "male", "мужчина", "парень"):
                    effective_gender = "man"
                elif gender_param in ("woman", "female", "женщина", "девушка"):
                    effective_gender = "woman"
                elif face_info is not None and face_info.gender:
                    effective_gender = face_info.gender
                else:
                    effective_gender = "man"

                # Калибровка возраста (InsightFace склонен завышать возраст азиатских лиц)
                raw_age = face_info.age if face_info is not None else 28
                effective_age = min(max(raw_age, 21), 36)

                # Корректировка волос для мужских портретов
                if effective_gender == "man" and params.hair_mode == "keep":
                    beard_str = ", neat light facial hair" if detected_has_beard else ", clean shaven"
                    hair_prompt = f" with neat short {detected_color} masculine haircut, clean hairline{beard_str}"

                ethnicity_prompt = ""
                eth_sel = params.ethnicity.lower()
                if "central_asian" in eth_sel or "азиат" in eth_sel or "казах" in eth_sel:
                    ethnicity_prompt = f"Central Asian Kazakh {effective_gender}"
                elif "east_asian" in eth_sel:
                    ethnicity_prompt = f"East Asian {effective_gender}"
                elif "european" in eth_sel or "европ" in eth_sel:
                    ethnicity_prompt = f"European {effective_gender}"
                elif "middle_eastern" in eth_sel:
                    ethnicity_prompt = f"Middle Eastern {effective_gender}"
                elif "latino" in eth_sel:
                    ethnicity_prompt = f"Latino {effective_gender}"

                if effective_gender == "man":
                    subject_desc = ethnicity_prompt if ethnicity_prompt else "handsome masculine man, 1man"
                    gender_negatives = "(woman:1.6), (female:1.6), (feminine:1.5), (dress:1.5)"
                    prompt = prompt.replace("in a beige linen suit", "in a tailored men's beige linen suit, broad shoulders")
                    prompt = prompt.replace("open collar white shirt", "open collar men's shirt, masculine chest")
                else:
                    subject_desc = ethnicity_prompt if ethnicity_prompt else "beautiful elegant woman, 1woman"
                    gender_negatives = "(man:1.6), (male:1.6), (masculine:1.5), (beard:1.5)"
                    prompt = prompt.replace("in a beige linen suit", "in an elegant women's beige linen suit")

                # Комплекция тела с сильными весами в промпте
                body_pos = ""
                if body_info.build_type == "plus_size":
                    body_pos = "(stocky burly build:1.25), (broad solid frame:1.2), " if effective_gender == "man" else "(curvy plus-size build:1.25), (fuller frame:1.2), "
                elif body_info.build_type == "athletic":
                    body_pos = "(athletic muscular build:1.2), " if effective_gender == "man" else "(athletic fit toned build:1.2), "
                elif body_info.build_type == "slender":
                    body_pos = "(slender lean build:1.2), "

                target_subject = f"a {effective_age}yo {body_pos}{subject_desc}{hair_prompt}"
                for placeholder in ["a person", "an elegant person", "a handsome person"]:
                    if placeholder in prompt:
                        prompt = prompt.replace(placeholder, target_subject)
                        break

                meta["gender"] = effective_gender
                meta["age"] = effective_age

                # Критические негативные модификаторы: шея + комплекция + пол (должны быть в начале!)
                neck_negatives = "(long neck:1.4), (elongated neck:1.4), (distorted neck:1.3), (disproportional neck:1.3)"
                body_neg = f", {body_info.negative_modifier}" if body_info.negative_modifier else ""
                negative_prompt = f"{neck_negatives}, {gender_negatives}{body_neg}, {negative_prompt}"

                # 4. Баланс силы лица IP-Adapter FaceID:
                # 0.40 обеспечивает гармоничную базу головы, шеи и ракурса,
                # а финальный 100% перенос биометрии выполняет INSwapper!
                eff_face_strength = min(float(params.face_strength), 0.42)
                if "group" in preset.id:
                    eff_face_strength = min(eff_face_strength, 0.35)

                # 5. Генерация через SD 1.5 Realistic Vision
                sd15 = self.engine
                generated = sd15.generate(
                    face_info=face_info,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    face_strength=eff_face_strength,
                    steps=params.steps,
                    guidance_scale=params.guidance_scale,
                    width=512,
                    height=768,
                )

                # 6. Финальная шлифовка и 100% перенос лица (INSwapper + CodeFormer с защитой шеи + Film Grain)
                final_output = self.postprocessor.process(
                    generated_img=generated,
                    source_face_info=face_info,
                    apply_face_swap=params.apply_face_swap,
                    apply_face_restore=params.apply_face_restore,
                    fidelity_weight=params.fidelity_weight,
                    match_lighting=True,
                    apply_film_grain=params.apply_film_grain,
                )

                return final_output, aligned_face_preview, meta

            finally:
                # Обязательная очистка кэша VRAM после каждого прохода
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

    def get_vram_info(self) -> dict[str, Any]:
        cuda_ok = torch.cuda.is_available()
        res: dict[str, Any] = {"cuda_available": cuda_ok, "device_name": None, "models_loaded": []}
        if cuda_ok:
            res["device_name"] = torch.cuda.get_device_name(0)
            res["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
            res["vram_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1)
            res["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)

        loaded = ["FaceIdentityExtractor", "PostProcessor", "HairAnalyzer", "BodyBuildEstimator"]
        if self._engine is not None:
            loaded.append("HeroshikoSD15Engine")
        res["models_loaded"] = loaded
        return res
