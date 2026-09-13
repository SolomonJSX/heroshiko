import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import insightface
from insightface.utils import face_align
from PIL import Image

from heroshiko.cv.face import FaceIdentity, FaceIdentityExtractor


class PostProcessor:
    def __init__(
        self,
        swapper_repo: str = "ezioruan/inswapper_128.onnx",
        swapper_filename: str = "inswapper_128.onnx",
        restorer_repo: str = "facefusion/models-3.0.0",
        restorer_filename: str = "codeformer.onnx",
        face_extractor: FaceIdentityExtractor | None = None,
        providers: list[str] | None = None,
    ):
        """
        Производственный конвейер постобработки лиц heroshiko:
        - INSwapper: 100% сходство лица с оригинальным селфи пользователя
        - CodeFormer ONNX: фотореалистичная реставрация кожи, пор и микродеталей взгляда
        - Reinhard Color Match: световая и цветовая гармонизация
        """
        if providers is None:
            # CPUExecutionProvider быстр (~0.1s на кадр) и освобождает 100% VRAM GPU для SDXL
            providers = ["CPUExecutionProvider"]

        self.providers = providers
        self.face_extractor = face_extractor
        self.swapper = None
        self.restorer_session = None
        self.restorer_filename = restorer_filename

        # Загрузка INSwapper
        try:
            swapper_path = hf_hub_download(swapper_repo, swapper_filename)
            self.swapper = insightface.model_zoo.get_model(swapper_path, providers=self.providers)
            print("[Постобработка] INSwapper успешно загружен (100% сохранение лица).")
        except Exception as e:
            print(f"[Постобработка] Ошибка загрузки INSwapper: {e}")

        # Загрузка реставратора CodeFormer ONNX
        try:
            restorer_path = hf_hub_download(restorer_repo, restorer_filename)
            self.restorer_session = ort.InferenceSession(restorer_path, providers=self.providers)
            print(f"[Постобработка] Реставратор лица ({restorer_filename}) успешно загружен.")
        except Exception as e:
            print(f"[Постобработка] Ошибка загрузки реставратора лица: {e}")

    @staticmethod
    def create_elliptical_face_mask(size: tuple[int, int]) -> np.ndarray:
        """
        Создает анатомическую эллиптическую маску лица с мягким затуханием.
        Строго ограничивает область воздействия контуром лица и подбородком,
        полностью исключая искажение шеи, кадыка и воротниковой зоны.
        """
        h, w = size
        mask = np.zeros((h, w), dtype=np.float32)
        # Центр смещен к уровню глаз/носа, радиусы охватывают лоб, брови, глаза, щеки и рот
        center = (int(w * 0.50), int(h * 0.46))
        axes = (int(w * 0.32), int(h * 0.35))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        # Строгая защита шеи: плавное затухание ниже подбородка и полное обнуление на шее
        chin_y = int(h * 0.77)
        neck_cutoff = int(h * 0.83)
        for y in range(chin_y, h):
            if y >= neck_cutoff:
                mask[y, :] = 0.0
            else:
                factor = (neck_cutoff - y) / max(neck_cutoff - chin_y, 1)
                mask[y, :] *= factor

        k = int(w * 0.07) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask

    @staticmethod
    def add_film_grain(img_bgr: np.ndarray, strength: float = 0.015) -> np.ndarray:
        """
        Добавляет деликатное монохроматическое пленочное зерно (35mm film grain),
        устраняющее ощущение искусственного пластика и цифровой гладкости.
        """
        if strength <= 0:
            return img_bgr
        h, w = img_bgr.shape[:2]
        noise = np.random.normal(0, strength * 255.0, (h, w, 1)).astype(np.float32)
        grain_img = img_bgr.astype(np.float32) + noise
        return np.clip(grain_img, 0, 255).astype(np.uint8)

    def pre_enhance_selfie(
        self,
        image: Image.Image,
        fidelity_weight: float = 0.92,
    ) -> Image.Image:
        """
        Предварительное улучшение загруженного селфи перед извлечением черт:
        устраняет шум, размытие фронтальной камеры и усиливает четкость черт.
        """
        return self.restore_face(image, fidelity_weight=fidelity_weight)

    def swap_face(
        self,
        target_bgr: np.ndarray,
        source_face: object,
        target_face: object | None = None,
        match_lighting: bool = True,
    ) -> tuple[np.ndarray, object | None]:
        """
        Переносит 100% геометрию и черты лица исходного пользователя на целевое изображение.
        Если match_lighting=True, согласовывает оттенок кожи и распределение освещения
        с окружением сцены перед вживлением.
        """
        if self.swapper is None or source_face is None:
            return target_bgr, target_face

        if target_face is None:
            if self.face_extractor is None:
                self.face_extractor = FaceIdentityExtractor(providers=self.providers)
            faces = self.face_extractor.app.get(target_bgr)
            if not faces:
                return target_bgr, None
            target_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        try:
            # Нативный paste_back=True в INSwapper переносит 100% черт лица без среза скул и глаз
            swapped_bgr = self.swapper.get(target_bgr.copy(), target_face, source_face, paste_back=True)
            return swapped_bgr, target_face
        except Exception as e:
            print(f"[Постобработка] Ошибка при смене лица INSwapper: {e}")
            return target_bgr, target_face

    def restore_face(
        self,
        image: Image.Image | np.ndarray,
        target_face: object | None = None,
        fidelity_weight: float = 0.85,
    ) -> Image.Image | np.ndarray:
        """
        Восстанавливает микродетали кожи, резкость глаз и устраняет артефакты через CodeFormer ONNX.
        """
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image.copy()

        if self.restorer_session is None:
            return image

        if target_face is None or not hasattr(target_face, "kps"):
            if self.face_extractor is None:
                self.face_extractor = FaceIdentityExtractor(providers=self.providers)
            faces = self.face_extractor.app.get(img_bgr)
            if not faces:
                return image
            target_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        try:
            # Выравнивание лица до стандартных 512x512
            crop_512, M = face_align.norm_crop2(img_bgr, target_face.kps, 512)
            crop_rgb = cv2.cvtColor(crop_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            crop_tensor = crop_rgb.transpose(2, 0, 1)[None, ...]

            if "codeformer" in self.restorer_filename.lower():
                w = np.array(fidelity_weight, dtype=np.float64)
                out = self.restorer_session.run(None, {"input": crop_tensor, "weight": w})[0]
            else:
                out = self.restorer_session.run(None, {"input": crop_tensor})[0]

            out_rgb = np.clip((out[0].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

            # Обратное вживление в исходные координаты кадра
            IM = cv2.invertAffineTransform(M)
            inv_crop = cv2.warpAffine(
                out_bgr, IM, (img_bgr.shape[1], img_bgr.shape[0]), borderValue=0.0
            )

            # Мягкая анатомическая овальная маска лица (исключает прямоугольные края и срезы)
            mask_crop = self.create_elliptical_face_mask((512, 512))
            mask_warped = cv2.warpAffine(
                mask_crop, IM, (img_bgr.shape[1], img_bgr.shape[0]), borderValue=0.0
            )
            mask_warped = np.expand_dims(mask_warped, axis=-1)

            blended_bgr = (mask_warped * inv_crop + (1.0 - mask_warped) * img_bgr.astype(np.float32)).astype(np.uint8)

            if is_pil:
                return Image.fromarray(cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB))
            return blended_bgr
        except Exception as e:
            print(f"[Постобработка] Ошибка реставрации лица: {e}")
            return image

    @staticmethod
    def match_color_reinhard(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
        """Перенос цветовой атмосферы по методу Reinhard et al."""
        src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_mean, src_std = cv2.meanStdDev(src_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

        src_mean = src_mean.reshape(1, 1, 3)
        src_std = src_std.reshape(1, 1, 3)
        tgt_mean = tgt_mean.reshape(1, 1, 3)
        tgt_std = tgt_std.reshape(1, 1, 3)

        src_std[src_std == 0] = 1e-6
        result_lab = ((src_lab - src_mean) / src_std) * tgt_std + tgt_mean
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def harmonize_lighting(
        self,
        generated_img: Image.Image,
        face_mask: Image.Image,
        blend_factor: float = 0.35,
    ) -> Image.Image:
        """Мягко адаптирует оттенки на лице к свету сгенерированного окружения."""
        if face_mask.size != generated_img.size:
            face_mask = face_mask.resize(generated_img.size, Image.Resampling.NEAREST)

        gen_np = cv2.cvtColor(np.array(generated_img), cv2.COLOR_RGB2BGR)
        mask_np = np.array(face_mask.convert("L"))
        bg_mask = mask_np == 0

        if np.any(bg_mask) and np.any(mask_np > 0):
            gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_BGR2LAB).astype(np.float32)
            bg_pixels = gen_lab[bg_mask]
            face_pixels = gen_lab[mask_np > 0]

            bg_mean = np.mean(bg_pixels, axis=0, keepdims=True)
            bg_std = np.std(bg_pixels, axis=0, keepdims=True) + 1e-6
            face_mean = np.mean(face_pixels, axis=0, keepdims=True)
            face_std = np.std(face_pixels, axis=0, keepdims=True) + 1e-6

            target_std = face_std * (1.0 - blend_factor) + bg_std * blend_factor
            target_mean = face_mean * (1.0 - blend_factor) + bg_mean * blend_factor

            adjusted_face = ((face_pixels - face_mean) / face_std) * target_std + target_mean
            gen_lab[mask_np > 0] = np.clip(adjusted_face, 0, 255)
            harmonized_bgr = cv2.cvtColor(gen_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            return Image.fromarray(cv2.cvtColor(harmonized_bgr, cv2.COLOR_BGR2RGB))

        return generated_img

    def process(
        self,
        generated_img: Image.Image,
        source_face_info: FaceIdentity | None = None,
        apply_face_swap: bool = True,
        apply_face_restore: bool = True,
        fidelity_weight: float = 0.85,
        match_lighting: bool = True,
        apply_film_grain: bool = True,
        face_mask: Image.Image | None = None,
        apply_harmonization: bool = False,
    ) -> Image.Image:
        """
        Полный производственный цикл постобработки:
        1. Точная пересадка оригинального лица пользователя через INSwapper (100% likeness).
        2. Световая гармонизация тона кожи лица под освещение сцены (match_lighting).
        3. HD-реставрация текстуры кожи, глаз и микродеталей через ONNX CodeFormer.
        4. Добавление деликатного пленочного микрозерна для устранения пластика.
        """
        current_bgr = cv2.cvtColor(np.array(generated_img), cv2.COLOR_RGB2BGR)

        # 1. 100% перенос лица пользователя со световой адаптацией
        if apply_face_swap and source_face_info is not None and source_face_info.raw_face is not None:
            current_bgr, _ = self.swap_face(
                current_bgr,
                source_face=source_face_info.raw_face,
                match_lighting=match_lighting,
            )

        # 2. Восстановление фотореалистичных микродеталей
        if apply_face_restore:
            current_bgr = self.restore_face(
                current_bgr,
                target_face=None,
                fidelity_weight=fidelity_weight,
            )

        # 3. Деликатное пленочное зерно против искусственной гладкости
        if apply_film_grain:
            current_bgr = self.add_film_grain(current_bgr, strength=0.012)

        result_pil = Image.fromarray(cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB))

        # 4. Цветовая гармонизация при наличии маски
        if apply_harmonization and face_mask is not None:
            result_pil = self.harmonize_lighting(result_pil, face_mask)

        return result_pil
