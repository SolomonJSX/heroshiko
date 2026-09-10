import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor


class PostProcessor:
    def __init__(
        self,
        codeformer_checkpoint: str = "weights/codeformer/codeformer.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.codeformer_net = None

        # Инициализируем CodeFormer, если веса доступны локально
        try:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            # Легковесный хелпер детекции и кропа лица для реставрации
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=self.device,
            )
            self._init_codeformer(codeformer_checkpoint)
        except Exception as e:
            print(
                f"[Постобработка] CodeFormer не инициализирован ({e}). Будет доступна только цветовая гармонизация."
            )

    def _init_codeformer(self, checkpoint_path: str):
        """Загрузка легковесной архитектуры CodeFormer."""
        try:
            from codeformer.archs.codeformer_arch import CodeFormer

            net = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(self.device)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")["params_ema"]
            net.load_state_dict(checkpoint)
            net.eval()
            self.codeformer_net = net
            print("[Постобработка] Модель CodeFormer успешно загружена.")
        except Exception as err:
            print(f"[Постобработка] Не удалось загрузить веса CodeFormer: {err}")

    @staticmethod
    def match_color_reinhard(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
        """
        Перенос цветовой атмосферы по методу Reinhard et al.
        Переводит изображения в пространство Lab, совмещает средние значения и СКО.
        """
        src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Вычисляем статистики по каналам L, a, b
        src_mean, src_std = cv2.meanStdDev(src_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

        src_mean = src_mean.reshape(1, 1, 3)
        src_std = src_std.reshape(1, 1, 3)
        tgt_mean = tgt_mean.reshape(1, 1, 3)
        tgt_std = tgt_std.reshape(1, 1, 3)

        # Защита от деления на 0
        src_std[src_std == 0] = 1e-6

        # Нормализация и перенос шкалы
        result_lab = ((src_lab - src_mean) / src_std) * tgt_std + tgt_mean
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def harmonize_lighting(
        self,
        generated_img: Image.Image,
        face_mask: Image.Image,
        blend_factor: float = 0.35,
    ) -> Image.Image:
        """
        Мягко адаптирует оттенки на лице к свету сгенерированного окружения.
        blend_factor: 0.0 - без изменений, 1.0 - полный перенос палитры фона на лицо.
        """
        if face_mask.size != generated_img.size:
            face_mask = face_mask.resize(generated_img.size, Image.Resampling.NEAREST)

        gen_np = cv2.cvtColor(np.array(generated_img), cv2.COLOR_RGB2BGR)
        mask_np = np.array(face_mask.convert("L"))
        bg_mask = mask_np == 0

        # Если на кадре есть и фон, и защищенная область лица
        if np.any(bg_mask) and np.any(mask_np > 0):
            gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_BGR2LAB).astype(np.float32)
            bg_pixels = gen_lab[bg_mask]
            face_pixels = gen_lab[mask_np > 0]

            bg_mean = np.mean(bg_pixels, axis=0, keepdims=True)
            bg_std = np.std(bg_pixels, axis=0, keepdims=True) + 1e-6
            face_mean = np.mean(face_pixels, axis=0, keepdims=True)
            face_std = np.std(face_pixels, axis=0, keepdims=True) + 1e-6

            # Мягко сдвигаем цветовую температуру и экспозицию лица в сторону фона
            target_std = face_std * (1.0 - blend_factor) + bg_std * blend_factor
            target_mean = face_mean * (1.0 - blend_factor) + bg_mean * blend_factor

            adjusted_face = ((face_pixels - face_mean) / face_std) * target_std + target_mean
            gen_lab[mask_np > 0] = np.clip(adjusted_face, 0, 255)
            harmonized_bgr = cv2.cvtColor(gen_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            return Image.fromarray(cv2.cvtColor(harmonized_bgr, cv2.COLOR_BGR2RGB))

        return generated_img

    def restore_face(
        self,
        image: Image.Image,
        fidelity_weight: float = 0.8,
    ) -> Image.Image:
        """
        Реставрация деталей лица через CodeFormer.
        fidelity_weight:
            1.0 = максимальное сходство с исходным сгенерированным лицом
            0.6 = более агрессивное восстановление резкости и удаление артефактов
        """
        if self.codeformer_net is None or not hasattr(self, "face_helper"):
            return image

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.face_helper.clean_all()
        self.face_helper.read_image(img_bgr)
        self.face_helper.get_face_landmarks_5(
            only_center_face=True, resize=640, eye_dist_threshold=5
        )
        self.face_helper.align_warp_face()

        for cropped_face in self.face_helper.cropped_faces:
            # Подготовка тензора для CodeFormer
            face_tensor = to_tensor(cropped_face)
            face_tensor = normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))[None, ...].to(
                self.device
            )

            with torch.no_grad():
                output = self.codeformer_net(face_tensor, w=fidelity_weight, adain=True)[0]
                restored_face = output.squeeze(0).permute(1, 2, 0).cpu().clamp_(-1, 1).numpy()
                restored_face = ((restored_face + 1) / 2.0 * 255.0).astype(np.uint8)

            self.face_helper.add_restored_face(restored_face)

        # Вставка отреставрированного лица обратно в исходное изображение с бесшовным смешиванием
        self.face_helper.get_inverse_affine(None)
        restored_img_bgr = self.face_helper.paste_faces_to_input_image()

        return Image.fromarray(cv2.cvtColor(restored_img_bgr, cv2.COLOR_BGR2RGB))

    def process(
        self,
        generated_img: Image.Image,
        face_mask: Image.Image,
        apply_harmonization: bool = True,
        apply_face_restoration: bool = True,
        fidelity_weight: float = 0.8,
    ) -> Image.Image:
        """Полный цикл пост-обработки кадра."""
        current_img = generated_img

        if apply_harmonization:
            current_img = self.harmonize_lighting(current_img, face_mask)

        if apply_face_restoration:
            current_img = self.restore_face(current_img, fidelity_weight=fidelity_weight)

        return current_img
