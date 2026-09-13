from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from heroshiko.cv.face import FaceIdentity


@dataclass
class BodyBuildInfo:
    """Результат анализа телосложения / комплекции."""

    build_type: Literal["plus_size", "regular", "slender", "athletic"]
    label_ru: str
    confidence: float
    fwhr: float
    cheek_ratio: float
    prompt_modifier: str
    negative_modifier: str


class BodyBuildEstimator:
    """
    Оценивает полноту / комплекцию тела на основе биометрических показателей лица:
    - FWHR (Facial Width-to-Height Ratio: отношение ширины скул к высоте лица)
    - Cheek-to-Eye Ratio (полнота щек относительно межзрачкового расстояния)
    - Jaw-to-Cheek Fullness (ширина нижней челюсти и угол челюстной дуги)
    """

    def __init__(self):
        pass

    def estimate(
        self,
        face_identity: FaceIdentity,
        image: Image.Image | None = None,
        manual_override: str | None = None,
    ) -> BodyBuildInfo:
        gender = face_identity.gender if face_identity else "man"

        # 1. Если задан ручной выбор в интерфейсе
        if manual_override and manual_override != "Авто (определить по чертам лица)":
            return self._build_from_choice(manual_override, gender)

        # 2. Автоматический биометрический анализ по ключевым точкам (InsightFace 68 landmarks)
        fwhr = 1.05
        cheek_ratio = 2.05
        buccal_ratio = 1.70

        if face_identity and face_identity.raw_face is not None:
            raw = face_identity.raw_face
            if hasattr(raw, "landmark_3d_68") and raw.landmark_3d_68 is not None:
                lmk = raw.landmark_3d_68[:, :2]
                left_eye = lmk[36:42].mean(axis=0)
                right_eye = lmk[42:48].mean(axis=0)
                eye_dist = max(float(np.linalg.norm(right_eye - left_eye)), 1e-3)

                # Ширина скул (точки 3 и 13)
                cheek_w = float(np.linalg.norm(lmk[13] - lmk[3]))
                # Ширина щек/челюсти (точки 4 и 12)
                buccal_w = float(np.linalg.norm(lmk[12] - lmk[4]))
                # Высота лица от подбородка до центра глаз
                eye_mid = (left_eye + right_eye) / 2.0
                face_h = max(float(np.linalg.norm(lmk[8] - eye_mid)), 1e-3)

                fwhr = cheek_w / face_h
                cheek_ratio = cheek_w / eye_dist
                buccal_ratio = buccal_w / eye_dist

        # Расчет индекса полноты (Facial Adiposity Score)
        # У стройных людей fwhr ~ 0.95..1.08, cheek_ratio ~ 1.85..2.10
        # У полных / пышных людей fwhr > 1.15, cheek_ratio > 2.25, buccal_ratio > 1.85
        adiposity_score = (fwhr - 1.0) * 2.5 + (cheek_ratio - 2.0) * 1.8 + (buccal_ratio - 1.7) * 1.5

        if adiposity_score >= 0.55:
            build_type = "plus_size"
        elif adiposity_score <= -0.25:
            build_type = "slender"
        else:
            build_type = "regular"

        return self._build_from_type(build_type, gender, fwhr=fwhr, cheek_ratio=cheek_ratio)

    def _build_from_choice(self, choice: str, gender: str) -> BodyBuildInfo:
        c = (choice or "").lower().strip()
        if any(k in c for k in ("plus", "пышн", "полн", "плотн", "крупн", "stocky", "heavy", "burly")):
            return self._build_from_type("plus_size", gender, is_manual=True)
        elif any(k in c for k in ("slender", "slim", "стройн", "худ", "lean", "skinny")):
            return self._build_from_type("slender", gender, is_manual=True)
        elif any(k in c for k in ("athletic", "fit", "спорт", "атлет", "мускул", "muscular")):
            return self._build_from_type("athletic", gender, is_manual=True)
        else:
            return self._build_from_type("regular", gender, is_manual=True)

    def _build_from_type(
        self,
        build_type: Literal["plus_size", "regular", "slender", "athletic"],
        gender: str,
        fwhr: float = 1.0,
        cheek_ratio: float = 2.0,
        is_manual: bool = False,
    ) -> BodyBuildInfo:
        is_woman = (gender == "woman")

        if build_type == "plus_size":
            label = "пышная / плотная фигура (Plus-size)"
            if is_woman:
                pos = (
                    "(curvy plus-size woman:1.25), (fuller natural build:1.2), broad shoulders, solid natural frame, "
                    "realistic natural fuller proportions, modest elegant attire, fully clothed"
                )
                neg = (
                    "(skinny:1.4), (slender:1.4), (thin waist:1.35), (narrow hips:1.3), (emaciated:1.4), "
                    "cleavage, exposed chest, bare chest, deep neckline, plunging neckline, revealing clothes, "
                    "unbuttoned jacket, shirtless, underwear, lingerie, nsfw, nude, naked, topless, "
                    "overly large breasts, huge breasts, close up, macro, cropped head, flat stomach, thin arms"
                )
            else:
                pos = (
                    "(stocky burly build:1.25), (broad solid frame:1.2), heavy natural frame, fuller masculine physique, "
                    "realistic natural fuller body, fully clothed"
                )
                neg = (
                    "(skinny:1.4), (slender:1.4), (thin waist:1.35), (narrow frame:1.3), (emaciated:1.4), "
                    "close up, macro, cropped head, fragile, underweight, bony, thin arms"
                )
        elif build_type == "slender":
            label = "стройная фигура (Slender / Slim)"
            if is_woman:
                pos = (
                    "(slender lean female build:1.2), slim elegant silhouette, delicate natural proportions, "
                    "modest elegant attire, fully clothed"
                )
                neg = (
                    "(overweight:1.35), (chubby:1.35), (heavy:1.3), (fat:1.35), (plus-size:1.35), bulky, plump, double chin, "
                    "cleavage, exposed chest, bare chest, deep neckline, plunging neckline, revealing clothes, "
                    "unbuttoned, nsfw, nude, topless, close up, cropped head"
                )
            else:
                pos = (
                    "(slender lean male build:1.2), slim elegant silhouette, light athletic frame, fully clothed"
                )
                neg = (
                    "(overweight:1.35), (chubby:1.35), (heavy:1.3), (fat:1.35), (plus-size:1.35), bulky, plump, double chin, "
                    "close up, cropped head"
                )
        elif build_type == "athletic":
            label = "атлетическая / спортивная (Athletic / Fit)"
            if is_woman:
                pos = (
                    "(athletic toned female build:1.2), fit sculpted silhouette, firm posture, healthy physique, "
                    "modest elegant attire, fully clothed"
                )
                neg = (
                    "(flabby:1.3), (overweight:1.3), (chubby:1.3), (emaciated:1.35), (frail:1.35), "
                    "cleavage, exposed chest, bare chest, deep neckline, revealing clothes, nsfw, nude, topless, "
                    "close up, cropped head"
                )
            else:
                pos = (
                    "(athletic muscular build man:1.2), broad shoulders, fit sculpted torso, strong physique, fully clothed"
                )
                neg = (
                    "(flabby:1.3), (overweight:1.3), (chubby:1.3), (emaciated:1.35), (frail:1.35), "
                    "close up, cropped head"
                )
        else:  # regular
            label = "естественная / стандартная (Natural build)"
            pos = (
                "natural realistic body proportions, balanced healthy build, modest elegant attire, fully clothed"
            )
            neg = (
                "cleavage, exposed chest, bare chest, deep neckline, revealing clothes, nsfw, nude, topless, "
                "close up, cropped head, emaciated, extremely skinny, morbidly obese"
            )

        confidence = 1.0 if is_manual else 0.85
        return BodyBuildInfo(
            build_type=build_type,
            label_ru=label,
            confidence=confidence,
            fwhr=fwhr,
            cheek_ratio=cheek_ratio,
            prompt_modifier=pos,
            negative_modifier=neg,
        )
