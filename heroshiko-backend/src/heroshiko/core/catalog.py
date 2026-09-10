from pathlib import Path

import yaml
from pydantic import BaseModel


class ThemePreset(BaseModel):
    id: str
    title: str
    prompt: str
    negative_prompt: str
    control_strength: float = 0.75
    denoising_strength: float = 0.65


class PresetManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._presets: dict[str, ThemePreset] = {}
        self.load_presets()

    def load_presets(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at {self.config_path}")

        with open(self.config_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        for preset_id, data in raw_data.get("presets", {}).items():
            self._presets[preset_id] = ThemePreset(id=preset_id, **data)

    def get_preset(self, preset_id: str) -> ThemePreset | None:
        return self._presets.get(preset_id)

    def list_presets(self) -> list[ThemePreset]:
        return list(self._presets.values())
