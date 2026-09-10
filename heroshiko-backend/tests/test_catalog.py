from pathlib import Path

import pytest

from heroshiko.core.catalog import PresetManager, ThemePreset


def test_load_presets_success(sample_presets_yaml: Path):
    manager = PresetManager(sample_presets_yaml)
    presets = manager.list_presets()

    assert len(presets) == 1
    assert isinstance(presets[0], ThemePreset)
    assert presets[0].id == "test_theme"
    assert presets[0].title == "Тестовая тема"
    assert presets[0].control_strength == 0.75


def test_get_nonexistent_preset(sample_presets_yaml: Path):
    manager = PresetManager(sample_presets_yaml)
    assert manager.get_preset("unknown_preset") is None


def test_missing_config_raises_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        PresetManager(tmp_path / "non_existing.yaml")
