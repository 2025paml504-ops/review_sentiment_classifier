from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else ROOT / "params.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {config_path} must be a mapping")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
