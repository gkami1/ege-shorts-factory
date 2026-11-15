"""Config loading: one YAML file, deep-merged over the packaged defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "factory.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else ROOT / p
    return cfg
