"""Task bank: YAML files under tasks/ (one task per file)."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import Task


def load_task_file(path: str | Path) -> Task:
    return Task.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_bank(root: str | Path) -> dict[str, Task]:
    bank: dict[str, Task] = {}
    for p in sorted(Path(root).rglob("*.yaml")):
        t = load_task_file(p)
        if t.id in bank:
            raise ValueError(f"duplicate task id {t.id} in {p}")
        if p.stem != t.id:
            raise ValueError(f"{p}: file name must equal id {t.id}")
        bank[t.id] = t
    return bank
