from __future__ import annotations

import hashlib
import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import soundfile as sf

ENGINES = {"silero": "factory.speech.tts.silero:SileroTTS"}


class TTSEngine(ABC):
    name: str = "base"
    sample_rate: int = 48000

    def __init__(self, cfg: dict, cache_dir: Path):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)

    @abstractmethod
    def synth(self, text: str) -> np.ndarray:
        """Return float32 mono samples at self.sample_rate."""

    def voice_id(self) -> str:
        return json.dumps(self.cfg, sort_keys=True)

    def synth_to_file(self, text: str, out: Path) -> Path:
        """Synthesize with a content-addressed cache (text + engine config)."""
        key = hashlib.sha1(f"{self.name}|{self.voice_id()}|{text}".encode()).hexdigest()[:16]
        cached = self.cache_dir / f"{self.name}_{key}.wav"
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            sf.write(cached, self.synth(text).astype(np.float32), self.sample_rate, subtype="PCM_16")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(cached.read_bytes())
        return out


def get_engine(cfg: dict, cache_dir: Path) -> TTSEngine:
    name = cfg["engine"]
    mod, cls = ENGINES[name].split(":")
    return getattr(importlib.import_module(mod), cls)(cfg.get(name, {}), Path(cache_dir) / "tts")
