from __future__ import annotations

import hashlib
import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import soundfile as sf

from ..normalize import stress_format

ENGINES = {
    "silero": "factory.speech.tts.silero:SileroTTS",
    "qwen": "factory.speech.tts.qwen:QwenTTS",
}

# Engines whose weights may be used on a monetized channel. Anything else is preview-only.
COMMERCIAL_OK = {"qwen"}


class TTSEngine(ABC):
    name: str = "base"
    sample_rate: int = 48000
    deterministic: bool = False      # same text → same audio (no point re-taking)
    default_stress: str = "strip"  # how '+' stress marks reach the engine: plus | acute | strip

    def __init__(self, cfg: dict, cache_dir: Path):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)

    @abstractmethod
    def synth(self, text: str, seed: int = 0) -> np.ndarray:
        """Return float32 mono samples at self.sample_rate. `seed` varies the take on retries."""

    def prepare(self, spoken: str) -> str:
        """Normalized text → the exact string this engine reads."""
        return stress_format(spoken, self.cfg.get("stress", self.default_stress))

    def voice_id(self) -> str:
        return json.dumps(self.cfg, sort_keys=True)

    def describe(self) -> str:
        return self.name

    def synth_to_file(self, text: str, out: Path, attempt: int = 0) -> Path:
        """Synthesize with a content-addressed cache (text + engine config + attempt)."""
        key = hashlib.sha1(f"{self.name}|{self.voice_id()}|{text}|{attempt}".encode()).hexdigest()[:16]
        cached = self.cache_dir / f"{self.name}_{key}.wav"
        if cached.exists():
            self.sample_rate = sf.info(cached).samplerate
        else:
            cached.parent.mkdir(parents=True, exist_ok=True)
            audio = self.synth(text, seed=attempt)
            sf.write(cached, _tidy(audio, self.sample_rate), self.sample_rate, subtype="PCM_16")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(cached.read_bytes())
        return out

    def unload(self) -> None:
        """Free model memory (bake-off loads engines one after another)."""


def _tidy(a: np.ndarray, sr: int) -> np.ndarray:
    """Float32 mono, peak ≤ -1 dBFS, leading/trailing silence trimmed to 50 ms."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return a
    peak = float(np.abs(a).max())
    if peak > 0.89:
        a = a * (0.89 / peak)
    hop = max(1, int(sr * 0.01))
    env = np.abs(a)
    thr = max(env.max() * 0.02, 1e-4)
    idx = np.where(env > thr)[0]
    if idx.size:
        lo = max(0, idx[0] - 5 * hop)
        hi = min(a.size, idx[-1] + 5 * hop)
        a = a[lo:hi]
    return a


def get_engine(cfg: dict, cache_dir: Path) -> TTSEngine:
    name = cfg["engine"]
    mod, cls = ENGINES[name].split(":")
    return getattr(importlib.import_module(mod), cls)(cfg.get(name, {}), Path(cache_dir) / "tts")
