"""Silero TTS (v4_ru). CPU real-time, good Russian prosody, '+' stress marks.

License note: Silero model weights are CC BY-NC-SA 4.0 (non-commercial).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .base import TTSEngine


class SileroTTS(TTSEngine):
    name = "silero"

    def __init__(self, cfg: dict, cache_dir: Path):
        super().__init__(cfg, cache_dir)
        self.sample_rate = int(cfg.get("sample_rate", 48000))
        self._model = None

    def _load(self):
        if self._model is None:
            import torch

            url = self.cfg["model_url"]
            path = self.cache_dir.parent / "models" / Path(url).name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, path)
            torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
            model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
            model.to(torch.device(self.cfg.get("device", "cpu")))
            self._model = model
        return self._model

    def synth(self, text: str) -> np.ndarray:
        model = self._load()
        audio = model.apply_tts(
            text=text,
            speaker=self.cfg.get("speaker", "aidar"),
            sample_rate=self.sample_rate,
            put_accent=bool(self.cfg.get("put_accent", True)),
            put_yo=bool(self.cfg.get("put_yo", True)),
        )
        return audio.cpu().numpy().astype(np.float32)
