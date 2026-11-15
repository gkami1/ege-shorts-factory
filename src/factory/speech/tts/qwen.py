"""Qwen3-TTS (Apache-2.0, weights and code; Russian is one of the 10 supported languages).

Three modes, all through the `qwen-tts` package:
  clone   — Base model + a fixed reference clip (assets/voices/<voice>.wav + .txt). This is the
            production mode: one narrator voice for every video. Make the reference once with
            `factory voice-design` (VoiceDesign model, text description → voice) and `factory voice-pick`.
  custom  — CustomVoice model with a preset speaker (Ryan, Aiden, Uncle_Fu, …) + optional instruct.
            None of the presets is a native Russian speaker; used as bake-off baselines.
  design  — VoiceDesign model directly (voice from a text description on every call; the timbre
            drifts between calls, so only for auditioning).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ...config import ROOT
from .base import TTSEngine

MODELS = {
    "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


def load_model(model_id: str, device: str = "auto", dtype: str = "auto", attn: str = "auto"):
    import torch
    from qwen_tts import Qwen3TTSModel

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    on_gpu = device.startswith("cuda")
    if dtype == "auto":
        dtype = "bfloat16" if on_gpu else "float32"
    if attn == "auto":
        attn = "sdpa"
        if on_gpu:
            try:
                import flash_attn  # noqa: F401

                attn = "flash_attention_2"
            except ImportError:
                pass
    print(f"[qwen-tts] loading {model_id} on {device} ({dtype}, {attn})", flush=True)
    return Qwen3TTSModel.from_pretrained(
        model_id, device_map=device, dtype=getattr(torch, dtype), attn_implementation=attn
    )


def seed_all(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def voice_paths(voice: str) -> tuple[Path, Path]:
    p = Path(voice)
    if not p.is_absolute():
        p = ROOT / p
    return p.with_suffix(".wav"), p.with_suffix(".txt")


class QwenTTS(TTSEngine):
    name = "qwen"
    default_stress = "strip"

    def __init__(self, cfg: dict, cache_dir: Path):
        super().__init__(cfg, cache_dir)
        self.mode = cfg.get("mode", "clone")
        self.model_id = cfg.get("model") or MODELS[self.mode]
        self._model = None
        self._prompt = None
        self.sample_rate = int(cfg.get("sample_rate", 24000))

    def voice_id(self) -> str:
        base = super().voice_id()
        if self.mode == "clone":  # the reference clip is part of the voice
            wav, txt = voice_paths(self.cfg["voice"])
            base += hashlib.sha1(wav.read_bytes() + txt.read_bytes()).hexdigest()[:12]
        return base

    def describe(self) -> str:
        who = {"clone": self.cfg.get("voice", ""), "custom": self.cfg.get("speaker", ""), "design": "design"}[self.mode]
        return f"qwen3-tts:{self.mode}:{Path(str(who)).name}"

    def _load(self):
        if self._model is None:
            self._model = load_model(
                self.model_id, self.cfg.get("device", "auto"), self.cfg.get("dtype", "auto"), self.cfg.get("attn", "auto")
            )
            if self.mode == "clone":
                wav, txt = voice_paths(self.cfg["voice"])
                if not wav.exists():
                    raise FileNotFoundError(
                        f"narrator reference {wav} is missing — run `factory voice-design` then `factory voice-pick`"
                    )
                self._prompt = self._model.create_voice_clone_prompt(
                    ref_audio=str(wav), ref_text=txt.read_text(encoding="utf-8").strip()
                )
        return self._model

    def synth(self, text: str, seed: int = 0) -> np.ndarray:
        model = self._load()
        seed_all(int(self.cfg.get("seed", 1234)) + 7919 * seed)
        lang = self.cfg.get("language", "Russian")
        extra = dict(self.cfg.get("generate") or {})
        if self.mode == "clone":
            wavs, sr = model.generate_voice_clone(text=text, language=lang, voice_clone_prompt=self._prompt, **extra)
        elif self.mode == "custom":
            kw = {"instruct": self.cfg["instruct"]} if self.cfg.get("instruct") else {}
            wavs, sr = model.generate_custom_voice(text=text, language=lang, speaker=self.cfg.get("speaker", "Ryan"), **kw, **extra)
        else:
            wavs, sr = model.generate_voice_design(text=text, language=lang, instruct=self.cfg["instruct"], **extra)
        self.sample_rate = int(sr)
        w = wavs[0]
        if hasattr(w, "detach"):
            w = w.detach().float().cpu().numpy()
        return np.asarray(w, dtype=np.float32).reshape(-1)

    def unload(self) -> None:
        self._model = self._prompt = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
