"""CosyVoice 3 (Fun-CosyVoice3-0.5B, Apache-2.0; Russian supported): zero-shot clone of the narrator.

Runs in its own venv (tools/cosyvoice_setup.sh) through cosyvoice_worker.py, because its pinned
dependencies clash with ours. Same reference clip as the Qwen clone mode (tts.qwen.voice), so the
two engines can be compared on one voice.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import TTSEngine
from .qwen import voice_paths


class CosyVoiceTTS(TTSEngine):
    name = "cosyvoice"
    default_stress = "strip"

    def __init__(self, cfg: dict, cache_dir: Path):
        super().__init__(cfg, cache_dir)
        self.sample_rate = 24000
        self._proc: subprocess.Popen | None = None

    def voice_id(self) -> str:
        wav, txt = voice_paths(self.cfg["voice"])
        return super().voice_id() + hashlib.sha1(wav.read_bytes() + txt.read_bytes()).hexdigest()[:12]

    def describe(self) -> str:
        return f"cosyvoice3:clone:{Path(self.cfg['voice']).name}"

    def _start(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            env = os.environ
            py = self.cfg.get("python") or env.get("COSYVOICE_PYTHON")
            repo = self.cfg.get("repo") or env.get("COSYVOICE_DIR")
            model_dir = self.cfg.get("model_dir") or env.get("COSYVOICE_MODEL_DIR")
            if not (py and repo and model_dir):
                raise RuntimeError("CosyVoice is not set up: run tools/cosyvoice_setup.sh and re-source factory.env")
            wav, txt = voice_paths(self.cfg["voice"])
            worker = Path(__file__).with_name("cosyvoice_worker.py")
            print(f"[cosyvoice] starting worker ({model_dir})", flush=True)
            self._proc = subprocess.Popen(
                [py, "-u", str(worker), "--repo", repo, "--model-dir", model_dir, "--prompt-wav", str(wav),
                 "--prompt-text-file", str(txt), "--text-frontend", str(int(bool(self.cfg.get("text_frontend", False))))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            )
            hello = json.loads(self._proc.stdout.readline() or '{"ok": false, "error": "worker died on start"}')
            if not hello.get("ok"):
                raise RuntimeError(f"cosyvoice worker: {hello}")
            self.sample_rate = int(hello["sr"])
        return self._proc

    def synth(self, text: str, seed: int = 0) -> np.ndarray:
        proc = self._start()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "c.wav"
            req = {"text": text, "out": str(out), "seed": int(self.cfg.get("seed", 1234)) + 7919 * seed,
                   "speed": float(self.cfg.get("speed", 1.0))}
            proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            rep = json.loads(proc.stdout.readline() or '{"ok": false, "error": "worker died"}')
            if not rep.get("ok"):
                raise RuntimeError(f"cosyvoice: {rep.get('error')}")
            a, sr = sf.read(out, dtype="float32")
        self.sample_rate = sr
        return a

    def unload(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.terminate()
        self._proc = None
