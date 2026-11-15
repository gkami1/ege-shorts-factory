"""Fake reading sessions for the fine-tuning self-test (pipeline mechanics only, not quality).

    python tools/make_fake_sessions.py OUT_DIR [N_LINES=60]

Reads the first N script lines with the current CosyVoice clone and writes two "sessions" with
~1 s pauses, one false start and one reread — what a real recording looks like to voice-dataset.
"""

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from factory.config import load_config  # noqa: E402
from factory.speech.dataset import load_script  # noqa: E402
from factory.speech.tts.cosyvoice import CosyVoiceTTS  # noqa: E402

out = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
out.mkdir(parents=True, exist_ok=True)
cfg = load_config()
eng = CosyVoiceTTS(cfg["tts"]["cosyvoice"], out / "_cache")
script = load_script(Path("assets/voices/script/reading_script.txt"))[:n]


def clip(text: str) -> np.ndarray:
    p = out / "_tmp.wav"
    eng.synth_to_file(text, p)
    a, sr = sf.read(p, dtype="float32")
    eng.sample_rate = sr
    return a


pause = lambda s: np.zeros(int(s * eng.sample_rate), np.float32)  # noqa: E731
half = n // 2
for k, part in enumerate((script[:half], script[half:])):
    buf = [pause(0.8)]
    for i, (_, text) in enumerate(part):
        a = clip(text)
        if i == 3:  # false start, then the full reread
            buf += [a[: len(a) // 3], pause(1.2)]
        buf += [a, pause(1.0)]
        print(k, i, flush=True)
    sf.write(out / f"session_{k + 1}.wav", np.concatenate(buf), eng.sample_rate, subtype="PCM_16")
eng.unload()
os._exit(0)
