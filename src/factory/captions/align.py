"""Word timestamps for each clip.

We know the exact text of every clip, so alignment = map our words onto
faster-whisper's word timestamps (difflib on normalized tokens); words whisper
missed are interpolated by syllable (vowel) count. Fallback without whisper:
spread words over the detected speech region proportionally to syllables.
"""

from __future__ import annotations

import difflib
import re

import numpy as np
import soundfile as sf

from ..speech.normalize import display_words

_VOWELS = set("аеёиоуыэюяaeiouy")


def _syll(w: str) -> int:
    return max(1, sum(ch in _VOWELS for ch in w.lower()))


def _norm(w: str) -> str:
    return re.sub(r"[^\wё]", "", w.lower().replace("ё", "е"))


def speech_region(path: str) -> tuple[float, float]:
    a, sr = sf.read(path, dtype="float32")
    hop = int(sr * 0.02)
    n = max(1, len(a) // hop)
    rms = np.sqrt(np.mean(a[: n * hop].reshape(n, hop) ** 2, axis=1))
    on = np.where(rms > rms.max() * 0.06)[0]
    if len(on) == 0:
        return 0.0, len(a) / sr
    return on[0] * hop / sr, (on[-1] + 1) * hop / sr


def _load_16k(path: str) -> np.ndarray:
    """Mono float32 @16 kHz for whisper (avoids PyAV decoding, whose API drifts between versions)."""
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        n = int(round(len(a) * 16000 / sr))
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return a


def _spread(words: list[str], t0: float, t1: float) -> list[tuple[float, float]]:
    w = np.array([_syll(x) for x in words], float)
    edges = t0 + (t1 - t0) * np.concatenate([[0], np.cumsum(w) / w.sum()])
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]


class Aligner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model = None
        self.mode = cfg.get("aligner", "whisper")

    def _whisper(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            dev = self.cfg.get("whisper_device", "auto")
            if dev == "auto":
                try:
                    import torch

                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    dev = "cpu"
            self._model = WhisperModel(
                self.cfg.get("whisper_model", "small"), device=dev, compute_type="float16" if dev == "cuda" else "int8"
            )
        return self._model

    def align(self, path: str, text: str) -> list[dict]:
        """Return [{w, start, end}] relative to the clip start."""
        words = display_words(text)
        t0, t1 = speech_region(path)
        times: list[tuple[float, float] | None] = [None] * len(words)
        if self.mode == "whisper":
            try:
                times = self._whisper_times(path, words)
            except Exception as e:  # missing package / model download failure → degrade, don't die
                print(f"[align] whisper unavailable ({e!r}); using proportional timing")
                self.mode = "proportional"
        if all(t is None for t in times):
            times = _spread(words, t0, t1)
        times = _fill_gaps(words, times, t0, t1)
        return [{"w": w, "start": round(a, 3), "end": round(b, 3)} for w, (a, b) in zip(words, times)]

    def _whisper_times(self, path: str, words: list[str]):
        segs, _ = self._whisper().transcribe(
            _load_16k(path), language="ru", word_timestamps=True, initial_prompt=" ".join(words), beam_size=5, vad_filter=False
        )
        hyp = [(w.word, w.start, w.end) for s in segs for w in (s.words or [])]
        a, b = [_norm(w) for w in words], [_norm(h[0]) for h in hyp]
        times: list = [None] * len(words)
        for blk in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
            for k in range(blk.size):
                h = hyp[blk.b + k]
                times[blk.a + k] = (h[1], h[2])
        return times


def _fill_gaps(words, times, t0, t1):
    """Interpolate unmatched words between their matched neighbours; enforce monotonic order."""
    out = list(times)
    i, n = 0, len(out)
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < n and out[j] is None:
            j += 1
        lo = out[i - 1][1] if i > 0 else t0
        hi = out[j][0] if j < n else t1
        if hi <= lo:
            hi = lo + 0.12 * (j - i)
        out[i:j] = _spread(words[i:j], lo, hi)
        i = j
    prev = 0.0
    for k, (a, b) in enumerate(out):
        a = max(a, prev)
        b = max(b, a + 0.06)
        out[k] = (a, b)
        prev = a
    return out
