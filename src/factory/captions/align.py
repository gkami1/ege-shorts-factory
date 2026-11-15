"""Word timestamps for each clip.

We know the exact text of every clip, so alignment = map our caption tokens onto
faster-whisper's word timestamps (difflib on canonical forms, where numbers are digits on
both sides); tokens whisper missed are interpolated by syllable count of their spoken words.
Fallback without whisper: spread tokens over the detected speech region by syllables.

The same transcript feeds the TTS QA (pipeline retries a clip whose WER is too high).
"""

from __future__ import annotations

import difflib
import re

import numpy as np
import soundfile as sf

from ..speech.normalize import LETTERS, Token, tokenize
from ..speech.qa import merge_number_words

_VOWELS = set("аеёиоуыэюяaeiouy")


def _syll(w: str) -> int:
    return max(1, sum(ch in _VOWELS for ch in w.lower()))


def _norm(w: str) -> str:
    w = re.sub(r"[^\w,/]", "", w.lower().replace("ё", "е").replace("+", "")).strip(",")
    return LETTERS.get(w, w) if re.fullmatch(r"[a-z]", w) else "икс" if w == "х" else w


def speech_region(path: str) -> tuple[float, float]:
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
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
        self._hyp: dict[str, list[tuple[str, float, float]]] = {}

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

    def transcribe(self, path: str) -> list[tuple[str, float, float]] | None:
        """Whisper words with times (no prompt, so it can't parrot our text). None if unavailable."""
        if self.mode != "whisper":
            return None
        if path not in self._hyp:
            try:
                segs, _ = self._whisper().transcribe(
                    _load_16k(path), language="ru", word_timestamps=True, beam_size=5, vad_filter=False,
                    condition_on_previous_text=False,
                )
                self._hyp[path] = [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]
            except Exception as e:  # missing package / model download failure → degrade, don't die
                print(f"[align] whisper unavailable ({e!r}); using proportional timing")
                self.mode = "proportional"
                return None
        return self._hyp[path]

    def forget(self, path: str) -> None:
        self._hyp.pop(path, None)

    def align(self, path: str, text: str | list[Token]) -> list[dict]:
        """Return [{w, start, end}] (caption tokens) relative to the clip start."""
        toks = tokenize(text) if isinstance(text, str) else text
        toks = [t for t in toks if t.display and t.words]
        disp = [t.display for t in toks]
        spoken = [" ".join(t.words) for t in toks]
        t0, t1 = speech_region(path)
        times: list[tuple[float, float] | None] = [None] * len(toks)
        hyp = self.transcribe(path)
        if hyp:
            hyp = merge_number_words(hyp)
            a, b = [_norm(d) for d in disp], [_norm(h[0]) for h in hyp]
            for blk in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
                for k in range(blk.size):
                    h = hyp[blk.b + k]
                    times[blk.a + k] = (h[1], h[2])
        if all(t is None for t in times):
            times = _spread(spoken, t0, t1)
        times = _fill_gaps(spoken, times, t0, t1)
        return [{"w": w, "start": round(a, 3), "end": round(b, 3)} for w, (a, b) in zip(disp, times)]


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
