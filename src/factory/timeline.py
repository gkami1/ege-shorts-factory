"""Beat timeline: one voice clip per visual beat, so every animation knows its time window."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .models import VideoScript


@dataclass
class Segment:
    key: str              # hook | given | target | step_0.. | substitute | answer | outro
    text: str             # narration as written in the script (caption source)
    spoken: str = ""      # normalized text sent to TTS
    audio: str = ""       # clip path
    audio_dur: float = 0.0
    start: float = 0.0    # segment start on the global timeline
    dur: float = 0.0
    voice_start: float = 0.0  # absolute time the clip starts
    words: list[dict] = field(default_factory=list)  # [{w, start, end}] absolute, filled by the aligner

    @property
    def end(self) -> float:
        return self.start + self.dur


def beats(script: VideoScript) -> list[Segment]:
    segs = [
        Segment("hook", script.hook.narration),
        Segment("given", script.read.given_narration),
        Segment("target", script.read.target_narration),
    ]
    segs += [Segment(f"step_{i}", s.narration) for i, s in enumerate(script.steps)]
    segs += [
        Segment("substitute", script.substitute.narration),
        Segment("answer", script.answer.narration),
        Segment("outro", script.outro.narration),
    ]
    return segs


def layout(segs: list[Segment], timing: dict, fps: int) -> list[Segment]:
    """Assign start/dur on a frame grid. dur = pad_before + clip + pad_after (>= min_segment)."""
    t = 0.0
    for s in segs:
        d = timing["pad_before"] + s.audio_dur + timing["pad_after"]
        if s.key == "outro":
            d += timing.get("outro_hold", 0.0)
        d = max(d, timing["min_segment"])
        d = math.ceil(d * fps) / fps
        s.start, s.dur = round(t, 6), d
        s.voice_start = s.start + timing["pad_before"]
        t += d
    return segs


def mix_voice(segs: list[Segment], out: Path, sample_rate: int) -> Path:
    total = segs[-1].end
    buf = np.zeros(int(math.ceil(total * sample_rate)) + 1, dtype=np.float32)
    for s in segs:
        if not s.audio:
            continue
        a, sr = sf.read(s.audio, dtype="float32")
        assert sr == sample_rate, f"{s.audio}: {sr} != {sample_rate}"
        i = int(round(s.voice_start * sample_rate))
        buf[i : i + len(a)] += a[: len(buf) - i]
    sf.write(out, buf, sample_rate, subtype="PCM_16")
    return out


def save(segs: list[Segment], path: Path) -> None:
    path.write_text(json.dumps([asdict(s) for s in segs], ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path) -> list[Segment]:
    return [Segment(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
