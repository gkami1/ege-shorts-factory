"""build(script) → out/<task_id>.mp4 + meta.json.

Stages (each writes into work/<task_id>/, so a failed run can be inspected):
  1. load + validate script against its bank task
  2. TTS: one clip per beat
  3. timeline: beat windows from clip durations; voice track mix
  4. word alignment → captions.ass
  5. Manim render (silent, CPU)
  6. ffmpeg: burn captions, mux voice (+music), loudnorm
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import soundfile as sf

from . import timeline as tl
from .bank import load_bank
from .captions.align import Aligner
from .captions.ass import write_ass
from .models import VideoScript, script_errors
from .render.style import palette
from .speech.normalize import to_speech
from .speech.tts import get_engine


class BuildError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[factory] {msg}", flush=True)


def load_script(path: Path) -> VideoScript:
    return VideoScript.model_validate_json(path.read_text(encoding="utf-8"))


def build(script_path: Path, cfg: dict, draft: bool = False, out_dir: Path | None = None) -> Path:
    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    paths = cfg["paths"]
    script = load_script(script_path)
    bank = load_bank(paths["tasks"])
    if script.task_id not in bank:
        raise BuildError(f"task {script.task_id} not in bank {paths['tasks']}")
    task = bank[script.task_id]
    if errs := script_errors(script, task):
        raise BuildError("script failed validation:\n  " + "\n  ".join(errs))

    work = Path(paths["work"]) / task.id
    work.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("FACTORY_CACHE") or Path(paths["work"]) / "_cache")  # models + TTS clip cache

    # 2. TTS
    t0 = time.perf_counter()
    segs = tl.beats(script)
    tts = get_engine(cfg["tts"], cache)
    for i, s in enumerate(segs):
        s.spoken = to_speech(s.text)
        clip = tts.synth_to_file(s.spoken, work / "audio" / f"{i:02d}_{s.key}.wav")
        s.audio = str(clip)
        s.audio_dur = sf.info(clip).duration
    timings["tts"] = time.perf_counter() - t0
    log(f"tts: {len(segs)} clips, {sum(s.audio_dur for s in segs):.1f}s of speech ({timings['tts']:.1f}s)")

    # 3. timeline + voice track
    fps = 15 if draft else cfg["video"]["fps"]
    tl.layout(segs, cfg["timing"], fps)
    voice = tl.mix_voice(segs, work / "voice.wav", tts.sample_rate)

    # 4. captions
    t0 = time.perf_counter()
    aligner = Aligner(cfg["captions"])
    for s in segs:
        if s.key == "outro":  # the CTA card already shows the text
            continue
        s.words = [
            {**w, "start": w["start"] + s.voice_start, "end": w["end"] + s.voice_start}
            for w in aligner.align(s.audio, s.text)
        ]
    colors = palette(cfg["render"]["palette"])
    v = cfg["video"]
    ass = write_ass(segs, cfg["captions"], colors, v["width"], v["height"], work / "captions.ass")
    tl.save(segs, work / "timeline.json")
    timings["align"] = time.perf_counter() - t0
    log(f"captions: aligner={aligner.mode} ({timings['align']:.1f}s)")

    # 5. render
    t0 = time.perf_counter()
    from .render.run import render_scene

    silent = render_scene(script, task, segs, cfg, work, draft=draft)
    timings["render"] = time.perf_counter() - t0
    log(f"render: {segs[-1].end:.1f}s video in {timings['render']:.1f}s")

    # 6. assemble
    t0 = time.perf_counter()
    from .assemble.ffmpeg import assemble, probe_duration

    out_dir = Path(out_dir or paths["out"])
    suffix = "_draft" if draft else ""
    out = assemble(silent, voice, ass, out_dir / f"{task.id}{suffix}.mp4", cfg)
    timings["assemble"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_all

    meta = {
        "task_id": task.id,
        "file": out.name,
        "title": script.meta.title,
        "description": (script.meta.description + "\n\n" + " ".join(script.meta.hashtags)).strip(),
        "hashtags": script.meta.hashtags,
        "answer": script.answer.value,
        "duration_s": round(probe_duration(out), 2),
        "style": {"palette": cfg["render"]["palette"], "voice": f"{cfg['tts']['engine']}:{cfg['tts'].get(cfg['tts']['engine'], {}).get('speaker', '')}"},
        "timings_s": {k: round(x, 1) for k, x in timings.items()},
        "draft": draft,
    }
    (out_dir / f"{task.id}{suffix}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done: {out} ({meta['duration_s']}s, total {timings['total']:.0f}s)")
    return out
