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
from .speech import qa
from .speech.normalize import to_speech
from .speech.tts import get_engine
from .speech.tts.base import COMMERCIAL_OK


class BuildError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[factory] {msg}", flush=True)


def _worst_cer(qa_report: list[dict]) -> float | None:
    checked = [r["cer"] for r in qa_report if r["cer"] is not None]
    return max(checked) if checked else None


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
    from .verify.gate import verify

    report = verify(script, task)
    for w in report.warnings:
        log(f"verify: {w}")
    if not report.ok:
        raise BuildError("verification gate failed:\n  " + "\n  ".join(report.errors))

    work = Path(paths["work"]) / task.id
    work.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("FACTORY_CACHE") or Path(paths["work"]) / "_cache")  # models + TTS clip cache

    # 2. TTS (+ QA: whisper must hear what the script says, else re-take with another seed)
    t0 = time.perf_counter()
    segs = tl.beats(script)
    tts = get_engine(cfg["tts"], cache)
    if tts.name not in COMMERCIAL_OK and not draft:
        log(f"WARNING: {tts.describe()} is not licensed for commercial use — do not publish this video")
    aligner = Aligner(cfg["captions"])
    qa_cfg = cfg["tts"].get("qa", {})
    qa_report = []
    for i, s in enumerate(segs):
        s.spoken = tts.prepare(to_speech(s.text))
        clip_path = work / "audio" / f"{i:02d}_{s.key}.wav"
        best = None
        retries = 0 if tts.deterministic else int(qa_cfg.get("retries", 2))
        for attempt in range(1 + retries):
            clip = tts.synth_to_file(s.spoken, clip_path, attempt=attempt)
            aligner.forget(str(clip))
            hyp = aligner.transcribe(str(clip)) if qa_cfg.get("enabled", True) else None
            score = qa.score(s.spoken, [h[0] for h in hyp]) if hyp is not None else 0.0
            if best is None or score < best[0]:
                best = (score, attempt, clip.read_bytes(), hyp)
            if score <= float(qa_cfg.get("max_cer", 0.1)):
                break
            log(f"tts qa: {s.key} take {attempt} CER {score:.2f} → retry")
        score, attempt, data, hyp = best
        clip_path.write_bytes(data)
        aligner.forget(str(clip_path))
        if hyp is not None:
            aligner._hyp[str(clip_path)] = hyp
        # cer None = not checked (QA off or whisper unavailable), never a fake 0.0
        qa_report.append({"beat": s.key, "cer": round(score, 3) if hyp is not None else None, "take": attempt,
                          "heard": " ".join(h[0] for h in hyp or [])})
        if score > float(qa_cfg.get("max_cer", 0.1)):
            log(f"tts qa: {s.key} still CER {score:.2f} after retries — check {clip_path.name}")
        s.audio = str(clip_path)
        s.audio_dur = sf.info(clip_path).duration
    timings["tts"] = time.perf_counter() - t0
    (work / "tts_qa.json").write_text(json.dumps(qa_report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"tts: {tts.describe()}, {len(segs)} clips, {sum(s.audio_dur for s in segs):.1f}s of speech, "
        + (f"worst CER {worst_cer:.2f}" if (worst_cer := _worst_cer(qa_report)) is not None
           else "TTS QA NOT RUN (whisper unavailable or qa disabled)") + f" ({timings['tts']:.1f}s)")

    # 3. timeline + voice track
    fps = 15 if draft else cfg["video"]["fps"]
    tl.layout(segs, cfg["timing"], fps)
    voice = tl.mix_voice(segs, work / "voice.wav", tts.sample_rate)

    # 4. captions
    t0 = time.perf_counter()
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
        "style": {"palette": cfg["render"]["palette"], "voice": tts.describe()},
        "commercial_ok": tts.name in COMMERCIAL_OK,
        "tts_worst_cer": _worst_cer(qa_report),
        "timings_s": {k: round(x, 1) for k, x in timings.items()},
        "draft": draft,
    }
    (out_dir / f"{task.id}{suffix}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done: {out} ({meta['duration_s']}s, total {timings['total']:.0f}s)")
    return out
