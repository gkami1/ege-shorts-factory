"""Narrator voice tools (GPU pod): design the voice once, then compare engines blind.

  factory voice-design  → work/voices/cand_XX.wav (+ the same voice reading test lines through the
                          production clone path) and work/voices/index.html for listening
  factory voice-pick N  → assets/voices/narrator_ru.{wav,txt,json} (committed: the channel's voice)
  factory tts-bakeoff   → work/bakeoff/: every variant reads the same lines; whisper CER, speed,
                          and a blind listening page (labels shuffled, key in key.json)
"""

from __future__ import annotations

import html
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import ROOT, _merge
from .normalize import to_speech
from .tts import get_engine
from .tts.base import COMMERCIAL_OK, _tidy

REF_TEXT = (
    "Смотри внимательно. Эту задачу можно решить за минуту, если заметить одну простую вещь: "
    "сумма в показателе — это произведение степеней. Дальше остаётся только посчитать."
)
INSTRUCTS = [
    "Male, about 30 years old, warm baritone, native Russian speaker. A calm, confident math tutor "
    "explaining a problem to a student. Clear articulation, moderate pace, friendly and engaging.",
    "Male, about 25 years old, bright baritone, native Russian speaker. An energetic narrator of short "
    "educational videos: lively, upbeat, fast but very clear delivery.",
    "Male, 35 to 40 years old, deep low voice, native Russian speaker. A documentary-style narrator: "
    "measured pace, authoritative, calm and trustworthy.",
]
TEST_LINES = [
    "Дано: два в степени икс равно десяти. А найти нужно два в степени икс плюс три.",
    "Скорость — это производная координаты по времени: две плюс шесть тэ.",
    "Ответ: восемьдесят!",
]


def _page(title: str, rows: list[tuple[str, list[str], str]], out: Path) -> None:
    body = "\n".join(
        f"<tr><td><b>{html.escape(name)}</b><br><small>{html.escape(note)}</small></td><td>"
        + "".join(f'<audio controls preload="none" src="{html.escape(a)}"></audio><br>' for a in audios)
        + "</td></tr>"
        for name, audios, note in rows
    )
    out.write_text(
        f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
        "<style>body{font:15px system-ui;margin:24px}td{padding:8px;border-bottom:1px solid #ddd;vertical-align:top}"
        "audio{width:360px}</style>"
        f"<h1>{html.escape(title)}</h1><table>{body}</table>",
        encoding="utf-8",
    )


def design(cfg: dict, n: int, out_dir: Path, instructs: list[str] | None = None, ref_text: str = REF_TEXT) -> Path:
    from .tts.qwen import MODELS, load_model, seed_all

    instructs = instructs or INSTRUCTS
    out_dir.mkdir(parents=True, exist_ok=True)
    qc = cfg["tts"]["qwen"]
    model = load_model(MODELS["design"], qc.get("device", "auto"), qc.get("dtype", "auto"), qc.get("attn", "auto"))
    cands = []
    for k in range(n):
        instruct = instructs[k % len(instructs)]
        seed = 1000 + k
        seed_all(seed)
        wavs, sr = model.generate_voice_design(text=ref_text, language="Russian", instruct=instruct)
        a = _tidy(np.asarray(wavs[0], np.float32), sr)
        p = out_dir / f"cand_{k:02d}.wav"
        sf.write(p, a, sr, subtype="PCM_16")
        meta = {"id": k, "instruct": instruct, "seed": seed, "ref_text": ref_text, "model": MODELS["design"], "sr": sr}
        (out_dir / f"cand_{k:02d}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        cands.append(meta)
        print(f"[voice-design] cand_{k:02d}: {len(a) / sr:.1f}s", flush=True)
    del model
    _free()

    # audition each candidate through the production path (Base model voice clone)
    clone = load_model(MODELS["clone"], qc.get("device", "auto"), qc.get("dtype", "auto"), qc.get("attn", "auto"))
    rows = []
    for c in cands:
        k = c["id"]
        prompt = clone.create_voice_clone_prompt(ref_audio=str(out_dir / f"cand_{k:02d}.wav"), ref_text=ref_text)
        audios = [f"cand_{k:02d}.wav"]
        for j, line in enumerate(TEST_LINES):
            seed_all(qc.get("seed", 1234))
            wavs, sr = clone.generate_voice_clone(text=to_speech(line).replace("+", ""), language="Russian", voice_clone_prompt=prompt)
            name = f"cand_{k:02d}_test{j}.wav"
            sf.write(out_dir / name, _tidy(np.asarray(wavs[0], np.float32), sr), sr, subtype="PCM_16")
            audios.append(name)
        rows.append((f"cand_{k:02d}", audios, c["instruct"]))
    _page("Narrator candidates (first: designed reference, then cloned test lines)", rows, out_dir / "index.html")
    return out_dir / "index.html"


def pick(k: int, src_dir: Path, voice: str) -> Path:
    dst = Path(voice)
    dst = dst if dst.is_absolute() else ROOT / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src_dir / f"cand_{k:02d}.json").read_text(encoding="utf-8"))
    shutil.copy(src_dir / f"cand_{k:02d}.wav", dst.with_suffix(".wav"))
    dst.with_suffix(".txt").write_text(meta["ref_text"] + "\n", encoding="utf-8")
    meta["license"] = "Synthetic voice generated with Qwen3-TTS VoiceDesign (Apache-2.0); no real person's voice."
    dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return dst.with_suffix(".wav")


def _free() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def bakeoff(cfg: dict, spec: dict, out_dir: Path, lines: list[str]) -> Path:
    from ..captions.align import Aligner
    from . import qa

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "_cache"
    results = []
    for v in spec["variants"]:
        name = v["name"]
        tcfg = _merge(cfg["tts"], v["tts"])
        try:
            eng = get_engine(tcfg, cache)
        except Exception as e:
            print(f"[bakeoff] {name}: cannot create engine ({e!r}), skipped", flush=True)
            continue
        vdir = out_dir / name
        files, speech_s, wall = [], 0.0, 0.0
        try:
            for i, line in enumerate(lines):
                t = time.perf_counter()
                p = eng.synth_to_file(eng.prepare(to_speech(line)), vdir / f"{i:02d}.wav")
                wall += time.perf_counter() - t
                speech_s += sf.info(p).duration
                files.append(p)
        except Exception as e:
            print(f"[bakeoff] {name}: failed ({e!r}), skipped", flush=True)
            eng.unload()
            continue
        eng.unload()
        _free()
        results.append({"name": name, "engine": eng.describe(), "commercial_ok": eng.name in COMMERCIAL_OK,
                        "files": files, "rtf": wall / max(speech_s, 1e-6), "speech_s": speech_s})
        print(f"[bakeoff] {name}: {speech_s:.0f}s of speech in {wall:.0f}s", flush=True)

    aligner = Aligner({**cfg["captions"], "aligner": "whisper", "whisper_model": spec.get("whisper_model", "large-v3")})
    for r in results:
        scores = []
        for line, f in zip(lines, r["files"]):
            hyp = aligner.transcribe(str(f)) or []
            scores.append(qa.score(to_speech(line), [h[0] for h in hyp]))
        r["cer_mean"] = float(np.mean(scores))
        r["cer_max"] = float(np.max(scores))
        r["bad_lines"] = int(sum(s > 0.1 for s in scores))

    rng = random.Random(spec.get("blind_seed", 7))
    labels = [chr(ord("A") + i) for i in range(len(results))]
    rng.shuffle(labels)
    key = {lab: r["name"] for lab, r in zip(labels, results)}
    (out_dir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = []
    for i, line in enumerate(lines):
        auds = sorted(((lab, r["files"][i]) for lab, r in zip(labels, results)), key=lambda x: x[0])
        rows.append((f"{i:02d}. {line}", [str(p.relative_to(out_dir)).replace("\\", "/") for _, p in auds],
                     "  ".join(lab for lab, _ in auds)))
    _page("TTS bake-off (blind: voices in order " + " ".join(sorted(labels)) + "; key in key.json)", rows, out_dir / "index.html")

    md = ["| variant | engine | commercial | CER mean | CER max | lines > 0.1 | RTF |", "|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: r["cer_mean"]):
        md.append(f"| {r['name']} | {r['engine']} | {'yes' if r['commercial_ok'] else 'NO'} | {r['cer_mean']:.3f} | "
                  f"{r['cer_max']:.2f} | {r['bad_lines']} | {r['rtf']:.2f} |")
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return out_dir / "report.md"
