"""`factory` CLI."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import typer

from .config import ROOT, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="ЕГЭ shorts factory")


@app.command()
def build(
    task_id: Optional[str] = typer.Option(None, "--task-id", help="Build from video_scripts/<id>.json (M3: generate with the LLM)"),
    script: Optional[Path] = typer.Option(None, "--script", help="Path to a video script JSON"),
    config: Optional[Path] = typer.Option(None, "--config"),
    draft: bool = typer.Option(False, "--draft", help="540x960@15fps for fast look-dev"),
    out: Optional[Path] = typer.Option(None, "--out"),
):
    """Render one short."""
    from .pipeline import build as run

    cfg = load_config(config)
    if script is None:
        if not task_id:
            raise typer.BadParameter("pass --task-id or --script")
        script = cfg["paths"]["scripts"] / f"{task_id}.json"
    code = 0
    try:
        run(script, cfg, draft=draft, out_dir=out)
    except Exception:
        traceback.print_exc()
        code = 1
    _hard_exit(code)


def _hard_exit(code: int) -> None:
    """ctranslate2 (faster-whisper) worker threads can keep the interpreter alive; exit for real."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


@app.command()
def batch(scripts: list[Path], config: Optional[Path] = typer.Option(None, "--config"), draft: bool = False):
    """Render many scripts; failures are reported and skipped (full batch mode + report in M4/M5)."""
    from .pipeline import build as run

    cfg = load_config(config)
    failed = []
    for s in scripts:
        try:
            run(s, cfg, draft=draft)
        except Exception as e:  # keep going
            failed.append((s, repr(e)))
            typer.echo(f"FAILED {s}: {e}", err=True)
    typer.echo(f"{len(scripts) - len(failed)}/{len(scripts)} built")
    _hard_exit(1 if failed else 0)


@app.command()
def validate(config: Optional[Path] = typer.Option(None, "--config")):
    """Validate the task bank and every hand-written script against it."""
    from .bank import load_bank
    from .models import script_errors
    from .pipeline import load_script

    cfg = load_config(config)
    bank = load_bank(cfg["paths"]["tasks"])
    typer.echo(f"bank: {len(bank)} tasks OK")
    bad = 0
    for p in sorted(Path(cfg["paths"]["scripts"]).glob("*.json")):
        s = load_script(p)
        errs = script_errors(s, bank[s.task_id]) if s.task_id in bank else [f"unknown task {s.task_id}"]
        bad += bool(errs)
        typer.echo(f"{p.name}: {'OK' if not errs else '; '.join(errs)}")
    raise typer.Exit(1 if bad else 0)


@app.command()
def verify(
    targets: Optional[list[str]] = typer.Argument(None, help="Task ids or script paths (default: all of video_scripts/)"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strict: bool = typer.Option(False, "--strict", help="Missing machine forms are errors (LLM output)"),
    to_review: bool = typer.Option(False, "--to-review", help="Copy failing scripts to review/<id>/"),
):
    """Run the verification gate (maths, answer, units, narration) on video scripts."""
    from .bank import load_bank
    from .pipeline import load_script
    from .verify.gate import verify as run
    from .verify.review import send_to_review

    cfg = load_config(config)
    bank = load_bank(cfg["paths"]["tasks"])
    paths = []
    for t in targets or sorted(str(p) for p in Path(cfg["paths"]["scripts"]).glob("*.json")):
        p = Path(t)
        paths.append(p if p.suffix == ".json" else cfg["paths"]["scripts"] / f"{t}.json")
    failed = 0
    for p in paths:
        s = load_script(p)
        if s.task_id not in bank:
            typer.echo(f"{p.name}: unknown task {s.task_id}")
            failed += 1
            continue
        rep = run(s, bank[s.task_id], strict=strict)
        typer.echo(rep.text())
        if not rep.ok:
            failed += 1
            if to_review:
                d = send_to_review(p.read_text(encoding="utf-8"), rep, cfg["paths"]["review"])
                typer.echo(f"  → {d}")
    typer.echo(f"{len(paths) - failed}/{len(paths)} passed")
    raise typer.Exit(1 if failed else 0)


def _pick_tasks(bank: dict, ids: list[str] | None, kim: Optional[str]) -> list:
    tasks = [bank[i] for i in ids] if ids else list(bank.values())
    if kim:  # e.g. "math_profile:7"
        subj, num = kim.split(":")
        tasks = [t for t in tasks if t.subject == subj and t.kim_number == int(num)]
    return tasks


@app.command()
def write(
    task_ids: Optional[list[str]] = typer.Argument(None, help="Task ids (default: every task without a script)"),
    config: Optional[Path] = typer.Option(None, "--config"),
    kim: Optional[str] = typer.Option(None, "--kim", help="Filter, e.g. math_profile:7"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing scripts"),
    jobs: int = typer.Option(4, "--jobs", help="Tasks in flight at once (the server batches them)"),
):
    """LLM: solve → write → verify. Passing scripts go to video_scripts/, the rest to review/."""
    from concurrent.futures import ThreadPoolExecutor

    from .bank import load_bank
    from .llm.scriptwriter import ScriptWriter

    cfg = load_config(config)
    sw = ScriptWriter(cfg)
    if not sw.client.ready():
        typer.echo(f"LLM server not reachable at {cfg['llm']['base_url']} (start it: bash tools/llm_serve.sh)")
        raise typer.Exit(2)
    results = []
    tasks = _pick_tasks(load_bank(cfg["paths"]["tasks"]), task_ids, kim)
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        for t, o in pool.map(lambda t: (t, sw.write(t, force=force)), tasks):
            results.append(o.__dict__)
            typer.echo(f"{t.id}: {o.status} ({o.seconds:.0f}s, answers {o.answers}, rounds {o.rounds})"
                       + (f" → {o.path}" if o.path else "") + "".join(f"\n    ✗ {e}" for e in o.errors[:5]))
    summary = {s: sum(r["status"] == s for r in results) for s in ("written", "review", "unsolved", "exists", "error")}
    typer.echo(f"summary: {summary}")
    rep = cfg["paths"]["work"] / "llm" / f"write_{time_stamp()}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({"model": cfg["llm"]["model"], "summary": summary, "results": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    _hard_exit(0)


@app.command("llm-eval")
def llm_eval(
    config: Optional[Path] = typer.Option(None, "--config"),
    n: int = typer.Option(5, "--n", help="Samples per task"),
    jobs: int = typer.Option(4, "--jobs", help="Tasks in flight at once (the server batches them)"),
):
    """Solve every bank task N times → accuracy, agreement and speed of the served model."""
    import time as _t
    from concurrent.futures import ThreadPoolExecutor

    from .bank import load_bank
    from .llm.scriptwriter import ScriptWriter
    from .models import answers_equal
    from .verify.gate import agreement

    cfg = load_config(config)
    sw = ScriptWriter(cfg)
    if not sw.client.ready():
        typer.echo(f"LLM server not reachable at {cfg['llm']['base_url']}")
        raise typer.Exit(2)
    rows, t_all = [], _t.perf_counter()

    def solve(t):
        t0 = _t.perf_counter()
        answers, _ = sw.solve(t, n)
        return t, answers, _t.perf_counter() - t0

    with ThreadPoolExecutor(max(1, jobs)) as pool:
        for t, answers, sec in pool.map(solve, load_bank(cfg["paths"]["tasks"]).values()):
            ok, maj, counts = agreement(answers, t.answer, float(cfg["llm"].get("min_agreement", 0.6)))
            acc = sum(bool(a) and answers_equal(a, t.answer) for a in answers) / max(1, len(answers))
            rows.append({"task": t.id, "accuracy": acc, "majority_ok": ok, "counts": counts, "s": round(sec)})
            typer.echo(f"{t.id}: {acc:.0%} correct, majority {'OK' if ok else 'WRONG'} {counts} ({rows[-1]['s']}s)")
    mean = sum(r["accuracy"] for r in rows) / max(1, len(rows))
    maj = sum(r["majority_ok"] for r in rows)
    typer.echo(f"{cfg['llm']['model']}: sample accuracy {mean:.0%}, majority correct {maj}/{len(rows)}, "
               f"{_t.perf_counter() - t_all:.0f}s total")
    out = cfg["paths"]["work"] / "llm" / f"eval_{cfg['llm']['model'].replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": cfg["llm"]["model"], "samples": n, "jobs": jobs, "accuracy": mean, "majority_ok": maj,
                               "total_s": round(_t.perf_counter() - t_all), "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    _hard_exit(0)


def time_stamp() -> str:
    import time as _t

    return _t.strftime("%Y%m%d_%H%M%S")


@app.command("review-accept")
def review_accept(task_id: str, config: Optional[Path] = typer.Option(None, "--config")):
    """Re-verify an edited script in review/<id>/ and move it to video_scripts/ if it passes."""
    from .bank import load_bank
    from .pipeline import load_script
    from .verify.gate import verify as run
    from .verify.review import accept

    cfg = load_config(config)
    src = cfg["paths"]["review"] / task_id / "script.json"
    rep = run(load_script(src), load_bank(cfg["paths"]["tasks"])[task_id], strict=True)
    typer.echo(rep.text())
    if not rep.ok:
        raise typer.Exit(1)
    typer.echo(f"accepted → {accept(task_id, cfg['paths']['review'], cfg['paths']['scripts'])}")


@app.command()
def preview(task_id: str, config: Optional[Path] = typer.Option(None, "--config"), draft: bool = True, at: float = 0.8):
    """Contact sheet: one frame per beat (at `at` of each beat) → work/<id>/preview.png."""
    import subprocess

    from PIL import Image

    from . import timeline as tl

    cfg = load_config(config)
    work = cfg["paths"]["work"] / task_id
    video = cfg["paths"]["out"] / f"{task_id}{'_draft' if draft else ''}.mp4"
    segs = tl.load(work / "timeline.json")
    tiles = []
    for i, s in enumerate(segs):
        png = work / "frames" / f"{i:02d}_{s.key}.png"
        png.parent.mkdir(exist_ok=True)
        t = s.start + at * s.dur
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=360:640", str(png)], check=True)
        tiles.append(Image.open(png))
    cols = 5
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * 360, rows * 640), "white")
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * 360, (i // cols) * 640))
    sheet.save(work / "preview.png")
    typer.echo(str(work / "preview.png"))


@app.command("voice-design")
def voice_design(
    n: int = typer.Option(9, "--n", help="Number of candidate voices"),
    config: Optional[Path] = typer.Option(None, "--config"),
    instruct: Optional[list[str]] = typer.Option(None, "--instruct", help="Voice description(s); default: 3 built-in"),
):
    """GPU: design narrator candidates with Qwen3-TTS VoiceDesign → work/voices/index.html."""
    from .speech.voices import design

    cfg = load_config(config)
    page = design(cfg, n, cfg["paths"]["work"] / "voices", instructs=instruct or None)
    typer.echo(f"listen: {page}\nthen: factory voice-pick <N>")
    _hard_exit(0)


@app.command("voice-pick")
def voice_pick(k: int, config: Optional[Path] = typer.Option(None, "--config")):
    """Make candidate N the channel's narrator (assets/voices/narrator_ru.*; commit it)."""
    from .speech.voices import pick

    cfg = load_config(config)
    typer.echo(str(pick(k, cfg["paths"]["work"] / "voices", cfg["tts"]["qwen"]["voice"])))


@app.command("voice-import")
def voice_import(
    recording: Path,
    text_file: Optional[Path] = typer.Option(None, "--text-file", help="Exact transcript (default: the recording script)"),
    config: Optional[Path] = typer.Option(None, "--config"),
):
    """Your own recording (any audio format, 10–25 s) → the narrator reference for voice cloning."""
    from .speech.voices import RECORD_TEXT, import_recording

    cfg = load_config(config)
    text = text_file.read_text(encoding="utf-8") if text_file else RECORD_TEXT
    wav, cer = import_recording(recording, text, cfg["tts"]["qwen"]["voice"], cfg)
    typer.echo(f"{wav}")
    if cer is not None:
        typer.echo(f"transcript check: CER {cer:.2f} " + ("OK" if cer <= 0.1 else
                   "— the text doesn't match what was said; fix it (see the .json 'heard' field) and re-import"))
    _hard_exit(0)


@app.command("voice-dataset")
def voice_dataset(
    sessions: list[Path],
    out: Path = typer.Option(Path("work/voice_dataset"), "--out"),
    script: Path = typer.Option(Path("assets/voices/script/reading_script.txt"), "--script"),
    config: Optional[Path] = typer.Option(None, "--config"),
    max_cer: float = typer.Option(0.12, "--max-cer"),
):
    """Recorded reading sessions (in script order) → per-line clips for fine-tuning (kaldi-style dir)."""
    from .speech.dataset import build_dataset

    cfg = load_config(config)
    rep = build_dataset(sessions, script, out, cfg, max_cer=max_cer)
    typer.echo(f"{rep['clips']}/{rep['lines']} lines, {rep['minutes']} min of speech → {out}")
    if rep["missing_lines"]:
        typer.echo(f"missing lines ({len(rep['missing_lines'])}): {' '.join(rep['missing_lines'][:60])}"
                   + (" …" if len(rep["missing_lines"]) > 60 else ""))
    _hard_exit(0)


@app.command("tts-bakeoff")
def tts_bakeoff(
    config: Optional[Path] = typer.Option(None, "--config"),
    spec: Path = typer.Option(Path("config/bakeoff.yaml"), "--spec"),
    only: Optional[list[str]] = typer.Option(None, "--only", help="Run only these variant names"),
):
    """Every TTS variant reads the same lines → work/bakeoff/{report.md,index.html,key.json}."""
    import yaml

    from .pipeline import load_script
    from .speech.voices import bakeoff

    cfg = load_config(config)
    sp = yaml.safe_load(spec.read_text(encoding="utf-8"))
    if only:
        sp["variants"] = [v for v in sp["variants"] if v["name"] in only]
    lines = list(sp.get("lines") or [])
    if sp.get("include_scripts", True):
        for p in sorted(Path(cfg["paths"]["scripts"]).glob("*.json")):
            s = load_script(p)
            lines += [s.hook.narration, s.read.given_narration, s.read.target_narration]
            lines += [st.narration for st in s.steps] + [s.substitute.narration, s.answer.narration]
    report = bakeoff(cfg, sp, cfg["paths"]["work"] / "bakeoff", lines)
    typer.echo(str(report))
    _hard_exit(0)


@app.command()
def say(
    text: str,
    config: Optional[Path] = typer.Option(None, "--config"),
    out: Path = typer.Option(Path("work/say.wav"), "--out"),
):
    """Speak one line with the configured voice (shows the normalized text and caption tokens)."""
    from .speech.normalize import to_speech, tokenize
    from .speech.tts import get_engine

    cfg = load_config(config)
    eng = get_engine(cfg["tts"], Path(os.environ.get("FACTORY_CACHE") or cfg["paths"]["work"] / "_cache"))
    spoken = eng.prepare(to_speech(text))
    typer.echo(f"spoken:   {spoken}")
    typer.echo("captions: " + " | ".join(t.display for t in tokenize(text) if t.display))
    eng.synth_to_file(spoken, out)
    typer.echo(f"{out} ({eng.describe()})")
    _hard_exit(0)


@app.command()
def schema(out: Path = typer.Option(ROOT / "schemas", "--out")):
    """Export JSON Schemas (used for constrained decoding in M3)."""
    from .models import Task, VideoScript

    out.mkdir(parents=True, exist_ok=True)
    for name, model in (("task", Task), ("video_script", VideoScript)):
        (out / f"{name}.schema.json").write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    typer.echo(f"wrote schemas to {out}")


if __name__ == "__main__":
    app()
