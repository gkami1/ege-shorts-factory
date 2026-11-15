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
