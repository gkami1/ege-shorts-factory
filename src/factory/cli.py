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
