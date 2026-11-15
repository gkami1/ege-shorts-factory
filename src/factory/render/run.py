"""Run the Manim scene programmatically (CPU, Cairo renderer)."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..models import Task, VideoScript
from ..timeline import Segment
from .style import palette


def render_scene(
    script: VideoScript, task: Task, segs: list[Segment], cfg: dict, work: Path, draft: bool = False
) -> Path:
    from manim import tempconfig

    from .scene import ShortScene

    v = cfg["video"]
    w, h, fps = (540, 960, 15) if draft else (v["width"], v["height"], v["fps"])
    colors = palette(cfg["render"]["palette"])
    media = work / "manim"
    opts = {
        "pixel_width": w,
        "pixel_height": h,
        "frame_width": 9.0,
        "frame_height": 16.0,
        "frame_rate": fps,
        "media_dir": str(media),
        "output_file": "scene",
        "background_color": colors["paper_bg"],
        "disable_caching": True,
        "progress_bar": "none",
        "verbosity": "WARNING",
        "write_to_movie": True,
        "format": "mp4",
    }
    with tempconfig(opts):
        scene = ShortScene(script, task, segs, colors, cfg["render"], work)
        scene.render()
        produced = Path(scene.renderer.file_writer.movie_file_path)
    out = work / "video_silent.mp4"
    shutil.copyfile(produced, out)
    return out
