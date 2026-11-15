"""Final mux: silent render + burned ASS captions + voice (+ optional ducked music)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _esc(p: Path) -> str:
    """Escape a path for use inside an ffmpeg filtergraph option."""
    return str(p).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def assemble(video: Path, voice: Path, ass: Path, out: Path, cfg: dict) -> Path:
    v, a = cfg["video"], cfg["assemble"]
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-i", str(voice)]
    vf = f"[0:v]scale={v['width']}:{v['height']}:flags=lanczos,ass=filename={_esc(ass)}[v]"
    voice_chain = "[1:a]aformat=sample_rates=48000:channel_layouts=stereo"
    if a.get("loudnorm", True):
        voice_chain += ",loudnorm=I=-14:TP=-1.5:LRA=11"
    music = a.get("music")
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        af = (
            f"{voice_chain}[vo];[vo]asplit=2[vo1][vo2];"
            f"[2:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={a.get('music_volume', 0.07)}[mu];"
            f"[mu][vo1]sidechaincompress=threshold=0.05:ratio=6:attack=20:release=300[duck];"
            f"[vo2][duck]amix=inputs=2:duration=first:normalize=0[a]"
        )
    else:
        af = f"{voice_chain}[a]"
    cmd += [
        "-filter_complex", f"{vf};{af}",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", v["preset"], "-crf", str(v["crf"]), "-pix_fmt", "yuv420p",
        "-r", str(v["fps"]), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest", "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out
