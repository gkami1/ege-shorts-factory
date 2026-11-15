"""Word-by-word burned-in captions as ASS (libass via ffmpeg).

Style: 2–4 words per chunk, white bold text on a colored rounded box, the word being
spoken tinted. The box is an ASS vector drawing sized from real glyph metrics, so it
hugs the text for any font.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from ..timeline import Segment

_BREAK_AFTER = (".", ",", "!", "?", ":", ";", "—")


def _ass_color(hex_rgb: str, alpha: int = 0) -> str:
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ts(t: float) -> str:
    cs = int(round(max(t, 0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


@lru_cache(maxsize=8)
def _measurer(font_file: str, size: int):
    """libass sizes fonts so that winAscent+winDescent == \\fs; PIL sizes by em. Convert."""
    from fontTools.ttLib import TTFont

    f = TTFont(font_file)
    os2, upm = f["OS/2"], f["head"].unitsPerEm
    em_px = size * upm / (os2.usWinAscent + os2.usWinDescent)
    font = ImageFont.truetype(font_file, size=max(1, round(em_px * 4)))  # 4x for sub-pixel accuracy
    cap_h = os2.sCapHeight * em_px / upm if getattr(os2, "sCapHeight", 0) else 0.7 * em_px
    return font, cap_h


def text_width(text: str, font_file: str, size: int) -> float:
    font, _ = _measurer(font_file, size)
    return font.getlength(text) / 4


def chunk_words(words: list[dict], max_words: int, max_chars: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(x["w"] for x in cur + [w])) > max_chars):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if w["w"].endswith(_BREAK_AFTER):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    # avoid orphan single short words when the previous chunk has room
    return chunks


def _rounded_rect(w: float, h: float, r: float) -> str:
    r = min(r, h / 2, w / 2)
    k = 0.5523 * r
    p = lambda *xy: " ".join(f"{v:.0f}" for v in xy)  # noqa: E731
    return (
        f"m {p(r, 0)} l {p(w - r, 0)} b {p(w - r + k, 0, w, r - k, w, r)} "
        f"l {p(w, h - r)} b {p(w, h - r + k, w - r + k, h, w - r, h)} "
        f"l {p(r, h)} b {p(r - k, h, 0, h - r + k, 0, h - r)} "
        f"l {p(0, r)} b {p(0, r - k, r - k, 0, r, 0)}"
    )


def build_ass(segs: list[Segment], cfg: dict, colors: dict, width: int, height: int) -> str:
    font, size, ffile = cfg["font"], int(cfg["size"]), cfg["font_file"]
    _, cap_h = _measurer(ffile, size)
    box_h = cap_h + 2 * cfg["pad_y"] + size * 0.28
    x, y = width // 2, int(cfg["y"])
    text_c, box_c, active_c = (_ass_color(colors[k]) for k in ("caption_text", "caption_box", "caption_active"))

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},{text_c},{text_c},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1
Style: Box,{font},{size},{box_c},{box_c},&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,0,6,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines: list[str] = []
    all_words = [w for s in segs for w in s.words]
    chunks = chunk_words(all_words, cfg["max_words"], cfg["max_chars"])
    for ci, ch in enumerate(chunks):
        start = ch[0]["start"]
        nxt = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else ch[-1]["end"] + 0.6
        end = min(nxt, ch[-1]["end"] + 0.6)
        if end - start < 0.25:
            end = start + 0.25
        text = " ".join(w["w"] for w in ch)
        bw = text_width(text, ffile, size) + 2 * cfg["pad_x"]
        pop = r"\t(0,90,\fscx108\fscy108)\t(90,170,\fscx100\fscy100)"
        lines.append(
            f"Dialogue: 0,{_ts(start)},{_ts(end)},Box,,0,0,0,,{{\\an5\\pos({x},{y}){pop}\\p1}}"
            f"{_rounded_rect(bw, box_h, cfg['radius'])}{{\\p0}}"
        )
        for wi, w in enumerate(ch):
            ws = start if wi == 0 else w["start"]
            we = ch[wi + 1]["start"] if wi + 1 < len(ch) else end
            if we <= ws:
                continue
            parts = [
                (f"{{\\c{active_c}}}{x_['w']}{{\\c{text_c}}}" if k == wi else x_["w"]) for k, x_ in enumerate(ch)
            ]
            anim = pop if wi == 0 else ""
            lines.append(
                f"Dialogue: 1,{_ts(ws)},{_ts(we)},Cap,,0,0,0,,{{\\an5\\pos({x},{y}){anim}}}" + " ".join(parts)
            )
    return head + "\n".join(lines) + "\n"


def write_ass(segs: list[Segment], cfg: dict, colors: dict, width: int, height: int, out: Path) -> Path:
    out.write_text(build_ass(segs, cfg, colors, width, height), encoding="utf-8")
    return out
