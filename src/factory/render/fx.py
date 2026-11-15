"""Hand-drawn effects, all procedural and seeded (deterministic output per task)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Rectangle,
    ValueTracker,
    VGroup,
    VMobject,
)
from PIL import Image, ImageFilter


# ----------------------------------------------------------------- background


def paper_texture(path: Path, w: int, h: int, color: str, seed: int = 7) -> Path:
    """Subtle paper: low-frequency blotches + fine grain + faint fibres. Cached on disk."""
    if path.exists():
        return path
    rng = np.random.default_rng(seed)
    base = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], np.float32)

    def blur_noise(scale: int, radius: float) -> np.ndarray:
        small = rng.normal(0, 1, (h // scale + 2, w // scale + 2)).astype(np.float32)
        img = Image.fromarray(((small - small.min()) / np.ptp(small) * 255).astype(np.uint8))
        img = img.resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(radius))
        a = np.asarray(img, np.float32) / 255.0
        return a - a.mean()

    lum = 10 * blur_noise(60, 30) + 5 * blur_noise(8, 2) + rng.normal(0, 2.2, (h, w))
    fib = Image.fromarray(np.zeros((h, w), np.uint8))
    from PIL import ImageDraw

    d = ImageDraw.Draw(fib)
    for _ in range(260):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang, ln = rng.uniform(0, np.pi), rng.uniform(10, 60)
        d.line([(x, y), (x + ln * np.cos(ang), y + ln * np.sin(ang))], fill=int(rng.uniform(20, 70)), width=1)
    fib_a = np.asarray(fib.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0
    lum = lum - 9 * fib_a
    # vignette
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    lum -= 10 * np.clip(r - 0.6, 0, None) ** 2
    rgb = np.clip(base[None, None, :] + lum[..., None], 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


# ------------------------------------------------------------------ strokes


def _wobbly(pts: np.ndarray) -> VMobject:
    m = VMobject()
    m.set_points_smoothly(pts)
    return m


def hand_ellipse(mob, color: str, seed: int, pad: float = 0.16, width: float = 7) -> VMobject:
    """A marker circle: slightly tilted, uneven, ends overshoot instead of closing."""
    rng = np.random.default_rng(seed)
    c = mob.get_center()
    a, b = mob.width / 2 + pad, mob.height / 2 + pad * 1.1
    n = 90
    t0 = rng.uniform(0.6, 1.4) * np.pi
    t = t0 + np.linspace(0, 2 * np.pi * 1.1, n)
    drift = np.linspace(1.0, 1.09, n)
    wob = 1 + 0.035 * np.sin(2 * t + rng.uniform(0, 6)) + 0.02 * np.sin(5 * t + rng.uniform(0, 6))
    x, y = a * np.cos(t) * wob * drift, b * np.sin(t) * wob * drift
    tilt = np.deg2rad(rng.uniform(-6, 6))
    xr, yr = x * np.cos(tilt) - y * np.sin(tilt), x * np.sin(tilt) + y * np.cos(tilt)
    pts = np.stack([xr + c[0], yr + c[1], np.zeros(n)], axis=1)
    return _wobbly(pts).set_stroke(color, width=width, opacity=0.95).set_fill(opacity=0)


def hand_arrow(start: np.ndarray, end: np.ndarray, color: str, seed: int, bend: float = 0.35, width: float = 7) -> VGroup:
    """Curved marker arrow from start to end with a two-stroke head."""
    rng = np.random.default_rng(seed)
    start, end = np.asarray(start, float), np.asarray(end, float)
    d = end - start
    nrm = np.array([-d[1], d[0], 0]) / (np.linalg.norm(d) + 1e-9)
    ts = np.linspace(0, 1, 24)
    pts = start + np.outer(ts, d) + np.outer(np.sin(np.pi * ts) * bend * np.linalg.norm(d), nrm)
    pts += np.c_[rng.normal(0, 0.012, (24, 2)), np.zeros(24)]
    shaft = _wobbly(pts).set_stroke(color, width=width)
    tip, prev = pts[-1], pts[-4]
    u = (tip - prev) / np.linalg.norm(tip - prev)
    heads = VGroup()
    for ang in (2.5, -2.5):
        rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        p = tip + rot @ u * 0.38
        heads.add(_wobbly(np.array([tip, (tip + p) / 2 + rng.normal(0, 0.01, 3) * [1, 1, 0], p])).set_stroke(color, width=width))
    return VGroup(shaft, heads)


def check_mark(center: np.ndarray, size: float, color: str, width: float = 10) -> VMobject:
    c = np.asarray(center)
    pts = np.array([c + size * np.array(v) for v in ([-0.5, 0.05, 0], [-0.15, -0.35, 0], [0.55, 0.5, 0])])
    m = VMobject().set_points_as_corners(pts).set_stroke(color, width=width)
    return m


def underline(mob, color: str, seed: int, width: float = 6) -> VMobject:
    rng = np.random.default_rng(seed)
    l, r = mob.get_corner(DOWN + LEFT), mob.get_corner(DOWN + RIGHT)
    xs = np.linspace(l[0] - 0.05, r[0] + 0.1, 12)
    ys = l[1] - 0.08 + rng.normal(0, 0.012, 12) + np.linspace(0.02, -0.03, 12)
    return _wobbly(np.c_[xs, ys, np.zeros(12)]).set_stroke(color, width=width)


# ---------------------------------------------------------------- highlighter


def line_groups(mob, tol_factor: float = 0.75) -> list[VGroup]:
    """Cluster glyphs into visual lines (a span may wrap)."""
    glyphs = [g for g in mob.family_members_with_points() if g.width > 1e-3 or g.height > 1e-3]
    if not glyphs:
        return [VGroup(mob)]
    h = float(np.median([g.height for g in glyphs]))
    groups: list[list] = []
    for g in sorted(glyphs, key=lambda g: -g.get_center()[1]):
        if groups and abs(np.mean([x.get_center()[1] for x in groups[-1]]) - g.get_center()[1]) < tol_factor * max(h, 0.15) * 1.6:
            groups[-1].append(g)
        else:
            groups.append([g])
    return [VGroup(*gs) for gs in groups]


def highlighter(mob, color: str, seed: int, opacity: float = 0.85) -> list[Rectangle]:
    """One marker stroke per visual line, slightly irregular, drawn behind the text."""
    rng = np.random.default_rng(seed)
    rects = []
    for grp in line_groups(mob):
        h = max(grp.height, 0.32) * 1.18
        r = Rectangle(width=grp.width + 0.14, height=h, stroke_width=0)
        r.set_fill(color, opacity=opacity)
        r.move_to(grp.get_center() + np.array([0.0, -0.02, 0]))
        r.rotate(np.deg2rad(rng.uniform(-1.2, 1.2)))
        rects.append(r)
    return rects


# ------------------------------------------------------------------- confetti


class Confetti(VGroup):
    """Seeded burst of paper bits; drive with `self.t` (ValueTracker, seconds)."""

    def __init__(self, origin, colors: list[str], n: int = 90, seed: int = 1, **kw):
        super().__init__(**kw)
        rng = np.random.default_rng(seed)
        self.t = ValueTracker(0.0)
        self.origin = np.asarray(origin, float)
        self.params = []
        for i in range(n):
            w, h = rng.uniform(0.09, 0.17), rng.uniform(0.16, 0.30)
            piece = Rectangle(width=w, height=h, stroke_width=0).set_fill(colors[i % len(colors)], opacity=1)
            ang = rng.uniform(np.deg2rad(20), np.deg2rad(160))
            spd = rng.uniform(6, 13)
            self.params.append(
                dict(
                    base=piece.points.copy() - piece.get_center(),
                    v=np.array([np.cos(ang) * spd * 0.75, np.sin(ang) * spd, 0]),
                    spin=rng.uniform(-9, 9),
                    flip=rng.uniform(4, 12),
                    phase=rng.uniform(0, 6),
                )
            )
            self.add(piece)
        self.add_updater(lambda m: m._step())
        self._step()

    def _step(self):
        t = self.t.get_value()
        g = np.array([0, -15.0, 0])
        drag = np.exp(-1.1 * t)
        for piece, p in zip(self.submobjects, self.params):
            pos = self.origin + p["v"] * (1 - drag) / 1.1 + 0.5 * g * t * t * 0.55
            a = p["spin"] * t + p["phase"]
            sx = np.cos(p["flip"] * t + p["phase"])
            rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            pts = p["base"] * np.array([sx if abs(sx) > 0.08 else 0.08, 1, 1])
            piece.points = pts @ rot.T + pos
            piece.set_fill(opacity=1.0 if t < 1.6 else max(0.0, 1 - (t - 1.6) / 0.6))


__all__ = [
    "paper_texture",
    "hand_ellipse",
    "hand_arrow",
    "check_mark",
    "underline",
    "highlighter",
    "line_groups",
    "Confetti",
    "LEFT",
    "RIGHT",
    "UP",
]
