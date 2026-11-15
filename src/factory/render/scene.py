"""The short as one Manim scene, driven by VideoScript + beat timeline.

Beat → visuals (each beat lasts exactly its voice clip + padding):
  hook       sheet on screen from frame 0, handwritten caption + marker arrow
  given      zoom into the sheet, yellow highlighter swipe over the given data
  target     marker circle around what we must find
  step_i     full-screen handwritten formula; morphs step to step
  substitute back to the sheet, known value flashes red, red calculation
  answer     answer written into the field, check mark, confetti
  outro      static CTA card
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AnimationGroup,
    Create,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    ImageMobject,
    Indicate,
    MathTex,
    Scene,
    Text,
    TransformFromCopy,
    TransformMatchingShapes,
    TransformMatchingTex,
    VGroup,
    Write,
    config,
    linear,
    rate_functions,
)

from ..models import Task, VideoScript
from ..timeline import Segment
from .fx import Confetti, check_mark, hand_arrow, hand_ellipse, highlighter, paper_texture, underline
from .sheet import ExamSheet
from .tex import hand_template


def wrap_lines(text: str, max_chars: int) -> str:
    """Greedy word wrap, then rebalance so lines have similar length."""
    words = text.split()
    n_lines = max(1, -(-len(text) // max_chars))
    target = len(text) / n_lines
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > target + 3 and len(lines) < n_lines - 1:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


class ShortScene(Scene):
    def __init__(self, script: VideoScript, task: Task, segs: list[Segment], colors: dict, render_cfg: dict, work: Path, **kw):
        self.script, self.task, self.colors, self.rcfg, self.work = script, task, colors, render_cfg, work
        self.S = {s.key: s for s in segs}
        self.seed = sum(map(ord, task.id))
        super().__init__(**kw)

    # ------------------------------------------------------------- timing utils

    def _now(self) -> float:
        return self.renderer.time

    def finish(self, seg: Segment) -> None:
        rem = seg.end - self._now()
        if rem > 0.5 / config.frame_rate:
            self.wait(rem)

    def budget(self, seg: Segment, want: float, reserve: float = 0.0) -> float:
        """Animation run time that still fits the beat."""
        return float(max(1 / config.frame_rate, min(want, seg.end - self._now() - reserve)))

    # ----------------------------------------------------------------- helpers

    def hand_text(self, s: str, size: float, color: str) -> Text:
        return Text(s, font=self.rcfg["hand_font"], weight=BOLD, font_size=size, color=color)

    def formula(self, tex: str, color: str, max_w: float = 7.8, max_h: float = 2.2, font_size: float = 300) -> MathTex:
        m = MathTex(tex, tex_template=hand_template(self.rcfg["hand_font"]), font_size=font_size, color=color)
        if m.width > max_w:
            m.scale_to_fit_width(max_w)
        if m.height > max_h:
            m.scale_to_fit_height(max_h)
        return m

    # ------------------------------------------------------------------- scene

    def construct(self):
        C, sc, task, S = self.colors, self.script, self.task, self.S
        tex_path = paper_texture(
            self.work.parent / "_cache" / f"paper_{C['paper_bg'][1:]}.png", 1080, 1920, C["paper_bg"]
        )
        bg = ImageMobject(str(tex_path)).set(height=config.frame_height)
        bg.set_z_index(-20)
        self.add(bg)

        spans = [*sc.read.given, sc.read.target, sc.substitute.flash]
        sheet = ExamSheet(task, spans, C, self.rcfg["sheet_font"], width=8.0)
        sheet.scale(0.98).move_to(DOWN * 0.45)

        # ---------------------------------------------------------------- hook
        seg = S["hook"]
        cap = self.hand_text(wrap_lines(sc.hook.caption, 17), 92, C["ink"])
        cap.set(width=min(cap.width, 7.8))
        cap.next_to(sheet, UP, buff=0.9).rotate(np.deg2rad(-2.5))
        if cap.get_top()[1] > 6.4:
            cap.shift(DOWN * (cap.get_top()[1] - 6.4))
        ul = underline(cap,C["accent"], self.seed)
        arrow = hand_arrow(
            cap.get_bottom() + LEFT * 1.6 + DOWN * 0.1,
            sheet.badge.get_top() + UP * 0.12 + RIGHT * 0.1,
            C["accent"],
            self.seed,
            bend=-0.25,
        )
        self.add(sheet)
        self.play(Write(cap), run_time=self.budget(seg, 0.9))
        self.play(Create(ul), Create(arrow), run_time=self.budget(seg, 0.45))
        self.finish(seg)

        # ---------------------------------------------------------------- given
        seg = S["given"]
        focus = sheet.focus()
        zoom = 8.5 / focus.width
        c = sheet.get_center()
        focus_c = c + zoom * (focus.get_center() - c)
        self.play(
            FadeOut(VGroup(cap, ul, arrow), shift=UP * 0.6),
            sheet.animate.scale(zoom, about_point=c).shift(np.array([0, 0.9, 0]) - focus_c),
            run_time=self.budget(seg, 0.6),
            rate_func=rate_functions.ease_in_out_cubic,
        )
        marks = VGroup()
        for i, span in enumerate(sc.read.given):
            for r in highlighter(sheet.part(span), C["highlight"], self.seed + i):
                r.set_z_index(-1)
                marks.add(r)
                sheet.add(r)
                self.play(GrowFromEdge(r, LEFT), run_time=self.budget(seg, 0.45), rate_func=linear)
        self.finish(seg)

        # --------------------------------------------------------------- target
        seg = S["target"]
        target = sheet.part(sc.read.target)
        circ = hand_ellipse(target, C["accent"], self.seed + 11)
        sheet.add(circ)
        self.play(Create(circ), run_time=self.budget(seg, 0.6), rate_func=rate_functions.ease_out_sine)
        self.finish(seg)

        # ---------------------------------------------------------------- steps
        prev = None
        for i, step in enumerate(sc.steps):
            seg = S[f"step_{i}"]
            f = self.formula(step.tex, C["formula"]).move_to(UP * 0.9)
            if prev is None:
                self.play(
                    FadeOut(sheet, scale=1.15),
                    TransformFromCopy(target, f),
                    run_time=self.budget(seg, 0.8),
                    rate_func=rate_functions.ease_in_out_cubic,
                )
            else:
                kind = TransformMatchingTex if "{{" in step.tex else TransformMatchingShapes
                self.play(kind(prev, f), run_time=self.budget(seg, 0.8), rate_func=rate_functions.ease_in_out_cubic)
            prev = f
            self.finish(seg)

        # ----------------------------------------------------------- substitute
        seg = S["substitute"]
        self.play(FadeOut(prev, shift=UP * 0.8), FadeIn(sheet, scale=0.9), run_time=self.budget(seg, 0.55))
        flash = sheet.part(sc.substitute.flash)
        flash.set_color(C["accent"])
        self.play(Indicate(flash, color=C["accent"], scale_factor=1.3), run_time=self.budget(seg, 0.7))
        if sc.substitute.tex:
            calc = self.formula(sc.substitute.tex, C["accent"], max_w=5.6, max_h=0.7, font_size=110)
            calc.move_to(sheet.work_point(), aligned_edge=LEFT).shift(RIGHT * 0.15)
            sheet.add(calc)
            self.play(Write(calc), run_time=self.budget(seg, 1.0, reserve=0.2))
        self.finish(seg)

        # --------------------------------------------------------------- answer
        seg = S["answer"]
        ans = self.hand_text(sc.answer.value, 96, C["pen"])
        ans.move_to(sheet.answer_slot(), aligned_edge=DOWN + LEFT)
        tick = check_mark(ans.get_right() + RIGHT * 0.6 + UP * 0.25, 0.6, C["good"])
        conf = Confetti(ans.get_center(), C["confetti"].split(","), n=110, seed=self.seed)
        conf.set_z_index(10)
        self.play(Write(ans), run_time=self.budget(seg, 0.45))
        self.play(Create(tick), run_time=self.budget(seg, 0.3))
        self.add(conf)
        t_conf = max(0.1, seg.end - self._now())
        self.play(conf.t.animate.set_value(t_conf), run_time=t_conf, rate_func=linear)
        self.remove(conf)
        self.finish(seg)

        # ---------------------------------------------------------------- outro
        seg = S["outro"]
        cta = self.hand_text(wrap_lines(sc.outro.cta, 14), 120, C["ink"])
        cta.set(width=min(cta.width, 7.6)).move_to(UP * 2.4)
        ul2 = underline(cta,C["accent"], self.seed + 5)
        arr = hand_arrow(cta.get_bottom() + DOWN * 0.5 + LEFT * 0.4, np.array([-3.0, -4.4, 0]), C["accent"], self.seed + 7, bend=0.3)
        everything = [m for m in self.mobjects if m is not bg]
        self.play(FadeOut(*everything), FadeIn(cta, shift=UP * 0.3), run_time=self.budget(seg, 0.45))
        self.add(ul2, arr)  # static card: no further animation
        self.finish(seg)
