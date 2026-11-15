"""The exam-sheet mobject: КИМ-style task card with number badge, condition and answer field."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UP,
    Line,
    VectorizedPoint,
    RoundedRectangle,
    Square,
    Tex,
    Text,
    VGroup,
)

from ..models import Task, split_condition
from .tex import sheet_template

SUBJECT_TITLE = {
    "math_profile": "Математика. Профильный уровень",
    "math_base": "Математика. Базовый уровень",
    "physics": "Физика",
}
MINIPAGE_CM = 6.0
TEX_FONT_SIZE = 48


@lru_cache(maxsize=4)
def _cm_to_units(font: str) -> float:
    """Manim width of a 10 cm TeX rule at TEX_FONT_SIZE (calibrates minipage → scene units)."""
    rule = Tex(rf"\rule{{{MINIPAGE_CM}cm}}{{2pt}}", tex_template=sheet_template(font), font_size=TEX_FONT_SIZE)
    return rule.width / MINIPAGE_CM


class ExamSheet(VGroup):
    def __init__(self, task: Task, spans: list[str], colors: dict, sheet_font: str, width: float = 8.2):
        super().__init__()
        self.colors = colors
        ink, muted = colors["ink"], colors["muted"]
        margin = 0.42
        badge_side = 0.62
        gap = 0.28
        cond_w = width - 2 * margin - badge_side - gap

        parts, self.span_index = split_condition(task.condition, spans)
        parts = [r"\raggedright " + parts[0], *parts[1:]]  # no glyphs, keeps part indices
        cond = Tex(
            *parts,
            tex_template=sheet_template(sheet_font),
            tex_environment=f"{{minipage}}{{{MINIPAGE_CM}cm}}",
            font_size=TEX_FONT_SIZE,
        )
        cond.set_color(ink)
        cond.scale(cond_w / (MINIPAGE_CM * _cm_to_units(sheet_font)))
        self.cond = cond

        header = Text(SUBJECT_TITLE[task.subject], font=sheet_font, font_size=22, color=muted)
        part = Text("Часть 1", font=sheet_font, font_size=22, color=muted)

        badge_box = Square(side_length=badge_side, stroke_width=0).set_fill(ink, 1)
        badge_num = Text(str(task.kim_number), font=sheet_font, weight=BOLD, font_size=30, color=colors["sheet"])
        badge_num.move_to(badge_box)
        self.badge = VGroup(badge_box, badge_num)

        ans_label = Text("Ответ:", font=sheet_font, font_size=30, color=ink)
        ans_line = Line(ORIGIN3, RIGHT * 3.4, stroke_width=3, color=ink)
        dot = Text(".", font=sheet_font, font_size=30, color=ink)

        # layout (top-left origin)
        x0, y0 = -width / 2 + margin, 0.0
        header.move_to([x0, y0, 0], aligned_edge=UL)
        part.move_to([width / 2 - margin, y0, 0], aligned_edge=UP + RIGHT)
        top = header.get_bottom()[1] - 0.38
        self.badge.move_to([x0, top, 0], aligned_edge=UL)
        cond.move_to([x0 + badge_side + gap, top + 0.04, 0], aligned_edge=UL)
        self.work_top = cond.get_bottom()[1] - 0.2
        work_h = 1.7
        ans_y = self.work_top - work_h
        ans_label.move_to([x0 + badge_side + gap, ans_y, 0], aligned_edge=LEFT)
        ans_line.put_start_and_end_on(
            np.array([ans_label.get_right()[0] + 0.2, ans_label.get_bottom()[1] - 0.04, 0]),
            np.array([ans_label.get_right()[0] + 3.6, ans_label.get_bottom()[1] - 0.04, 0]),
        )
        dot.next_to(ans_line, RIGHT, buff=0.05, aligned_edge=DOWN)
        bottom = ans_y - 0.7
        h = (y0 + margin) - bottom
        paper = RoundedRectangle(width=width, height=h, corner_radius=0.1, stroke_width=0)
        paper.set_fill(colors["sheet"], 1).move_to([0, (y0 + margin + bottom) / 2, 0])
        shadow = paper.copy().set_fill("#000000", 0.13).shift(np.array([0.07, -0.11, 0]))
        shadow.set_z_index(-4)
        paper.set_z_index(-3)

        self.paper = paper
        self.ans_label, self.ans_line = ans_label, ans_line
        # invisible anchor for the red calculation; moves with the sheet on zoom
        self.work_anchor = VectorizedPoint(np.array([cond.get_left()[0], self.work_top - 0.55, 0]))
        self.add(shadow, paper, header, part, self.badge, cond, ans_label, ans_line, dot, self.work_anchor)
        self.center()

    def part(self, span: str):
        return self.cond.submobjects[self.span_index[span]]

    def answer_slot(self) -> np.ndarray:
        """Point on the answer line where the handwritten answer goes (left part, baseline)."""
        return self.ans_line.get_start() + RIGHT * 0.25 + UP * 0.06

    def work_point(self) -> np.ndarray:
        """Left edge of the free area between condition and answer line (for the red calculation)."""
        return self.work_anchor.get_center()

    def focus(self) -> VGroup:
        return VGroup(self.badge, self.cond, self.ans_label, self.ans_line)


ORIGIN3 = np.zeros(3)
