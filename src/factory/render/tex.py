"""XeLaTeX templates.

* sheet: Times-like serif (Liberation Serif) for the exam sheet (КИМ look).
* hand:  handwriting face (Caveat Bold) for text *and* math, so the big formula morphs
  look hand-written while TeX still lays out powers and fractions.

Math uses unicode-math with Latin Modern Math as the base (radicals, fraction bars,
big operators) and remaps digits, Latin letters and the basic operators to the
text font by Unicode range. (mathastext leaves digits in CM under XeTeX.)
"""

from __future__ import annotations

from functools import lru_cache

from manim import TexTemplate

# + = − ( ) / < > , . : |   (· and × stay in Latin Modern: Caveat lacks them)
_OPS = r'"2B,"3D,"2212,"28,"29,"2F,"3C,"3E,"2C,"2E,"3A,"7C'

_BASE = r"""
\usepackage{amsmath}
\usepackage{fontspec}
\usepackage{unicode-math}
\setmainfont{%(font)s}%(font_opts)s
\setmathfont{Latin Modern Math}
\setmathfont{%(up)s}[range={up/num,up/latin,up/Latin,%(ops)s}]
\setmathfont{%(it)s}[range={it/latin,it/Latin}]
\hyphenpenalty=10000
\exhyphenpenalty=10000
\tolerance=9999
\emergencystretch=3em
\setlength{\parindent}{0pt}
\linespread{%(spread)s}
"""


def _template(font: str, up: str, it: str, font_opts: str = "", spread: str = "1.15") -> TexTemplate:
    return TexTemplate(
        tex_compiler="xelatex",
        output_format=".xdv",
        documentclass=r"\documentclass[preview]{standalone}",
        preamble=_BASE % {"font": font, "font_opts": font_opts, "up": up, "it": it, "ops": _OPS, "spread": spread},
    )


@lru_cache(maxsize=4)
def sheet_template(font: str = "Liberation Serif") -> TexTemplate:
    return _template(font, up=font, it=f"{font} Italic")


@lru_cache(maxsize=4)
def hand_template(font: str = "Caveat") -> TexTemplate:
    # Caveat has no italic; upright handwriting for letters too.
    return _template(font, up=f"{font} Bold", it=f"{font} Bold", font_opts="[UprightFont={* Bold}]", spread="1.0")
