"""Text → spoken Russian.

M1: minimal version (digits → words, strip markup). M2 replaces this with the full
LaTeX→speech normalizer (powers, fractions, units, decimals, stress marks) behind
the same `to_speech` entry point.
"""

from __future__ import annotations

import re

from num2words import num2words

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _num(m: re.Match) -> str:
    s = m.group(0).replace(",", ".")
    if "." in s:
        whole, frac = s.split(".")
        return f"{num2words(int(whole), lang='ru')} целых {num2words(int(frac), lang='ru')}"
    return num2words(int(s), lang="ru")


def to_speech(text: str) -> str:
    """Return text suitable for the TTS engine (may contain '+' stress marks)."""
    t = text.replace("$", "")
    t = t.replace("—", ",").replace("–", ",")
    t = _NUM.sub(_num, t)
    return re.sub(r"\s+", " ", t).strip()


def display_words(text: str) -> list[str]:
    """Words as they should appear in captions (stress marks removed)."""
    return [w.replace("+", "") for w in text.replace("$", "").split() if w not in {"—", "–", "-"}]
