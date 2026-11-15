"""Palettes and per-video style choice (variation knobs grow here in M4)."""

from __future__ import annotations

PALETTES: dict[str, dict[str, str]] = {
    "classic": {
        "paper_bg": "#EFE8D8",   # desk / background paper
        "sheet": "#FFFFFF",
        "ink": "#1E1E1E",
        "muted": "#7A7468",
        "accent": "#E0362B",      # red pen, circle, flash
        "pen": "#1F49C7",         # blue pen for the answer
        "highlight": "#FFE24A",
        "good": "#1FA35B",
        "formula": "#1E1E1E",
        "caption_text": "#FFFFFF",
        "caption_box": "#E0362B",
        "caption_active": "#FFE24A",
        "confetti": "#E0362B,#FFC93C,#1F49C7,#1FA35B,#FF7EB6,#8B5CF6",
    },
    "mint": {
        "paper_bg": "#E4EFE9",
        "sheet": "#FFFFFF",
        "ink": "#18231F",
        "muted": "#6C7B74",
        "accent": "#E5484D",
        "pen": "#2453D6",
        "highlight": "#B6F36B",
        "good": "#13915A",
        "formula": "#18231F",
        "caption_text": "#FFFFFF",
        "caption_box": "#0E8F6A",
        "caption_active": "#FFE24A",
        "confetti": "#E5484D,#FFC93C,#2453D6,#13915A,#FF7EB6",
    },
}


def palette(name: str) -> dict[str, str]:
    return PALETTES[name]
