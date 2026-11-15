"""Pluggable TTS engines. Each engine turns one text into one mono clip."""

from __future__ import annotations

from .base import TTSEngine, get_engine

__all__ = ["TTSEngine", "get_engine"]
