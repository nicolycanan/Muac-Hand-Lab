"""Supported animated fills for persistent rectangle instances."""

from enum import StrEnum


class RectangleFillStyle(StrEnum):
    PSYCHEDELIC = "PSYCHEDELIC"
    MONOCHROME = "MONOCHROME"
    GLITCH = "GLITCH"

    def next(self) -> "RectangleFillStyle":
        styles = tuple(RectangleFillStyle)
        return styles[(styles.index(self) + 1) % len(styles)]
