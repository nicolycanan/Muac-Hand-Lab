"""Procedural animated plasma texture rendered only in a rectangle ROI."""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    GLITCH_BAND_COUNT,
    GLITCH_BLOCK_SIZE,
    GLITCH_INTENSITY,
    GLITCH_OPACITY,
    GLITCH_RGB_SHIFT,
    GLITCH_SCANLINE_STRENGTH,
    GLITCH_SPEED,
    MONOCHROME_CONTRAST,
    PSYCHEDELIC_FILL_MAX_DIMENSION,
    PSYCHEDELIC_FILL_HUE_SHIFT,
    PSYCHEDELIC_FILL_OPACITY,
    PSYCHEDELIC_FILL_SATURATION,
    PSYCHEDELIC_FILL_SCALE,
    PSYCHEDELIC_FILL_SPEED,
)
from effects.rectangle_style import RectangleFillStyle


class PsychedelicFill:
    GRID_BUCKET_SIZE = 16

    def __init__(self) -> None:
        self.time = 0.0
        self._grid_shape: tuple[int, int] | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def _coordinate_grid(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        if self._grid_shape != (height, width):
            x = np.linspace(0.0, 1.0, width, dtype=np.float32)
            y = np.linspace(0.0, 1.0, height, dtype=np.float32)
            self._x, self._y = np.meshgrid(x, y)
            self._grid_shape = (height, width)
        return self._x, self._y

    def _texture(self, height: int, width: int) -> np.ndarray:
        x, y = self._coordinate_grid(height, width)
        scale = max(float(PSYCHEDELIC_FILL_SCALE), 0.05)
        phase = self.time
        diagonal = np.sin((x * 4.2 + y * 3.1) * scale * 2.0 * np.pi + phase * 1.7)
        crossing = np.cos((x * 3.3 - y * 4.7) * scale * 2.0 * np.pi - phase * 1.2)
        radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        radial = np.sin(radius * scale * 38.0 - phase * 2.0)
        plasma = diagonal + crossing + radial

        hue = np.mod(
            (plasma * 0.105 + x * 0.16 + phase * 0.018) * 180.0
            + float(PSYCHEDELIC_FILL_HUE_SHIFT) * 0.5,
            180.0,
        )
        saturation = np.full_like(hue, np.clip(PSYCHEDELIC_FILL_SATURATION, 0.0, 1.0) * 255.0)
        value = np.clip(170.0 + 85.0 * (0.5 + 0.5 * np.sin(plasma * 1.25 + phase)), 0.0, 255.0)
        hsv = cv2.merge((
            hue.astype(np.uint8),
            saturation.astype(np.uint8),
            value.astype(np.uint8),
        ))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _monochrome_texture(self, height: int, width: int) -> np.ndarray:
        x, y = self._coordinate_grid(height, width)
        scale = max(float(PSYCHEDELIC_FILL_SCALE), 0.05)
        phase = self.time
        diagonal = np.sin((x * 4.2 + y * 3.1) * scale * 2.0 * np.pi + phase * 1.7)
        crossing = np.cos((x * 3.3 - y * 4.7) * scale * 2.0 * np.pi - phase * 1.2)
        radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        radial = np.sin(radius * scale * 38.0 - phase * 2.0)
        plasma = diagonal + crossing + radial
        gray = np.clip(
            127.5 + 127.5 * np.tanh(plasma * float(MONOCHROME_CONTRAST)),
            0.0,
            255.0,
        ).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _glitch_texture(self, roi: np.ndarray) -> np.ndarray:
        intensity = float(np.clip(GLITCH_INTENSITY, 0.0, 1.0))
        if intensity <= 0.0:
            return roi.copy()

        block_size = max(2, int(GLITCH_BLOCK_SIZE))
        phase = self.time * max(float(GLITCH_SPEED), 0.0)
        height, width = roi.shape[:2]
        glitched = roi.copy()
        band_count = max(1, min(int(GLITCH_BAND_COUNT), height))
        max_shift = max(1, int(round(block_size * intensity * 0.75)))

        for band in range(band_count):
            top = band * height // band_count
            bottom = (band + 1) * height // band_count
            shift = int(round(np.sin(phase * 3.0 + band * 1.7) * max_shift))
            if shift:
                glitched[top:bottom] = np.roll(roi[top:bottom], shift, axis=1)

        channel_shift = max(0, int(GLITCH_RGB_SHIFT))
        if channel_shift:
            glitched[:, :, 0] = np.roll(glitched[:, :, 0], -channel_shift, axis=1)
            glitched[:, :, 2] = np.roll(glitched[:, :, 2], channel_shift, axis=1)

        block_count = max(1, int(height * width / (block_size * block_size * 80)))
        for block in range(block_count):
            x_phase = phase * 1.3 + block * 2.17
            y_phase = phase * 0.9 + block * 1.71
            left = int((np.sin(x_phase) * 0.5 + 0.5) * max(width - block_size, 0))
            top = int((np.cos(y_phase) * 0.5 + 0.5) * max(height - block_size, 0))
            block_width = min(block_size, width - left)
            block_height = min(block_size, height - top)
            shift = int(round(np.sin(phase * 2.4 + block) * max_shift * 1.5))
            source_left = int(np.clip(left + shift, 0, width - block_width))
            glitched[top:top + block_height, left:left + block_width] = roi[
                top:top + block_height,
                source_left:source_left + block_width,
            ]

        scanline_strength = float(np.clip(GLITCH_SCANLINE_STRENGTH * intensity, 0.0, 0.3))
        if scanline_strength:
            glitched[::3] = np.clip(
                glitched[::3].astype(np.float32) * (1.0 - scanline_strength),
                0,
                255,
            ).astype(np.uint8)
        return glitched

    def render(
        self,
        frame: np.ndarray,
        first: tuple[int, int],
        second: tuple[int, int],
        dt: float,
        fade_alpha: float = 1.0,
        style: RectangleFillStyle = RectangleFillStyle.PSYCHEDELIC,
        source_roi: np.ndarray | None = None,
    ) -> None:
        x_min = min(first[0], second[0])
        x_max = max(first[0], second[0])
        y_min = min(first[1], second[1])
        y_max = max(first[1], second[1])
        if x_max - x_min <= 2 or y_max - y_min <= 2:
            return

        height, width = frame.shape[:2]
        left = max(0, x_min + 1)
        top = max(0, y_min + 1)
        right = min(width, x_max)
        bottom = min(height, y_max)
        roi_width = right - left
        roi_height = bottom - top
        if roi_width <= 0 or roi_height <= 0:
            return

        opacity = np.clip(PSYCHEDELIC_FILL_OPACITY * fade_alpha, 0.0, 1.0)
        if style == RectangleFillStyle.GLITCH:
            if GLITCH_OPACITY * fade_alpha <= 0.0:
                return
        elif opacity <= 0.0:
            return
        self.time += max(float(dt), 0.0) * max(float(PSYCHEDELIC_FILL_SPEED), 0.0)

        max_dimension = max(1, int(PSYCHEDELIC_FILL_MAX_DIMENSION))
        resolution_scale = min(1.0, max_dimension / max(roi_width, roi_height))
        target_width = max(1, int(round(roi_width * resolution_scale)))
        target_height = max(1, int(round(roi_height * resolution_scale)))
        texture_width = min(
            max_dimension,
            ((target_width + self.GRID_BUCKET_SIZE - 1) // self.GRID_BUCKET_SIZE)
            * self.GRID_BUCKET_SIZE,
        )
        texture_height = min(
            max_dimension,
            ((target_height + self.GRID_BUCKET_SIZE - 1) // self.GRID_BUCKET_SIZE)
            * self.GRID_BUCKET_SIZE,
        )
        if style == RectangleFillStyle.PSYCHEDELIC:
            texture = self._texture(texture_height, texture_width)
        elif style == RectangleFillStyle.MONOCHROME:
            texture = self._monochrome_texture(texture_height, texture_width)
        elif style == RectangleFillStyle.GLITCH:
            original_roi = (
                frame[top:bottom, left:right].copy()
                if source_roi is None
                else source_roi
            )
            source = cv2.resize(
                original_roi,
                (texture_width, texture_height),
                interpolation=cv2.INTER_AREA,
            )
            texture = self._glitch_texture(source)
        else:
            raise ValueError(f"Unsupported rectangle fill style: {style}")
        if (texture_width, texture_height) != (roi_width, roi_height):
            texture = cv2.resize(texture, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR)

        roi = frame[top:bottom, left:right]
        if style == RectangleFillStyle.GLITCH:
            original_roi = (
                roi.copy()
                if source_roi is None
                else source_roi
            )
            glitch_opacity = np.clip(GLITCH_OPACITY * fade_alpha, 0.0, 1.0)
            cv2.addWeighted(
                texture,
                float(glitch_opacity),
                original_roi,
                1.0 - float(glitch_opacity),
                0.0,
                dst=roi,
            )
        else:
            cv2.addWeighted(texture, float(opacity), roi, 1.0 - float(opacity), 0.0, dst=roi)
