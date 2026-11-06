"""Generation utilities for particle-based heart shapes."""

from __future__ import annotations

import cv2
import numpy as np


def heart_parametric(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = 16.0 * np.sin(t) ** 3
    y = (
        13.0 * np.cos(t)
        - 5.0 * np.cos(2.0 * t)
        - 2.0 * np.cos(3.0 * t)
        - np.cos(4.0 * t)
    )
    return x, -y


def generate_heart_targets(count: int = 1500, radius: float = 1.0) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")

    t = np.linspace(0.0, 2.0 * np.pi, max(count * 8, 4000), endpoint=False)
    x, y = heart_parametric(t)
    x_norm = x / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else x
    y_norm = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

    canvas = np.zeros((512, 512), dtype=np.uint8)
    pts = np.column_stack(
        [
            (x_norm * 160.0 + 256.0).astype(np.int32),
            (y_norm * 160.0 + 256.0).astype(np.int32),
        ]
    )
    cv2.fillPoly(canvas, [pts], 255)
    ys, xs = np.nonzero(canvas)
    if len(xs) < count:
        raise ValueError("heart target generation failed to cover enough points")

    rng = np.random.default_rng(0)
    sample = rng.choice(len(xs), size=count, replace=False)
    coords = np.column_stack((xs[sample], ys[sample])).astype(np.float32)
    coords[:, 0] = (coords[:, 0] - 256.0) / 160.0
    coords[:, 1] = -(coords[:, 1] - 256.0) / 160.0
    coords *= radius
    return coords.astype(np.float32)
