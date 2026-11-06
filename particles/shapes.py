"""Gera os destinos das partículas do coração e das letras."""

from __future__ import annotations

from functools import lru_cache
from string import ascii_uppercase

import cv2
import numpy as np

from config import PARTICLE_COUNT


def heart_parametric(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retorna a curva em coordenadas cartesianas, com Y positivo para cima."""
    x = 16.0 * np.sin(t) ** 3
    y = (
        13.0 * np.cos(t)
        - 5.0 * np.cos(2.0 * t)
        - 2.0 * np.cos(3.0 * t)
        - np.cos(4.0 * t)
    )
    return x, y


def generate_heart_targets(count: int = PARTICLE_COUNT, radius: float = 1.0) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")

    t = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    x, y = heart_parametric(t)
    extent = max(np.abs(x).max(), np.abs(y).max())
    center, raster_scale = 256, 160.0

    canvas = np.zeros((512, 512), dtype=np.uint8)
    pts = np.column_stack(
        [
            np.rint(center + x * raster_scale / extent),
            np.rint(center - y * raster_scale / extent),
        ]
    ).astype(np.int32)
    cv2.fillPoly(canvas, [pts], 255)
    # Na máscara, Y cresce para baixo. Amostra faixas de área equivalente
    # em metade da forma e espelha os pontos para manter a simetria.
    ys, xs = np.nonzero(canvas[:, :center])
    pairs = count // 2
    if pairs > len(xs):
        raise ValueError("heart target generation failed to cover enough points")
    rng = np.random.default_rng(0)
    if pairs:
        boundaries = np.arange(pairs + 1) * len(xs) // pairs
        sample = rng.integers(boundaries[:-1], boundaries[1:])
    else:
        sample = np.empty(0, dtype=int)
    half = np.column_stack((xs[sample] - center, ys[sample] - center))
    mirrored = half * (-1, 1)
    points = np.concatenate((half, mirrored))
    if count % 2:
        axis = np.flatnonzero(canvas[:, center])
        points = np.vstack((points, (0, axis[len(axis) // 2] - center)))
    return points.astype(np.float32) * (radius / raster_scale)


@lru_cache(maxsize=len(ascii_uppercase))
def generate_letter_targets(letter: str, count: int = PARTICLE_COUNT) -> np.ndarray:
    """Amostra o traço da letra e o centraliza no mesmo espaço local do coração."""
    if len(letter) != 1 or letter not in ascii_uppercase or count <= 0:
        raise ValueError("Use uma letra de A a Z e uma quantidade positiva de partículas.")

    font, scale, stroke = cv2.FONT_HERSHEY_DUPLEX, 11.0, 24
    (width, height), baseline = cv2.getTextSize(letter, font, scale, stroke)
    margin = stroke
    mask = np.zeros((height + baseline + 2 * margin, width + 2 * margin), dtype=np.uint8)
    cv2.putText(mask, letter, (margin, height + margin), font, scale, 255, stroke)
    y, x = np.nonzero(mask)
    origin = np.array([(x.min() + x.max()) * 0.5, (y.min() + y.max()) * 0.5])
    extent = max(np.ptp(x), np.ptp(y)) * 0.5
    rng = np.random.default_rng(ord(letter))
    sample = rng.choice(len(x), count, replace=count > len(x))
    targets = ((np.column_stack((x[sample], y[sample])) - origin) * (0.9 / extent)).astype(np.float32)
    targets.setflags(write=False)
    return targets
