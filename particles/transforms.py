"""Transformações afins para deslocar, girar e redimensionar os destinos do coração."""

from __future__ import annotations

import numpy as np


def translate_points(points: np.ndarray, dx: float, dy: float) -> np.ndarray:
    translated = np.asarray(points, dtype=np.float32).copy()
    translated[:, 0] += dx
    translated[:, 1] += dy
    return translated


def scale_points(points: np.ndarray, scale: float, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    points_arr = np.asarray(points, dtype=np.float32).copy()
    cx, cy = center
    points_arr[:, 0] = (points_arr[:, 0] - cx) * scale + cx
    points_arr[:, 1] = (points_arr[:, 1] - cy) * scale + cy
    return points_arr


def rotate_points(points: np.ndarray, angle: float, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    points_arr = np.asarray(points, dtype=np.float32).copy()
    cx, cy = center
    x = points_arr[:, 0] - cx
    y = points_arr[:, 1] - cy
    c = np.cos(angle)
    s = np.sin(angle)
    points_arr[:, 0] = x * c - y * s + cx
    points_arr[:, 1] = x * s + y * c + cy
    return points_arr


def angle_difference(current: float, target: float) -> float:
    diff = (target - current + np.pi) % (2.0 * np.pi) - np.pi
    return diff
