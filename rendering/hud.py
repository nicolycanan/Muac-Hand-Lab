"""Desenha sobre o vídeo o painel de estado e o diagnóstico de gestos."""

from __future__ import annotations

import cv2
import numpy as np

from core.app_state import AppState
from core.gesture_controller import GestureController, InteractionMode


def _draw_panel(frame: np.ndarray, x: int, width: int, lines: list[str]) -> None:
    top = 12
    bottom = top + 14 + 20 * len(lines)
    cv2.rectangle(frame, (x, top), (x + width, bottom), (20, 20, 20), -1)
    cv2.rectangle(frame, (x, top), (x + width, bottom), (255, 120, 140), 1)
    for index, line in enumerate(lines):
        y = top + 21 + index * 20
        cv2.putText(frame, line, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    state: AppState,
    gestures: GestureController,
    mode: InteractionMode,
    shape: str | None,
) -> None:
    height, width = frame.shape[:2]
    _draw_panel(frame, 12, 240, [
        "MUAC / MAGIC HAND LAB",
        f"MODE: {mode}",
        f"SHAPE: {shape or 'NONE'}",
        f"HANDS: {state.hands_detected} / 2",
        f"PARTICLES: {state.particle_count}",
        f"FPS: {state.fps:.1f}",
    ])

    if state.show_debug:
        labels = dict.fromkeys(("LEFT", "RIGHT")) | gestures.hands
        lines = []
        for label, hand in labels.items():
            if hand is None:
                lines.append(f"{label}: --")
                continue
            fingers = sum(hand.fingers.values())
            lines.append(f"{label}: {hand.raw} | {fingers} fingers")
            lines.append(f"  PINCH GAP / PALM: {hand.pinch_ratio:.2f}")
        lines.extend([
            f"RAW: {gestures.raw}",
            f"CANDIDATE: {gestures.candidate}",
            f"CONFIRMED: {gestures.confirmed}",
            f"INTERACTION: {mode}",
            f"ACTION: {gestures.action}",
            f"CAMERA: {state.camera_resolution}",
            f"DISPLAY: {state.display_resolution}",
            f"TRACKER: {'ON' if state.tracker_active else 'OFF'}",
        ])
        _draw_panel(frame, width - 312, 300, lines)

    if state.show_landmarks:
        cv2.putText(frame, "LANDMARKS ON", (28, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
