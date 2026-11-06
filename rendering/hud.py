"""Desenha sobre o vídeo o painel de estado e o diagnóstico de gestos."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from config import GRAB_RADIUS, REPEL_RADIUS
from core.app_state import AppState
from core.gesture_controller import GestureController, InteractionMode, RawGesture

if TYPE_CHECKING:
    from effects.effect_manager import EffectManager


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
    summary = [
        "MUAC / MAGIC HAND LAB",
        f"MODE: {mode}",
        f"SHAPE: {shape or 'NONE'}",
        f"HANDS: {state.hands_detected} / 2",
        f"PARTICLES: {state.particle_count}",
        f"FPS: {state.fps:.1f}",
    ]
    if shape is None:
        summary.append("Levante o indicador para formar")
    if gestures.candidate == RawGesture.INDEX_RAISED and not gestures.triggered and gestures.confirmed == RawGesture.IDLE:
        progress = int(100 * gestures.pending.elapsed / gestures.hold_time)
        summary.append(f"Confirmando HEART: {progress}%")
    _draw_panel(frame, 12, 270, summary)

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


def draw_hand_controls(frame: np.ndarray, gestures: GestureController, effects: EffectManager) -> None:
    hands = [gestures.hands[key] for key in gestures.confirmation.hands]
    if effects.transform_active:
        first, second = (tuple(np.rint(hand.pinch_point).astype(int)) for hand in hands)
        color = (245, 215, 95)
        cv2.line(frame, first, second, color, 2, cv2.LINE_AA)
        for point in (first, second):
            cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)
        center = tuple((a + b) // 2 for a, b in zip(first, second))
        cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
        degrees = (np.rad2deg(effects.transform_rotation) + 180) % 360 - 180
        label = f"MOVER / ESCALAR / GIRAR  {effects.transform_scale:.2f}x  {degrees:+.0f} graus"
        x = min(max(center[0] + 9, 10), max(frame.shape[1] - 285, 10))
        cv2.putText(frame, label, (x, max(center[1] - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1, cv2.LINE_AA)
    elif effects.interaction_mode in (InteractionMode.REPEL, InteractionMode.GRAB):
        for hand in hands:
            repel = effects.interaction_mode == InteractionMode.REPEL
            point = hand.palm_center if repel else hand.pinch_point
            radius = REPEL_RADIUS if repel else GRAB_RADIUS
            color = (90, 110, 240) if repel else (105, 190, 110)
            center = tuple(np.rint(point).astype(int))
            cv2.circle(frame, center, int(radius), color, 1, cv2.LINE_AA)
            cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
