# Gesture notes

The gesture recognition layer is intentionally lightweight and deterministic:

- Open hand: several extended fingers create a consistent score for repelling particles.
- Fist: strong bend scores trigger the explosion state.
- Pinch: thumb-index proximity triggers a local particle drag or transform mode when two pinch gestures occur simultaneously.
- Recognized gestures require a short time window before becoming active.

This keeps the implementation testable without depending on a live webcam during unit tests.
