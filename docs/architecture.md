# MUAC Magic Hand Lab architecture

The application is intentionally modular:

- `core/` handles the webcam and MediaPipe integration.
- `particles/` contains the geometric and physical model for heart particles.
- `effects/` manages effect selection and effect-specific state.
- `rendering/` samples the HUD and overlay logic.
- `tests/` validates particle math and gesture heuristics without a camera.

The Particle Heart is generated from a mathematical heart curve and then transformed through translation, scale, and rotation while preserving a stable local target set.
