# Experiments and extension notes

To add a new effect without modifying the main webcam loop, create a new class under `effects/` and register it in `EffectManager`.

The recommended pattern is:

1. Add a visual effect module.
2. Define an `update(frame, hands, dt)` API and optional `draw(frame)` routine.
3. Register the effect in `EffectManager` or a future preset registry.
4. Expose a keyboard toggle in `MagicHandApp._handle_keys`.

This keeps the main loop stable while allowing more particle presets, custom gestures, JSON settings, or sound-reactive behavior in later versions.
