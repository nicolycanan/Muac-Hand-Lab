import numpy as np

from config import PARTICLE_COUNT

from effects.effect_manager import EffectManager
from particles.particle_system import ParticleSystem
from particles.shapes import generate_heart_targets


def test_generate_heart_targets_count():
    points = generate_heart_targets(1500)
    assert points.shape == (1500, 2)
    assert np.isfinite(points).all()
    assert np.abs(points).max() < 5.0


def test_particle_system_update_approaches_target():
    targets = generate_heart_targets(80)
    system = ParticleSystem(count=80, initial_targets=targets.copy())
    system.positions = np.zeros_like(targets)
    system.velocities = np.zeros_like(targets)

    start_distance = np.linalg.norm(system.targets, axis=1).mean()
    for _ in range(200):
        system.update(0.016)

    end_distance = np.linalg.norm(system.positions - system.targets, axis=1).mean()
    assert end_distance < start_distance


def test_effect_manager_starts_with_heart_enabled():
    manager = EffectManager(start_with_heart=True)
    assert manager.heart_enabled is True
    assert manager.heart.system.positions.shape == (PARTICLE_COUNT, 2)


def test_particle_system_draw_changes_frame():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    system = ParticleSystem(count=8)
    system.positions = np.array([[50, 50], [80, 80], [110, 110], [140, 140], [170, 170], [200, 200], [230, 230], [260, 260]], dtype=np.float32)
    system.draw(frame)
    assert frame.any() > 0


def test_set_targets_preserves_particle_state_and_read_only_base_geometry():
    manager = EffectManager(start_with_heart=True)
    system = manager.heart.system
    base = system.base_targets.copy()
    old_targets = system.targets.copy()
    system.velocities[:] = (2.0, -1.0)
    positions = system.positions.copy()
    velocities = system.velocities.copy()

    manager.set_transform((300, 220), 2.0, np.pi / 2)
    np.testing.assert_array_equal(system.positions, positions)
    np.testing.assert_array_equal(system.velocities, velocities)
    np.testing.assert_array_equal(system.base_targets, base)
    assert not system.base_targets.flags.writeable
    assert not np.array_equal(old_targets, system.targets)
    np.testing.assert_allclose(
        np.linalg.norm(system.targets - manager.transform_center, axis=1),
        np.linalg.norm(base, axis=1) * 240,
        atol=5e-5,
    )


def test_mixed_particle_effects_stay_finite_and_return_to_latest_targets():
    system = ParticleSystem(2000, generate_heart_targets(2000) * 100 + (320, 240))
    system.scatter()
    system.set_targets(system.targets + (10, -20))
    targets = system.targets.copy()
    system.break_apart(np.array([320, 240]))
    for _ in range(12):
        system.apply_local_repulsion((320, 240), dt=0.033)
        system.update(0.033)
    for _ in range(240):
        system.update(0.033)
    assert not system.exploding
    assert all(np.isfinite(a).all() for a in (system.positions, system.velocities, system.targets, system.colors))
    np.testing.assert_array_equal(system.targets, targets)
    assert np.linalg.norm(system.positions - targets, axis=1).mean() < 0.1


def test_free_cloud_drifts_and_then_forms_heart():
    from core.gesture_controller import GestureController
    from test_gestures import _make_hand

    manager = EffectManager()
    controller = GestureController(hold_time=0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    system = manager.heart.system
    positions = system.positions.copy()
    targets = system.targets.copy()

    manager.update(frame, controller, 1 / 30)
    assert manager.current_shape is None
    assert manager.interaction_mode == "FREE"
    assert frame.any()
    assert np.linalg.norm(system.positions - positions, axis=1).mean() > 0.1
    assert ((system.positions >= 0) & (system.positions <= (639, 479))).all()

    initial_distance = np.linalg.norm(system.positions - targets, axis=1).mean()
    controller.update([_make_hand()], 1 / 30)
    manager.update(frame, controller, 1 / 30)
    controller.update([], 1 / 30)
    for _ in range(100):
        manager.update(frame, controller, 1 / 30)
    assert manager.current_shape == "HEART"
    assert np.linalg.norm(system.positions - targets, axis=1).mean() < initial_distance * 0.01
    np.testing.assert_array_equal(system.targets, targets)
