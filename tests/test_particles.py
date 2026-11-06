import numpy as np

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
    manager = EffectManager()
    assert manager.heart_enabled is True
    assert manager.heart.system.positions.shape == (1500, 2)


def test_particle_system_draw_changes_frame():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    system = ParticleSystem(count=8)
    system.positions = np.array([[50, 50], [80, 80], [110, 110], [140, 140], [170, 170], [200, 200], [230, 230], [260, 260]], dtype=np.float32)
    system.draw(frame)
    assert frame.any() > 0
