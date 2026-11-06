import cv2
import numpy as np

from config import HEART_BASE_SCALE, PARTICLE_COUNT
from effects.particle_heart import ParticleHeartEffect
from effects.effect_manager import EffectManager
from particles.shapes import generate_heart_targets, heart_parametric


def test_heart_parametric_is_finite():
    t = np.linspace(0.0, 2.0 * np.pi, 128)
    x, y = heart_parametric(t)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    assert x.min() < 0.0 < x.max()
    assert y.min() < 0.0 < y.max()


def test_generate_heart_targets_valid_range():
    points = generate_heart_targets(600)
    assert points.shape == (600, 2)
    assert np.abs(points).max() < 10.0


def test_heart_shape_orientation_reference_export(tmp_path):
    heart = ParticleHeartEffect()
    points = heart.base_targets.copy()
    assert points.shape == (PARTICLE_COUNT, 2)
    half = PARTICLE_COUNT // 2
    x, y = heart_parametric(np.array([0.0, np.pi]))
    assert np.allclose(x, 0.0)
    assert y[0] > 0 > y[1]
    np.testing.assert_array_equal(points[:half] * (-1, 1), points[half:])
    top = points[:, 1].min()
    assert points[np.abs(points[:, 0]) < 0.08, 1].min() > top + 0.15
    lobes = points[(np.abs(points[:, 0]) > 0.4) & (points[:, 1] < -0.1)]
    tip = points[(np.abs(points[:, 0]) < 0.1) & (points[:, 1] > 0.8)]
    assert len(lobes) > 0
    assert len(tip) > 0
    assert not np.any((np.abs(points[:, 0]) > 0.6) & (points[:, 1] > 0.5))

    heart.set_transform((256.0, 256.0), 1.0, 0.0)
    expected_targets = heart.base_targets * HEART_BASE_SCALE + np.array([256.0, 256.0])
    assert np.allclose(heart.system.targets, expected_targets)
    heart.system.positions = heart.system.targets.copy()
    canvas = np.full((512, 512, 3), 40, dtype=np.uint8)
    heart.draw(canvas)

    output = tmp_path / "heart-orientation.png"
    ok = cv2.imwrite(str(output), canvas)
    assert ok

    rendered = EffectManager()
    main_targets = rendered.heart.system.targets
    main_center = rendered.transform_center
    main_relative = main_targets - main_center
    assert np.any((np.abs(main_relative[:, 0]) > 40.0) & (main_relative[:, 1] < -10.0))
    assert np.any((np.abs(main_relative[:, 0]) < 10.0) & (main_relative[:, 1] > 70.0))
