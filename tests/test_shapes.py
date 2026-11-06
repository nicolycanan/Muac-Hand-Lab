import numpy as np

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
