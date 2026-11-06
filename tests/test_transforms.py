import numpy as np

from particles.transforms import angle_difference, rotate_points, scale_points, translate_points


def test_translate_and_scale():
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    translated = translate_points(points, 2.0, 3.0)
    scaled = scale_points(points, 2.0, center=(0.0, 0.0))
    assert np.allclose(translated, np.array([[2.0, 3.0], [3.0, 3.0]], dtype=np.float32))
    assert np.allclose(scaled, np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))


def test_rotation_and_angle_difference():
    points = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rotated = rotate_points(points, np.pi / 2.0, center=(0.0, 0.0))
    assert np.allclose(rotated[0], np.array([0.0, 1.0]), atol=1e-5)
    assert np.isclose(angle_difference(np.pi - 0.1, -np.pi + 0.1), 0.2, atol=1e-5)
