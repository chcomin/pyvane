"""Label measurements of pyvane.graph.label_geometry."""

import numpy as np
import pytest

from pyvane.graph.label_geometry import label_elongation_no_skeleton


def _tubes_along_x(elongation, radius=4, shape=(80, 40, 90)):
    """Two tubes along x, ``elongation`` times taller than wide."""
    z, y, x = np.indices(shape).astype(float)
    xc = np.clip(x, 10, 80)
    img = np.zeros(shape, bool)
    for z0, y0 in ((25, 12), (55, 28)):
        img |= ((z - z0) / elongation) ** 2 + (y - y0) ** 2 + (x - xc) ** 2 <= radius ** 2
    return img


@pytest.mark.parametrize("elongation", [1, 2, 4])
def test_elongation_of_tubes_along_x(elongation):
    measured = label_elongation_no_skeleton(_tubes_along_x(elongation))
    assert measured == pytest.approx(elongation, rel=0.2)


def test_a_vertical_vessel_barely_changes_it():
    img = _tubes_along_x(4)
    z, y, x = np.indices(img.shape)
    img |= (y - 20) ** 2 + (x - 45) ** 2 <= 16      # vessel along z through the whole stack
    assert label_elongation_no_skeleton(img) == pytest.approx(4, rel=0.2)


def test_no_usable_run_gives_one():
    img = np.zeros((20, 20, 20), bool)
    img[5:15, 9:11, 2:18] = True                     # thinner than the minimum half-width
    assert label_elongation_no_skeleton(img) == 1.0
