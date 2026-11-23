import numpy as np
import pytest
from scipy import ndimage as ndi

from pyvane.skeletonization import lee
from pyvane.skeletonization.thinning import skeletonize

S26 = np.ones((3, 3, 3))


@pytest.mark.parametrize("w", [1, 2, 3, 4, 5, 6])
def test_bars_of_any_width_keep_a_full_line(w):
    b = np.zeros((w + 6, w + 6, 40), bool)
    b[3:3 + w, 3:3 + w, 5:35] = True
    s = skeletonize(b)
    assert ndi.label(s, S26)[1] == 1
    xs = np.nonzero(s)[2]
    assert xs.max() - xs.min() >= 30 - w - 2


@pytest.mark.parametrize("r,center", [(2.0, 7.5), (3.3, 7.5), (2.0, 7.0), (3.3, 7.0)])
def test_capsules_are_not_removed(r, center):
    z, y, x = np.mgrid[:16, :16, :70]
    xc = np.clip(x, 10, 54)
    cap = np.sqrt((z - center) ** 2 + (y - center) ** 2 + (x - xc) ** 2) <= r
    s = skeletonize(cap)
    xs = np.nonzero(s)[2]
    assert ndi.label(s, S26)[1] == 1
    assert xs.min() <= 10 + r and xs.max() >= 54 - r


def test_topology_is_preserved():
    z, y, x = np.mgrid[:9, :40, :40]
    rr = np.sqrt((y - 20) ** 2 + (x - 20) ** 2)
    ring = (np.abs(rr - 12) <= 2.5) & (np.abs(z - 4) <= 2.5)
    s = skeletonize(ring)
    assert ndi.label(s, S26)[1] == 1
    # the skeleton is a closed loop: removing any voxel keeps it connected
    pts = np.argwhere(s)
    t = s.copy()
    t[tuple(pts[0])] = False
    assert ndi.label(t, S26)[1] == 1
    z, y, x = np.mgrid[:21, :21, :21]
    d = np.sqrt((z - 10) ** 2 + (y - 10) ** 2 + (x - 10) ** 2)
    shell = (d <= 8) & (d >= 5)
    s = skeletonize(shell)
    assert ndi.label(~s)[1] == 2  # the cavity is kept


def test_skeleton_is_thin():
    rng = np.random.default_rng(0)
    vol = ndi.gaussian_filter(rng.random((30, 60, 60)), 3) > 0.52
    s = skeletonize(vol)
    blocks = ndi.convolve(s.astype(np.uint8), np.ones((2, 2, 2), np.uint8), mode="constant")
    assert (blocks == 8).sum() == 0
    assert ndi.label(s, S26)[1] == ndi.label(vol, S26)[1]


def test_lee_wrapper_methods():
    b = np.zeros((8, 8, 30), np.uint8)
    b[3:5, 3:5, 5:25] = 1
    assert lee.skeletonize(b, method="fixed").sum() >= 18
    with pytest.raises(ValueError):
        lee.skeletonize(b, method="other")


def _tall_tube(z_center, elongation=4, shape=(48, 20, 80)):
    """A tube along x (x 12-68 on its axis, radius 3), ``elongation`` times taller than wide."""
    z, y, x = np.indices(shape).astype(float)
    xc = np.clip(x, 12, 68)
    return ((z - z_center) / elongation) ** 2 + (y - 10) ** 2 + (x - xc) ** 2 <= 9


def test_elongation_puts_the_ends_of_a_tall_tube_on_its_axis():
    tube = _tall_tube(24)
    z, _, x = np.nonzero(skeletonize(tube))
    assert np.abs(z - 24).max() >= 6 and x.min() >= 18   # without: climbs, stops short
    s = skeletonize(tube, elongation=4)
    assert ndi.label(s, S26)[1] == 1
    z, _, x = np.nonzero(s)
    assert np.abs(z - 24).max() <= 2                  # no climb on the tall end caps
    assert x.min() <= 13 and x.max() >= 67            # the ends reach the end of the axis


def test_elongation_keeps_stacked_vessels_apart():
    # two tall tubes along z separated by a 1-voxel gap
    tubes = _tall_tube(11.5, shape=(50, 20, 80)) | _tall_tube(37.5, shape=(50, 20, 80))
    assert ndi.label(tubes, S26)[1] == 2
    s = skeletonize(tubes, elongation=4)
    assert ndi.label(s, S26)[1] == 2


def test_elongation_one_is_the_default_and_2d_ignores_it():
    tube = _tall_tube(24)
    assert np.array_equal(skeletonize(tube, elongation=1.0), skeletonize(tube))
    plane = tube[24]
    assert np.array_equal(skeletonize(plane, elongation=4), skeletonize(plane))
    with pytest.raises(ValueError):
        skeletonize(tube, elongation=0.5)
