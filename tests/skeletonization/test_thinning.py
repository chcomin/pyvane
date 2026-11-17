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
