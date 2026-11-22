"""Placement of the leaf tips in labels taller than wide (guided.place_leaf_tips)."""

import numpy as np
import pytest

from pyvane.graph.creation import create_graph
from pyvane.skeletonization import lee

AXIS_Z, AXIS_Y = 20, 20


def _tall_tube(bump_gap):
    """A tube along x (x 10-90 on its axis, radius 3) whose cross-section is 4 times taller
    than wide, with balls on its top and bottom surfaces ``bump_gap`` voxels from its left
    end (None: no balls)."""
    zz, yy, xx = np.indices((41, 40, 100)).astype(float)
    along = np.clip(xx, 10, 90)
    img = ((zz - AXIS_Z) / 4) ** 2 + (yy - AXIS_Y) ** 2 + (xx - along) ** 2 <= 9
    if bump_gap is not None:
        x = 10 + bump_gap
        z_label = np.nonzero(img[:, AXIS_Y, x])[0]
        for z in (z_label[0], z_label[-1]):
            img |= (zz - z) ** 2 + (yy - AXIS_Y) ** 2 + (xx - x) ** 2 <= 4
    return img


@pytest.mark.parametrize("bump_gap", [None, 2, 6])
def test_tips_are_on_the_axis_at_the_vessel_ends(bump_gap):
    img = _tall_tube(bump_gap)
    graph = create_graph(lee.skeletonize(img, method="fixed"), img)
    tips = sorted((graph.nodes[n]["center"] for n in graph.nodes if graph.degree(n) == 1),
                  key=lambda center: center[2])
    assert len(tips) == 2
    label_x = np.nonzero(img.any(axis=(0, 1)))[0]
    for tip, label_end in zip(tips, [label_x[0], label_x[-1]], strict=True):
        assert abs(tip[0] - AXIS_Z) <= 4          # not on the end cap or a bump
        assert abs(tip[2] - label_end) <= 4       # at the vessel end in XY
