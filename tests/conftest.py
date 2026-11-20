"""Shared helpers for the tests: binary images of tubes."""

import numpy as np
import pytest


def draw_tubes(shape, segments, radius):
    """Binary image of tubes of ``radius`` voxels around the segments ((z, y, x), (z, y, x))."""
    zz, yy, xx = np.indices(shape)
    voxels = np.stack([zz, yy, xx], -1).astype(float)
    img = np.zeros(shape, bool)
    for start, end in segments:
        start, end = np.asarray(start, float), np.asarray(end, float)
        axis = end - start
        t = np.clip(((voxels - start) @ axis) / (axis @ axis), 0, 1)
        img |= np.linalg.norm(voxels - (start + t[..., None] * axis), axis=-1) <= radius
    return img


@pytest.fixture
def draw():
    return draw_tubes
