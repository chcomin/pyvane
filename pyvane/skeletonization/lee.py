"""Skeletonization with Lee's thinning scheme."""

import numpy as np
from skimage.morphology import skeletonize as skel

# TODO: Add soft skeletonization on GPU


def skeletonize(img: np.ndarray, method: str = "skimage", elongation: float = 1.0
                ) -> np.ndarray:
    """Skeletonizes a binary image using Lee's algorithm.

    Args:
        img: Input binary image (2D or 3D).
        method: "skimage" uses scikit-image's implementation. In 3D it does not preserve the
            free ends of branches whose width is even (in voxels): such branches are shortened
            back to their junction or removed, and isolated objects of even width disappear.
            "fixed" uses ``pyvane.skeletonization.thinning``, the same thinning scheme with the
            end points preserved (slower, about 2x).
        elongation: "fixed" method, 3D only: how many times taller than wide the label is
            along z; the label is thinned as if its z coordinates were divided by it (see
            ``pyvane.skeletonization.thinning``).

    Returns:
        Skeletonized binary image with the same dtype as the input.
    """
    if method == "skimage":
        if elongation != 1.0:
            raise ValueError('elongation needs method="fixed"')
        return skel(img).astype(img.dtype)
    if method == "fixed":
        from pyvane.skeletonization.thinning import skeletonize as thin
        return thin(img, elongation).astype(img.dtype)
    raise ValueError(f"unknown method {method!r}")
