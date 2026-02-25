"""Wrapper for scikit-image skeletonization function."""

import numpy as np
from skimage.morphology import skeletonize as skel

# TODO: Add soft skeletonization on GPU

def skeletonize(img: np.ndarray) -> np.ndarray:
    """Skeletonizes a binary image using Lee's algorithm.

    Args:
        img: Input binary image (2D or 3D).

    Returns:
        Skeletonized binary image with the same dtype as the input.
    """
    return skel(img).astype(img.dtype)

