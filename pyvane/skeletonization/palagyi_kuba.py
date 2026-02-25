"""Wrapper function around compiled library for skeletonization."""

import ctypes as ct
from pathlib import Path

import numpy as np

from pyvane.util.image import Image

try:
    libskeleton = ct.CDLL(Path(__file__).parent/"libskeleton.so")
except Exception:
    print("Could not load skeleton library. Unable to calculate skeletons.")

def skeletonize(img_bin, num_threads=1, verbosity=0):
    """Generates the skeleton of a binary image using the Palàgyi-Kuba algorithm.

    Supports 2D and 3D images. See reference [1] for algorithmic details.

    Args:
        img_bin: Binary Image object with values 0 and 1.
        num_threads: Number of threads for skeleton computation.
        verbosity: Verbosity level. 0 = silent; 1 = prints iteration index; values > 1
            save an intermediate skeleton image every ``verbosity`` iterations.

    Returns:
        Binary Image object containing the skeleton.

    References:
        [1] Palàgyi, K. and Kuba, A. (1998). A 3D 6-subiteration thinning algorithm
        for extracting medial lines. Pattern Recognition Letters 19, 613-627.
    """

    if tuple(img_bin.unique()) != (0, 1):
        raise ValueError("Image must only have values 0 and 1")

    if img_bin.ndim==2:
        img_data_2d = img_bin.data
        img_data = np.zeros((3, img_data_2d.shape[0], img_data_2d.shape[1]))
        img_data[1] = img_data_2d
    else:
        img_data = img_bin.data

    num_threads = int(num_threads)

    img_data = np.ascontiguousarray(img_data, dtype=np.uint16)

    size_z, size_x, size_y = img_data.shape
    size_z, size_x, size_y = int(size_z), int(size_x), int(size_y)

    img_data_res = np.zeros([size_z, size_x, size_y], dtype=np.uint16)

    libskeleton.skel_interface(img_data.ctypes.data_as(ct.POINTER(ct.c_ushort)),
                      img_data_res.ctypes.data_as(ct.POINTER(ct.c_ushort)),
                      size_z, size_x, size_y, num_threads, verbosity)

    if img_bin.ndim==2:
        img_data_res = img_data_res[1]

    img_res = Image(img_data_res.astype(np.uint8), img_bin.path, pix_size=img_bin.pix_size)

    return img_res