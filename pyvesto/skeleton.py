"""Wrapper function around compiled library for skeletonization."""

from pathlib import Path
import ctypes as ct
import numpy as np
from .image import Image

try:
    libskeleton = ct.CDLL(Path(__file__).parent/'libskeleton.so')
except Exception:
    print('Could not load skeleton library. Unable to calculate skeletons.')

def skeletonize(img_bin, num_threads=1, verbosity=0):
    """Generate the skeleton of binary image `img_bin` using the method published in [1]. The
    input image can be 2D or 3D.

    Parameters
    ----------
    img_bin : Image
        Binary image. Must have only values 0 and 1.
    num_threads : int
        Number of threads to use for calculating the skeleton.
    verbosity : int
        Verbosity level of the method. If 0, nothing is printed. If 1, the current iteration
        index is printed. If larger than 1, saves an image with name temp.tif containing the
        current skeleton each `verbosity` iterations. In some systems and terminals the values
        might not be printed.

    Returns
    -------
    img_res : Image
        A binary image containing the skeleton.

    [1] Palàgyi, K. and Kuba, A. (1998). A 3D 6-subiteration thinning algorithm for
    extracting medial lines. Pattern Recognition Letters 19, 613–627.
    """

    if tuple(img_bin.unique()) != (0, 1):
        raise ValueError('Image must only have values 0 and 1')

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