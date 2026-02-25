"""Blood vessel segmentation."""

import numpy as np
import scipy.ndimage as ndi

from pyvane.util.image import Image
from pyvane.util.misc import remove_small_comp


def vessel_segmentation(img, threshold, sigma=None, radius=40, comp_size=500, hole_size=None):
    """Segments blood vessels using adaptive thresholding.

    For each pixel, marks it as a vessel candidate when
    ``img[pixel] - avg(img[window]) > threshold``, where ``window`` is a Gaussian-weighted
    region centred on the pixel. Small connected components are removed afterwards.

    Args:
        img: Input image (2D or 3D). Accepts an Image object or a plain ndarray.
        threshold: Pixels with values above the local average by more than this value
            are classified as vessels.
        sigma: Gaussian standard deviations for pre-smoothing, in physical units. If
            None, unit sigma is used for each dimension.
        radius: Controls the Gaussian averaging window; the standard deviation of the
            averaging Gaussian is ``radius / 2`` (in pixels).
        comp_size: Connected components smaller than this number of pixels are removed.
        hole_size: For 2D images, background (hole) components smaller than this are
            filled. Not applied to 3D images.

    Returns:
        Binary image with segmented vessels. Returns an ndarray if the input was an
        ndarray, otherwise returns an Image object.
    """

    ndim = img.ndim

    if sigma is None:
        sigma = [1.]*ndim
    sigma = np.array(sigma)

    return_array = False
    if isinstance(img, np.ndarray):
        img_data = img
        pix_size = (1.,)*img.ndim
        img_path = None
        return_array = True
    else:
        img_data = img.data
        pix_size = img.pix_size
        img_path = img.path

    if img_data.dtype!=float:
        img_data = img_data.astype(float)

    pix_size = np.array(pix_size)
    img_data_diffused = ndi.gaussian_filter(img_data, sigma=sigma/pix_size)

    if ndim==2:
        img_final = _vessel_segmentation_2d(img_data_diffused, threshold, radius, comp_size, 
                                            hole_size)
    elif ndim==3:
        img_final = _vessel_segmentation_3d(img_data_diffused, threshold, radius, comp_size)

    if return_array:
        return img_final

    return Image(img_final.astype(np.uint8), img_path, pix_size=pix_size)

def _vessel_segmentation_2d(img_data, threshold, radius=40, comp_size=500, hole_size=None):
    """Blood vessel segmentation of a 2D image. See function `vessel_segmentation` for details."""

    if img_data.dtype!=float:
        img_data = img_data.astype(float)

    img_bin = adaptive_thresholding(img_data, threshold, radius)
    img_no_small_comp = remove_small_comp(img_bin, comp_size)
    if hole_size is None:
        img_final = img_no_small_comp
    else:
        img_no_small_hole = remove_small_comp(1-img_no_small_comp, hole_size)
        img_final = 1 - img_no_small_hole

    return img_final

def _vessel_segmentation_3d(img_data, threshold, radius=40, comp_size=500):
    """Blood vessel segmentation of a 3D image. See function `vessel_segmentation` for details."""

    if img_data.dtype!=float:
        img_data = img_data.astype(float)

    img_bin = np.zeros_like(img_data, dtype=np.uint8)
    for idx in range(img_data.shape[0]):
        img_bin[idx] = adaptive_thresholding(img_data[idx], threshold, radius)

    img_bin_comp = remove_small_comp(img_bin, comp_size)

    img_lab, num_comp = ndi.label(1-img_bin_comp)
    tam_comp = ndi.sum(1-img_bin_comp, labels=img_lab, index=range(1,num_comp+1))
    ind_background = np.argmax(tam_comp) + 1
    img_final = img_lab != ind_background

    return img_final

def adaptive_thresholding(img_data, threshold, radius):
    """Applies adaptive thresholding to segment a bright object on a dark background.

    For each pixel, marks it as foreground when
    ``img_data[pixel] - avg(img_data[window]) > threshold``, where the window average
    is computed with a Gaussian filter of standard deviation ``radius / 2``.

    Args:
        img_data: 2D input image to threshold.
        threshold: Pixels above the local average by more than this value are foreground.
        radius: Controls the averaging window; standard deviation equals ``radius / 2``.

    Returns:
        Binary boolean array where True indicates foreground pixels.
    """

    if img_data.dtype!=float:
        img_data = img_data.astype(float)

    img_blurred = ndi.gaussian_filter(img_data, sigma=radius/2.)
    img_corr = img_data - img_blurred
    img_bin = img_corr > threshold

    return img_bin