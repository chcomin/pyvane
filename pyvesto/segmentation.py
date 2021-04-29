"""Blood vessel segmentation."""

import scipy.ndimage as ndi
import numpy as np
from .image import Image
from . import util

def vessel_segmentation(img, threshold, sigma=None, radius=40, comp_size=500, hole_size=None):
    """Blood vessel segmentation using adaptive thresholding. In short terms, for each
    pixel, if img[pixel]-avg(img[window])>threshold the pixel is marked as blood vessel,
    where window is a region centered at the pixel. The function also removes connected
    components smaller than `comp_size`.

    Parameters
    ----------
    img : Image
        Image to be segmented. Can be 2D or 3D.
    threshold : float
        Pixels with values larger than avg(img[window])+threshold are blood vessel candidates, where
        window is a region centered at the pixel.
    sigma : list of float, optional
        Gaussian standard deviations for smoothing the image before thresholding. The values should
        be given as physical units (e.g., micrometers). If None, unitary values are used.
    radius : int
        Window size to use for intensity averaging. Since a Gaussian is used, this is actually
        2x the standard deviation of the Gaussian used for averaging pixel intensities. Note
        that this Gaussian is different than the one defined by parameter `sigma`. The value
        is in pixels.
    comp_size : int
        Connected components smaller than `comp_size` are removed from the image.

    Returns
    -------
    Image
        A binary image containing segmented blood vessels.
    """

    ndim = img.ndim

    if sigma is None:
        sigma = [1.]*ndim
    sigma = np.array(sigma)

    img_data = img.data

    if img_data.dtype!=np.float:
        img_data = img_data.astype(np.float)

    pix_size = np.array(img.pix_size)
    img_data_diffused = ndi.gaussian_filter(img_data, sigma=sigma/pix_size)

    if ndim==2:
        img_final = _vessel_segmentation_2d(img_data_diffused, threshold, radius, comp_size, hole_size)
    elif ndim==3:
        img_final = _vessel_segmentation_3d(img_data_diffused, threshold, radius, comp_size)

    return Image(img_final.astype(np.uint8), img.path, pix_size=img.pix_size)

def _vessel_segmentation_2d(img_data, threshold, radius=40, comp_size=500, hole_size=None):
    """Blood vessel segmentation of a 2D image. See function `vessel_segmentation` for details.
    """

    if img_data.dtype!=np.float:
        img_data = img_data.astype(np.float)

    img_bin = adaptive_thresholding(img_data, threshold, radius)
    img_no_small_comp = util.remove_small_comp(img_bin, comp_size)
    if hole_size is None:
        img_final = img_no_small_comp
    else:
        img_no_small_hole = util.remove_small_comp(1-img_no_small_comp, hole_size)
        img_final = 1 - img_no_small_hole

    return img_final

def _vessel_segmentation_3d(img_data, threshold, radius=40, comp_size=500):
    """Blood vessel segmentation of a 3D image. See function `vessel_segmentation` for details.
    """

    if img_data.dtype!=np.float:
        img_data = img_data.astype(np.float)

    img_bin = np.zeros_like(img_data, dtype=np.uint8)
    for idx in range(img_data.shape[0]):
        img_bin[idx] = adaptive_thresholding(img_data[idx], threshold, radius)

    img_bin_comp = util.remove_small_comp(img_bin, comp_size)

    img_lab, num_comp = ndi.label(1-img_bin_comp)
    tam_comp = ndi.sum(1-img_bin_comp, labels=img_lab, index=range(1,num_comp+1))
    ind_background = np.argmax(tam_comp) + 1
    img_final = img_lab != ind_background

    return img_final

def adaptive_thresholding(img_data, threshold, radius):
    """Segmentation using adaptive thresholding of a bright object on a dark background. In short
    terms, for each pixel, if img_data[pixel]-avg(img_data[window])>threshold the pixel is marked
    as belonging to the object, where window is a region centered at the pixel.

    Parameters
    ----------
    img_data : ndarray
        Image to be thresholded. Must be 2D.
    threshold : float
        Threshold to decide if a pixel belongs to the object.
    radius : int
        Window size to use for intensity averaging. Since a Gaussian is used, this is actually
        2x the standard deviation of the Gaussian used for averaging pixel intensities.

    Returns
    -------
    img_bin : ndarray
        The resulting binary image.
    """

    if img_data.dtype!=np.float:
        img_data = img_data.astype(np.float)

    img_blurred = ndi.gaussian_filter(img_data, sigma=radius/2.)
    img_corr = img_data - img_blurred
    img_bin = img_corr > threshold

    return img_bin