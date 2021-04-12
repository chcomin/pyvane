"""Image class used for storing image data."""

from pathlib import Path
import math
import numpy as np
import scipy.ndimage as ndi

class Image:
    """Class representing a 2D or 3D image. The pixel data is saved in attribute `data`. The class
    also contains some basic methods for changing the range and type of the values.

    Parameters
    ----------
    data : array_like
        The matrix/tensor containing the pixel values.
    path : str, optional
        The location in the system of the image file.
    pix_size : tuple of float, optional
        The physical size of the pixels or voxels.

    Attributes
    ----------
    data : ndarray
        Matrix/tensor containing the pixel values.
    path : str
        The location in the system of the image file.
    pix_size : tuple of float
        The physical size of the pixels or voxels.
    filename : str
        The name of the file corresponding to the image.
    ndim : int
        The dimension of the image
    shape : tuple of int
        The size of each image axis.
    or_vmin : Union[int, float]
        The minimum image intensity when the image is first initialized. Does not change after
        transformations.
    or_vmax : Union[int, float]
        The maximum image intensity when the image is first initialized. Does not change after
        transformations.
    """

    def __init__(self, data, path=None, pix_size=None):

        try:
            data = np.array(data)
        except Exception:
            raise TypeError("Parameter `data` must be an array like object")

        try:
            path = str(path)
        except Exception:
            raise TypeError("Parameter `path` must be of type str")

        if pix_size is None:
            pix_size = tuple([1.]*data.ndim)

        filename = Path(path).stem
        or_vmax, or_vmin = data.max(), data.min()

        self.data = data
        self.path = path
        self.pix_size = pix_size
        self.filename = filename
        self.ndim = data.ndim
        self.shape = data.shape
        self.or_vmax = or_vmax
        self.or_vmin = or_vmin

    def get_range(self):
        """Get minimum and maximum intensities as a tuple

        Returns
        -------
        tuple of Union[int, float]
            The minimum and maximum values.
        """

        return (self.data.min(), self.data.max())


    def copy(self):
        """Copy the image to a new image. Attributes `or_vmax` and `or_vmin` are also copied
        from the original image.

        Returns
        -------
        img : Image
            The new image.
        """

        img = Image(self.data, self.path, self.pix_size)
        img.or_vmin = self.or_vmin
        img.or_vmax = self.or_vmax

        return img

    def change_type_range(self, curr_max=255, new_max=1):
        """Change range of the data type (e.g., 2^8 to 2^16), only doing type conversion if
        necessary for the computation.

        This function is useful when you want to change the pixel type of an image but also
        keep relative intensities. For instance, for an np.uint8 image with maximum value 128
        (2**8/2), you may want to convert it to np.uint16 with maximum value 32896 (~2**16/2).
        In this case, you first convert the image to np.uint16 and then call
        change_type_range(curr_max=2**8-1, new_max=2**16-1).

        Parameters
        ----------
        curr_max : Union[int, float]
            The maximum value of the current data type. For instance, it is 255 for uint8 and usually
            1. for np.float.
        new_max : Union[int, float]
            The maximum value of the new data type.
        """

        data = self.data
        if np.isclose(curr_max, 1.):
            new_data = new_max*data
        else:
            new_data = new_max*(data/curr_max)

        self.data = new_data

    def stretch_data_range(self, new_max=255):
        """Pointwise linear transformation [self.data.min(), self.data.max()] -> [0, new_max].

        Parameters
        ----------
        new_max : Union[int, float]
            The new maximum value of the image.
        """

        data = self.data
        vmin_data = data.min()
        new_data = new_max*((data - vmin_data)/(data.max() - vmin_data))

        self.data = new_data

    def make_isotropic(self):

        pix_size = self.pix_size
        zoom_ratio = pix_size/np.min(pix_size)
        self.data = ndi.interpolation.zoom(self.data, zoom_ratio, order=2)
        self.pix_size = (np.min(pix_size),)*self.data.ndim

    def to_uint8(self):
        """Convert data type to np.uint8"""

        self.data = self.data.astype(np.uint8)

    def to_float(self):
        """Convert data type to np.float"""

        self.data = self.data.astype(np.float)

    def unique(self):
        """Return unique values in the image.

        Returns
        -------
        ndarray of Union[int, float]
            Unique values in the image, in increasing order.
        """

        return np.unique(self.data)

    def get_info(self):
        """Return string containing information about the image.

        Returns
        -------
        repr_str : str
            String containing information about the image.
        """

        data = self.data
        shape = data.shape
        repr_str = f'Path: {self.path}\n'
        repr_str += f'Size: {shape}\n'
        repr_str += f'Pixel size: {self.pix_size}\n'
        repr_str += f'Type: {self.data.dtype}\n'
        repr_str += f'Range: [{data.min()},{data.max()}]\n'

        return repr_str

    def __repr__(self):

        return self.get_info() + self.data.__repr__()

    def __str__(self):

        return self.get_info() + self.data.__str__()

if __name__=='__main__':

    # Run some tests
    data = np.random.rand(20, 20)*(2e16-1)
    img = Image(data, 'root/folder/image1.tiff')
    img.change_type_range(2e16-1, 255)
    img.stretch_data_range(255)
    img.to_uint8()

    test_data = 255*(data - data.min())/(data.max() - data.min())
    test_data = test_data.astype(np.uint8)
    np.allclose(test_data, img.data)