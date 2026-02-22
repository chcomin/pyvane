"""Image class used for storing image data."""

from pathlib import Path

import numpy as np
import scipy.ndimage as ndi


class Image:
    """Class representing a 2D or 3D image. The pixel data is saved in attribute `data`. The class
    also contains some basic methods for changing the range and type of the values.

    The image can be initialized with a disk location path. This information is used only as
    metadata.

    Attributes:
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

    def __init__(
            self, 
            data: np.ndarray, 
            path: str | Path | None = None, 
            pix_size: tuple[float, ...] | None = None
            ):
        """Args:
        data: A numpy array containing the pixel values. Can be 2D or 3D.
        path: The location in the system of the image file.
        pix_size: The physical size of the pixels or voxels.
        """

        if not isinstance(data, np.ndarray):
            raise TypeError("Parameter `data` must be a numpy array")

        if path is not None and not isinstance(path, (str, Path)):
            raise TypeError("Parameter `path` must be of type str or Path")

        if pix_size is None:
            pix_size = (1.,)*data.ndim

        filename = None
        if path is not None:
            filename = Path(path).stem

        self.data = data
        self.path = path
        self.pix_size = pix_size
        self.filename = filename

    def get_range(self):
        """Get minimum and maximum intensities as a tuple."""

        return self.data.min(), self.data.max()

    def copy(self):
        """Copy the image to a new image. Attributes `or_vmax` and `or_vmin` are also copied
        from the original image.

        Returns:
        -------
        img : Image
            The new image.
        """

        return Image(self.data, self.path, self.pix_size)

    def to_isotropic(self):
        """Return a new image with isotropic pixel size. """

        pix_size = self.pix_size
        zoom_ratio = pix_size/np.min(pix_size)
        new_data = ndi.interpolation.zoom(self.data, zoom_ratio, order=2)
        new_pix_size = (np.min(pix_size),)*new_data.ndim
        return Image(new_data, self.path, new_pix_size)


    def get_info(self):
        """Return string containing information about the image.

        Returns:
        -------
        repr_str : str
            String containing information about the image.
        """

        data = self.data
        
        repr_str =  (
            f"{data.shape} {data.dtype} array with range [{data.min()},{data.max()}]\n"
            f"Pixel size: {self.pix_size}\n"
            f"Path: {self.path}\n"
        )

        return repr_str

    def __repr__(self):
        return self.get_info() + self.data.__repr__()

    def __str__(self):
        return self.get_info() + self.data.__str__()
    
    def __array__(self, dtype=None, copy=None):
        return self.data.__array__(dtype, copy)

    def __getitem__(self, index):
        return self.data[index]

    def __getattr__(self, name):
        return getattr(self.data, name)