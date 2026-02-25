"""Image class used for storing image data."""

from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi


class Image:
    """Class representing a 2D or 3D image. The pixel data is saved in attribute `data`.

    The image can be initialized with a disk location path. This information is used only
    as metadata.

    Attributes:
        data: Matrix/tensor containing the pixel values.
        path: The location in the system of the image file.
        pix_size: The physical size of the pixels or voxels.
        filename: The stem of the file name corresponding to the image.
        ndim: The number of dimensions of the image.
        shape: The size of each image axis.
    """

    def __init__(
            self, 
            data: np.ndarray, 
            path: str | Path | None = None, 
            pix_size: tuple[float, ...] | np.ndarray | None = None
            ):
        """Initializes the Image.

        Args:
            data: Array containing the pixel values. Can be 2D or 3D.
            path: The location in the system of the image file.
            pix_size: The physical size of the pixels or voxels. If None, defaults to
                unit size in all dimensions.
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

    def get_range(self) -> tuple[Any, Any]:
        """Returns the minimum and maximum pixel intensities.

        Returns:
            A tuple (min_value, max_value).
        """

        return self.data.min(), self.data.max()

    def copy(self) -> "Image":
        """Creates a copy of the image.

        Returns:
            A new Image instance with the same data, path, and pixel size.
        """

        return Image(self.data, self.path, self.pix_size)

    def to_isotropic(self) -> "Image":
        """Returns a new image resampled to have isotropic pixel size.

        The image is zoomed along each axis so that all pixel dimensions become equal
        to the minimum pixel size of the original image.

        Returns:
            A new Image with isotropic voxel spacing.
        """

        pix_size = self.pix_size
        zoom_ratio = pix_size/np.min(pix_size)
        new_data = ndi.interpolation.zoom(self.data, zoom_ratio, order=2)
        new_pix_size = (np.min(pix_size),)*new_data.ndim
        return Image(new_data, self.path, new_pix_size)


    def get_info(self) -> str:
        """Returns a string containing information about the image.

        Returns:
            A formatted string with the image shape, dtype, value range, pixel size,
            and file path.
        """

        data = self.data
        
        repr_str =  (
            f"{data.shape} {data.dtype} array with range [{data.min()},{data.max()}]\n"
            f"Pixel size: {self.pix_size}\n"
            f"Path: {self.path}\n"
        )

        return repr_str

    def __repr__(self) -> str:
        return self.get_info() + self.data.__repr__()

    def __str__(self) -> str:
        return self.get_info() + self.data.__str__()
    
    def __array__(self, dtype: np.typing.DTypeLike = None, copy: bool | None = None) -> np.ndarray:
        return self.data.__array__(dtype, copy)

    def __getitem__(self, index: Any) -> Any:
        return self.data[index]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.data, name)