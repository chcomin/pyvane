"""Utility functions for reading 3D images."""

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as etree

import numpy as np

# Try to import the required libraries for reading images.
try: 
    import tifffile
    _tifffile_available = True
except ImportError:
    _tifffile_available = False

try: 
    import oiffile
    _oiffile_available = True
except ImportError:
    _oiffile_available = False

try:
    import czifile
    _czifile_available = True
except ImportError:
    _czifile_available = False

from .image import Image


def read_img(
        path: str | Path,
        channel: int | None = None,
        read_pix_size: bool = True,
        ) -> Image:
    """Reads an image from disk.

    Supported formats: 'tif', 'tiff', 'oib', 'lsm', 'czi'. The function also
    attempts to read the pixel size from file metadata if `read_pix_size=True`,
    but this may fail since the format is not standardized across software.

    Args:
        path: Path to the image file.
        channel: Image channel to read. If None, all channels are read.
        read_pix_size: If True, attempts to read the pixel size from file metadata.

    Returns:
        The image read from disk.
    """

    file_type = str(path).split(".")[-1]

    if (file_type == "tif") or (file_type == "tiff"):
        reader_func = read_tiff
    elif file_type == "oib":
        reader_func = read_oib
    elif file_type == "lsm":
        reader_func = read_lsm
    elif file_type == "czi":
        reader_func = read_czi
    else:
        raise RuntimeError(f"Image type '{file_type}' is not supported by the image reader."
            " Current supported extensions: 'tif/tiff', 'oib', 'lsm', 'czi'.")

    img_data = reader_func(path, read_pix_size)
    if read_pix_size:
        img_data, pix_size = img_data

    img_data = img_data.squeeze()
    shape = img_data.shape
    if not read_pix_size:
        pix_size = len(shape)*[1.]

    pix_size = np.array(pix_size)

    if len(shape)==2:
        pass
    elif len(shape)==3:
        min_dim = np.min(shape)
        if min_dim<=4:
            print("Warning, image seems to be 2D with colors")
    elif len(shape)==4:                  # 3D color image
        pass
    else:
        raise ValueError(f"Image has unrecognized shape: {shape}")
    
    if channel is not None:
        img_data = img_data[channel]

    img = Image(img_data, path, pix_size=pix_size)

    return img

def read_tiff(
        path: str | Path,
        return_pix_size: bool = False,
        ) -> np.ndarray | tuple[np.ndarray, tuple[float, ...]]:
    """Reads a TIFF image from disk.

    Args:
        path: Path to the image file.
        return_pix_size: If True, also returns the pixel size read from the file
            metadata. Reading the scale may fail since each software saves it in
            a different format.

    Returns:
        The image data array, or a tuple (image_data, pix_size) if
        `return_pix_size` is True.
    """

    if not _tifffile_available:
        raise ImportError(
            "The tifffile package is required for reading tiff files but it is not available.")

    tiff_data = tifffile.TiffFile(path)
    img_data = tiff_data.asarray()

    if not return_pix_size:
        return img_data

    pix_size = find_pix_size_tiff(tiff_data)
    pix_size = pix_size[-img_data.ndim:]
    return img_data, pix_size

def read_oib(
        path: str | Path,
        return_pix_size: bool = False,
        ) -> np.ndarray | tuple[np.ndarray, tuple[float, ...]]:
    """Reads an OIB image from disk.

    Args:
        path: Path to the image file.
        return_pix_size: If True, also returns the pixel size read from the file
            metadata. Reading the scale may fail since each software saves it in
            a different format.

    Returns:
        The image data array, or a tuple (image_data, pix_size) if
        `return_pix_size` is True.
    """

    if not _oiffile_available:
        raise ImportError(
            "The oiffile package is required for reading oif files but it is not available.")

    oif_data = oiffile.OifFile(path)
    img_data = oif_data.asarray()

    if not return_pix_size:
        return img_data

    img_info = oif_data.mainfile["Reference Image Parameter"]
    pix_size_z = 1.
    pix_size_x = img_info["HeightConvertValue"]
    pix_size_y = img_info["WidthConvertValue"]

    pix_size = (pix_size_z, pix_size_x, pix_size_y)
    print("Warning, assuming image depth equal to 1")

    return img_data, pix_size

def read_lsm(
        path: str | Path,
        return_pix_size: bool = False,
        ) -> np.ndarray | tuple[np.ndarray, tuple[float, ...]]:
    """Reads an LSM image from disk.

    Args:
        path: Path to the image file.
        return_pix_size: If True, also returns the pixel size read from the file
            metadata. Reading the scale may fail since each software saves it in
            a different format.

    Returns:
        The image data array, or a tuple (image_data, pix_size) if
        `return_pix_size` is True.
    """

    if not _tifffile_available:
        raise ImportError(
            "The tifffile package is required for reading tiff files but it is not available.")

    lsm_data = tifffile.TIFFfile(path)
    img_data = lsm_data.asarray(series=0)

    if not return_pix_size:
        return img_data

    spacing = lsm_data.pages[0].cz_lsm_scan_info["line_spacing"]
    pix_size_x = pix_size_y = float(spacing)
    pix_size_z = lsm_data.pages[0].cz_lsm_scan_info["plane_spacing"]

    pix_size = (pix_size_z, pix_size_x, pix_size_y)

    return img_data, pix_size

def read_czi(
        path: str | Path,
        return_pix_size: bool = False,
        ) -> np.ndarray | tuple[np.ndarray, tuple[float, ...]]:
    """Reads a CZI image from disk.

    Args:
        path: Path to the image file.
        return_pix_size: If True, also returns the pixel size read from the file
            metadata. Reading the scale may fail since each software saves it in
            a different format.

    Returns:
        The image data array, or a tuple (image_data, pix_size) if
        `return_pix_size` is True.
    """

    if not _czifile_available:
        raise ImportError(
            "The czifile package is required for reading czi files but it is not available.")

    czi_data = czifile.CziFile(path)
    img_data = czi_data.asarray()

    if not return_pix_size:
        return img_data

    pix_size = find_pix_size_czi(czi_data)

    return img_data, pix_size

def find_pix_size_tiff(tiff_data: Any) -> tuple[float, float, float]:
    """Finds the pixel size in the metadata of a TIFF file.

    Args:
        tiff_data: The tiff file object, as returned by `tifffile.TiffFile(...)`.

    Returns:
        A tuple (pix_size_z, pix_size_x, pix_size_y). Returns (1.0, 1.0, 1.0) if
        the pixel size cannot be determined from the metadata.
    """

    try:
        pix_size_z = 1.
        num_char = 21
        try:
            imagej_tags = tiff_data.pages[0].imagej_tags
            if "spacing" in imagej_tags:
                pix_size_z = imagej_tags["spacing"]
            if ("info" in imagej_tags):
                img_info = tiff_data.pages[0].imagej_tags["info"]
                k1 = img_info.find("HeightConvertValue")
                if k1!=-1:
                    aux = img_info[k1+num_char:k1+num_char+10]
                    k2 = aux.find("\n")
                    pix_size_x = float(aux[:k2])
                    k1 = img_info.find("WidthConvertValue")
                    aux = img_info[k1+num_char-1:k1+num_char+10-1]
                    k2 = aux.find("\n")
                    pix_size_y = float(aux[:k2])
                else:
                    pix_size_x, pix_size_y = -1, -1

        except  Exception as e:
            p = tiff_data.pages[0]
            if "x_resolution" in p.tags:
                tag_x = p.tags["x_resolution"].value
                tag_y = p.tags["y_resolution"].value
            elif "XResolution" in p.tags:
                tag_x = p.tags["XResolution"].value
                tag_y = p.tags["YResolution"].value
            else:
                raise ValueError from e
                
            pix_size_x = tag_x[1]/float(tag_x[0])
            pix_size_y = tag_y[1]/float(tag_y[0])
    except  Exception:
        pix_size_z = pix_size_x = pix_size_y = 1.

    return (pix_size_z, pix_size_x, pix_size_y)

def find_pix_size_czi(czi_data: Any) -> tuple[float, ...]:
    """Finds the pixel size in the metadata of a CZI file.

    Args:
        czi_data: The CZI file object, as returned by `czifile.CziFile(...)`.

    Returns:
        A tuple (pix_size_z, pix_size_x, pix_size_y) in micrometers.

    Raises:
        ValueError: If the pixel size information is not found in the metadata.
    """

    metadata = czi_data.metadata(True)
    if isinstance(metadata, str):
        metadata = etree.fromstring(metadata)

    scaling = metadata.find(".//Scaling")
    if scaling is None:
        raise ValueError("Pixel size information is not available.")

    pix_size = []
    for axis in ["Z", "X", "Y"]:
        axis_tag = scaling.find(f'.//Distance[@Id="{axis}"]')
        if axis_tag is None:
            raise ValueError(f"Pixel size for axis {axis} is not available.")
        try:
            scaling_value = float(axis_tag.find("Value").text)
        except Exception as e:
            raise ValueError(f"Pixel size for axis {axis} is not available.") from e

        if axis_tag.find("DefaultUnitFormat").text != "\xb5m":
            print("Warning, pixel size unit is not microns")
        else:
            pix_size.append(scaling_value)

    pix_size = tuple([1e6*item for item in pix_size])

    return pix_size
