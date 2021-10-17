"""Utility functions for reading 3D images."""

from xml.etree import cElementTree as etree
import oiffile as oif
import tifffile
import numpy as np
import czifile
from .image import Image

def read_img(path, channel=None, read_pix_size=True):
    """Read image from the disk. Supported formats are {'tif', 'tiff', 'oib', 'lsm', 'czi'}. The function
    also tries to find the pixel size of the image if `read_pix_size=True`, but may fail since this
    information is not standardized for most file formats.

    Parameters
    ----------
    channel : int, optional
        Image channel to read. If None, all channels are read.
    read_pix_size : bool
        Return pixel size of the image, if available in the file.

    Returns
    -------
    img : Image
        Image read.
    """

    file_type = str(path).split('.')[-1]

    if (file_type == 'tif') or (file_type == 'tiff'):
        reader_func = read_tiff
    elif file_type == 'oib':
        reader_func = read_oib
    elif file_type == 'lsm':
        reader_func = read_lsm
    elif file_type == 'czi':
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
        #print('Warning, image is 2D')
        #img_data = img_data[None]      # Add one dimension
        pass
    elif len(shape)==3:
        min_dim = np.min(shape)
        if min_dim<=4:
            print('Warning, image seems to be 2D with colors')
    elif len(shape)==4:                  # 3D color image
        pass
    else:
        raise ValueError(f'Image has unrecognized shape: {shape}')
    
    if channel is not None:
        img_data = img_data[channel]

    img = Image(img_data, path, pix_size=pix_size)

    return img

def read_tiff(path, return_pix_size=False):
    """Read tiff image.

    Parameters
    ----------
    path : str
        Location of the image.
    return_pix_size : bool
        Return pixel size of the image, if available in the file. Reading the scale might fail
        because each software saves the scale in a different format, thus this information is not
        standardized.

    Returns
    -------
    img_data : ndarray
        The image read.
    pix_size : tuple of float
        The physical size of each pixel.
    """

    tiff_data = tifffile.TiffFile(path)
    img_data = tiff_data.asarray()

    if not return_pix_size:
        return img_data

    pix_size = find_pix_size_tiff(tiff_data)
    pix_size = pix_size[-img_data.ndim:]
    return img_data, pix_size

def read_oib(path, return_pix_size=False):
    """Read oib image.

    Parameters
    ----------
    path : str
        Location of the image.
    return_pix_size : bool
        Return pixel size of the image, if available in the file. Reading the scale might fail
        because each software saves the scale in a different format, thus this information is not
        standardized.

    Returns
    -------
    img_data : ndarray
        The image read.
    pix_size : tuple of float
        The physical size of each pixel.
    """

    oib_data = oif.OifFile(path)
    img_data = oib_data.asarray()

    if not return_pix_size:
        return img_data

    img_info = data.mainfile['Reference Image Parameter']
    piz_size_z = 1.
    piz_size_x = img_info['HeightConvertValue']
    piz_size_y = img_info['WidthConvertValue']

    pix_size = (piz_size_z, piz_size_x, piz_size_y)
    print('Warning, assuming image depth equal to 1')

    return img_data, pix_size

def read_lsm(path, return_pix_size=False):
    """Read lsm image.

    Parameters
    ----------
    path : str
        Location of the image.
    return_pix_size : bool
        Return pixel size of the image, if available in the file. Reading the scale might fail
        because each software saves the scale in a different format, thus this information is not
        standardized.

    Returns
    -------
    img_data : ndarray
        The image read.
    pix_size : tuple of float
        The physical size of each pixel.
    """

    lsm_data = tifffile.TIFFfile(path)
    img_data = lsm_data.asarray(series=0)

    if not return_pix_size:
        return img_data

    spacing = lsm_data.pages[0].cz_lsm_scan_info['line_spacing']
    pix_size_x = pix_size_y = float(spacing)
    pix_size_z = lsm_data.pages[0].cz_lsm_scan_info['plane_spacing']

    pix_size = (pix_size_z, pix_size_x, pix_size_y)

    return img_data, pix_size

def read_czi(path, return_pix_size=False):
    """Read tif image.

    Parameters
    ----------
    path : str
        Location of the image.
    return_pix_size : bool
        Return pixel size of the image, if available in the file. Reading the scale might fail
        because each software saves the scale in a different format, thus this information is not
        standardized.

    Returns
    -------
    img_data : ndarray
        The image read.
    pix_size : tuple of float
        The physical size of each pixel.
    """

    czi_data = czifile.CziFile(path)
    img_data = czi_data.asarray()

    if not return_pix_size:
        return img_data

    pix_size = find_pix_size_czi(czi_data)

    return img_data, pix_size


def find_pix_size_tiff(tiff_data):
    """Find pixel size in the metadata of a tiff file.

    Parameters
    ----------
    tiff_data : tifffile.TiffFile
        The file to search. Object returned by reading an image using tifffile.TiffFile(...)

    Returns
    -------
    tuple of float
        The pixel size.
    """

    try:
        pix_size_z = 1.
        num_char = 21
        imagej_tags = tiff_data.pages[0].imagej_tags
        if 'spacing' in imagej_tags:
            pix_size_z = imagej_tags['spacing']
        if ('info' in imagej_tags):
            img_info=data.pages[0].imagej_tags['info']
            k1 = img_info.find('HeightConvertValue')
            if k1!=-1:
                aux = img_info[k1+num_char:k1+num_char+10]
                k2 = aux.find('\n')
                pix_size_x = float(aux[:k2])
                k1 = img_info.find('WidthConvertValue')
                aux = img_info[k1+num_char-1:k1+num_char+10-1]
                k2 = aux.find('\n')
                pix_size_y = float(aux[:k2])
            else:
                pix_size_x, pix_size_y = -1, -1

        else:
            p = data.pages[0]
            v = p.tags['x_resolution'].value
            pix_size_x = v[1]/float(v[0])
            v = p.tags['y_resolution'].value
            pix_size_y = v[1]/float(v[0])
    except  Exception:
        pix_size_z=pix_size_x=pix_size_y=1.

    return (pix_size_z, pix_size_x, pix_size_y)

def find_pix_size_czi(czi_data):
    """Find pixel size in the metadata of a czi file.

    Parameters
    ----------
    czi_data : czifile.CziFile.
        The file to search. Object returned by reading an image using czifile.CziFile(...)

    Returns
    -------
    tuple of float
        The pixel size.
    """

    metadata = czi_data.metadata(True)
    if isinstance(metadata, str):
        metadata = etree.fromstring(metadata)

    scaling = metadata.find('.//Scaling')
    if scaling is None:
        raise ValueError('Pixel size information is not available.')

    pix_size = []
    for idx, axis in enumerate(["Z", "X", "Y"]):
        axis_tag = scaling.find(f'.//Distance[@Id="{axis}"]')
        if axis_tag is None:
            raise ValueError(f'Pixel size for axis {axis} is not available.')
        try:
            scaling_value = float(axis_tag.find('Value').text)
        except Exception:
            raise ValueError(f'Pixel size for axis {axis} is not available.')

        if axis_tag.find('DefaultUnitFormat').text != u'\xb5m':
            print('Warning, pixel size unit is not microns')
        else:
            pix_size.append(scaling_value)

    pix_size = tuple([1e6*item for item in pix_size])

    return pix_size

def _find_pix_size_czi_backup(czi_data):
    """Find pixel size in the metadata of a czi file. Function `find_pix_size_czi` should be used
    instead of this one. This is only for backup purposes.

    Parameters
    ----------
    czi_data : czifile.CziFile.
        The file to search. Object returned by reading an image using czifile.CziFile(...)

    Returns
    -------
    tuple of float
        The pixel size.
    """

    metadata = czi_data.metadata(False)
    metadata = metadata['ImageDocument']['Metadata']

    try:
        hs_v = metadata['HardwareSetting']['ParameterCollection'][1]['ImagePixelDistances']['value']
    except KeyError:
        hs_v = None
    try:
        is_v = metadata['ImageScaling']['ImagePixelSize']
    except KeyError:
        is_v = None
    try:
        s_v = metadata['Scaling']['AutoScaling']['CameraPixelDistance']
    except KeyError:
        s_v = None

    if (hs_v is None) and (is_v is None) and (s_v is None):
        raise ValueError('Could not find scaling for X and Y')
    else:
        # The three values should be the same, if they exist
        if hs_v is not None:
            for v in [is_v, s_v]:
                if hs_v!=v:
                    raise ValueError(f'Scaling values differ: {hs_v} vs {v}')

            x_pix_size, y_pix_size = list(map(float, hs_v.split(',')))

    # Warning! Not sure if this is the correct tag for Z scaling
    try:
        z_pix_size = metadata['Information']['Image']['Dimensions']['Z']['Positions']['Interval']['Increment']
    except KeyError:
        raise ValueError('Could not find scaling for Z')

    scaling = (z_pix_size, x_pix_size, y_pix_size)

    return scaling