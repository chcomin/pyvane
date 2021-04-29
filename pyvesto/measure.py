"""Blood vessel morphometry analysis."""

import networkx as nx
import numpy as np
import scipy.ndimage as ndi
from .graph.adjustment import path_length
from .tortuosity import tortuosity as tort_func

def xy_roi_to_volume(img_roi, num_planes):
    """Convert 2D image to 3D by copying the same image `num_planes` times.

    Parameters
    ----------
    img_roi : ndarray
        Image to be copied.
    num_planes : int
        Size of the first dimension of the output image.

    Returns
    -------
    ndarray
        A 3D image.
    """

    return np.tile(img_roi, (num_planes, 1, 1))

def labeled_roi(img_roi):
    """Return label image containing connected components in `img_roi`.

    Parameters
    ----------
    img_roi : ndarray
        Image containing only values 0 and 1. This usually represents regions of interest
        to be quantified.

    Returns
    -------
    img_roi : ndarray
        The labeled image.
    """

    if img_roi.max()==1:
        img_roi, num_comps = ndi.label(img_roi, np.ones((3,)*img_roi.ndim))

    return img_roi

def measure_rois(graph, img_roi, mea_funcs, to_volume=False, scale_factor=1):
    """Apply a set of measurement functions defined in `mea_funcs` to characterize
    a graph. The characterization can be done at different regions of interest defined
    in the mask image `img_roi`.

    Parameters
    ----------
    graph : networkx.MultiGraph
        The graph to be quantified.
    img_roi : ndarray
        Binary image containing regions of interest for quantification.
    mea_funcs : list of func
        Functions for characterizing the graph. Must have signature func(graph,
        img_roi, scale_factor).
    to_volume : bool
        If False, `img_roi` is used as is. Note that if the graph represents a 3D
        system and `img_roi` is 2D, an error will occur. If True, the 2D ROI is tranformed
        into a 3D image by copying the same image as many times as necessary.
    scale_factor : float
        Conversion constant that can be used for returning the measurement at a specific
        physical scale (m, cm, mm, um, etc).

    Returns
    -------
    measured_values : list of float
        The calculated values.
    """

    pix_size = graph.graph['pix_size']

    img_roi = labeled_roi(img_roi)
    if to_volume:
        num_planes = graph.graph['shape'][0]
        img_roi = xy_roi_to_volume(img_roi, num_planes)

    structure = np.ones(img_roi.ndim*(3,))

    img_label, num_comps = ndi.label(img_roi, structure)      # Connected components
    comp_indices = np.unique(img_label)[1:]

    measured_values = {}
    for comp_idx in comp_indices:
        measured_values[comp_idx] = []
        img_roi_comp = img_label==comp_idx
        for mea_func in mea_funcs:
            value = mea_func(graph, img_roi=img_roi_comp, scale_factor=scale_factor)
            measured_values[comp_idx].append(value)

    return measured_values


def img_volume(pix_size, img_shape, img_roi=None, scale_factor=1):
    """Calculate the volume of an image considering the physical size of the
    pixels and possible regions of interest.

    Parameters
    ----------
    pix_size : tuple of float
        The physical size of the pixels at each dimension.
    img_shape : tuple of int
        Size of the image at each dimension.
    img_roi : ndarray
        Binary image containing a region of interest.
    scale_factor : float
        Conversion constant that can be used for returning the measurement at a specific
        physical scale (m, cm, mm, um, etc).

    Returns
    -------
    volume : float
        The volume of the image or region of interest.
    """

    pix_size = np.array(pix_size)
    img_shape = np.array(img_shape)

    if img_roi is None:
        volume = np.product(img_shape*pix_size*scale_factor)
    else:
        volume = np.sum(img_roi)*np.product(pix_size*scale_factor)

    return volume

def total_length(graph, img_roi=None):
    """Calculate the total length of the edges in the graph. The graph must contain
    attribute 'pix_size' (the physical size of the pixels) and its edges must contain
    attribute 'path' (the path represented by each edge).

    Parameters
    ----------
    graph : networkx.MultiGraph
        The input graph.
    img_roi : ndarray
        Binary array. Only edges where `img_roi` has value 1 will be used in the calculation.

    Returns
    -------
    total_length : float
        The total length.
    """

    pix_size = graph.graph['pix_size']

    paths = nx.get_edge_attributes(graph, 'path').values()
    total_length = 0
    for path in paths:
        total_length += path_length(path, pix_size, img_roi)

    return total_length


def num_branch_points(graph, img_roi=None):
    """Calculate the number of branching points of the graph. Nodes in the graph must
    contain attribute 'center'.

    Parameters
    ----------
    graph : networkx.MultiGraph
        The input graph.
    img_roi : ndarray
        Binary array. Only edges where `img_roi` has value 1 will be used in the calculation.

    Returns
    -------
    int
        The number of branching points.
    """

    degrees = dict(graph.degree())
    degree_three_nodes = list(filter(lambda item: item[1]>=3, degrees.items()))
    if len(degree_three_nodes)==0:
        return 0

    bif_nodes, _ = zip(*degree_three_nodes)
    if img_roi is not None:
        nodes_center = [tuple(graph.nodes[node]['center']) for node in bif_nodes]
        is_inside_roi = img_roi[tuple(zip(*nodes_center))]
        bif_nodes = filter(lambda item: item[1], zip(bif_nodes, is_inside_roi))

    return len(list(bif_nodes))

def vessel_density(graph, img_shape=None, img_roi=None, scale_factor=1):
    """Calculate the density of blood vessels inside an image represented by `graph`.
    The density is given by the total length of the vessels divided by the image volume.

    Parameters
    ----------
    graph : networkx.MultiGraph
        The input graph.
    img_shape : tuple of int
        The size of the image.
    img_roi : ndarray
        Binary array. Only blood vessels where `img_roi` has value 1 will be used in the calculation.
    scale_factor : float
        Conversion constant that can be used for returning the measurement at a specific
        physical scale (m, cm, mm, um, etc).

    Returns
    -------
    density : float
        The blood vessel density in the image.
    """

    pix_size = graph.graph['pix_size']

    length = total_length(graph, img_roi)*scale_factor
    volume = img_volume(pix_size, img_shape, img_roi, scale_factor)
    density = length/volume

    return density

def branch_point_density(graph, img_shape=None, img_roi=None, scale_factor=1):
    """Calculate the density of blood vessels branching points inside an image represented by `graph`.
    The density is given by the number of branching points divided by the image volume.

    Parameters
    ----------
    graph : networkx.MultiGraph
        The input graph.
    img_shape : tuple of int
        The size of the image.
    img_roi : ndarray
        Binary array. Only blood vessels where `img_roi` has value 1 will be used in the calculation.
    scale_factor : float
        Conversion constant that can be used for returning the measurement at a specific
        physical scale (m, cm, mm, um, etc).

    Returns
    -------
    bp_density : float
        The branching point density in the image.
    """

    pix_size = graph.graph['pix_size']

    num_bp = num_branch_points(graph, img_roi)
    volume = img_volume(pix_size, img_shape, img_roi, scale_factor)
    bp_density = num_bp/volume

    return bp_density

def tortuosity(graph, scale, use_valid=True):
    """Calculate the tortuosity of the blood vessels represented by `graph`.

    Parameters
    ----------
    graph : networkx.MultiGraph
        The input graph.
    scale : float
        The scale at which the tortuosity will be calculated. That is, smaller values indicate
        that the tortuosity should be calculated for local changes in direction of the blood vessels,
        while larger values indicate that small changes should be ignored and only large variations
        should be taken into account.
    use_valid : bool
        If True, regions close to terminations and bifurcations of blood vessels are not considered
        in the calculation. This ignored region becomes larger with the `scale` parameter.

    Returns
    -------
    float
        The tortuosity of the blood vessels.
    """

    pix_size = graph.graph['pix_size']
    length_threshold = scale/2**0.5
    avg_func = lambda x: sum(x)/len(x)

    return tort_func(graph, scale, length_threshold, graph_reduction_func=avg_func, path_reduction_func=avg_func,
                     pix_size=pix_size, use_valid=use_valid)