"""Blood vessel morphometry analysis."""

import networkx as nx
import numpy as np
import scipy.ndimage as ndi

from pyvane.graph.util import get_euclidean_trim_indices
from pyvane.metrics.tortuosity import tortuosity as tort_func
from pyvane.util.misc import gaussian_filter_with_anchors


def path_length(path, pix_size, img_roi=None):
    """Calculate the arc-length of a parametric sequence of pixels.

    Parameters
    ----------
    path : list of tuple
        A sequence of pixels representing a path/curve.
    pix_size : tuple of float
        The physical size that a pixel represents, can be a different value for each axis.
    img_roi : ndarray
        A region of interest image. Only paths where `img_roi` has value 1 will be considered.

    Returns:
    -------
    path_length : float
        The total length of the path.
    """

    dpath = np.diff(path, axis=0)*pix_size
    dlengths = np.sqrt(np.sum(dpath**2, axis=1))

    if img_roi is not None:
        is_inside_roi = img_roi[tuple(zip(*path))]==1
        dlengths = dlengths[is_inside_roi[1:]]

    path_length = np.sum(dlengths)

    return path_length

def xy_roi_to_volume(img_roi, num_planes):
    """Convert 2D image to 3D by copying the same image `num_planes` times.

    Parameters
    ----------
    img_roi : ndarray
        Image to be copied.
    num_planes : int
        Size of the first dimension of the output image.

    Returns:
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

    Returns:
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

    Returns:
    -------
    measured_values : list of float
        The calculated values.
    """

    img_roi = labeled_roi(img_roi)
    if to_volume:
        num_planes = graph.graph["shape"][0]
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

def img_volume(
        pix_size: tuple | np.ndarray, 
        img_shape: tuple | np.ndarray, 
        img_roi: np.ndarray | None = None, 
        scale_factor: float = 1.0
        ):
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

    Returns:
    -------
    volume : float
        The volume of the image or region of interest.
    """

    pix_size = np.array(pix_size)
    img_shape = np.array(img_shape)

    if img_roi is None:
        volume = np.prod(img_shape*pix_size*scale_factor)
    else:
        volume = np.sum(img_roi)*np.prod(pix_size*scale_factor)

    return volume

def total_length(graph, pix_size, img_roi=None):
    """Calculate the total length of the edges in the graph. The graph must contain
    attribute 'pix_size' (the physical size of the pixels) and its edges must contain
    attribute 'path' (the path represented by each edge).

    Parameters
    ----------
    graph : networkx.MultiGraph
        The input graph.
    img_roi : ndarray
        Binary array. Only edges where `img_roi` has value 1 will be used in the calculation.

    Returns:
    -------
    total_length : float
        The total length.
    """

    paths = nx.get_edge_attributes(graph, "path").values()
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

    Returns:
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
        nodes_center = [tuple(graph.nodes[node]["center"]) for node in bif_nodes]
        is_inside_roi = img_roi[tuple(zip(*nodes_center))]
        bif_nodes = filter(lambda item: item[1], zip(bif_nodes, is_inside_roi))

    return len(list(bif_nodes))

def vessel_density(
        graph: nx.MultiGraph, 
        img_shape: tuple | np.ndarray, 
        pix_size: tuple | np.ndarray | None = None, 
        img_roi: np.ndarray | None = None, 
        scale_factor: float = 1.0
        ):
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

    Returns:
    -------
    density : float
        The blood vessel density in the image.
    """

    pix_size = pix_size if pix_size is not None else graph.graph.get("pix_size", None)
    if pix_size is None:
        pix_size = np.ones(len(img_shape), dtype=float)

    if len(pix_size) != len(img_shape):
        raise ValueError("The length of pix_size must match the length of img_shape.")
    
    if img_roi is not None and img_roi.shape != img_shape:
        raise ValueError("The shape of img_roi must match the shape of img_shape.")

    length = total_length(graph, pix_size, img_roi)*scale_factor
    volume = img_volume(pix_size, img_shape, img_roi, scale_factor)
    density = length/volume

    return density

def branch_point_density(
        graph: nx.MultiGraph, 
        img_shape: tuple | np.ndarray, 
        pix_size: tuple | np.ndarray | None = None,
        img_roi: np.ndarray | None = None, 
        scale_factor: float = 1.0
        ):
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

    Returns:
    -------
    bp_density : float
        The branching point density in the image.
    """

    pix_size = pix_size if pix_size is not None else graph.graph.get("pix_size", None)
    if pix_size is None:
        pix_size = np.ones(len(img_shape), dtype=float)

    if len(pix_size) != len(img_shape):
        raise ValueError("The length of pix_size must match the length of img_shape.")
    
    if img_roi is not None and img_roi.shape != img_shape:
        raise ValueError("The shape of img_roi must match the shape of img_shape.")

    num_bp = num_branch_points(graph, img_roi)
    volume = img_volume(pix_size, img_shape, img_roi, scale_factor)
    bp_density = num_bp/volume

    return bp_density

def assign_edge_radius(graph, bin_img):
    """
    Calculates physiological metrics for every edge.
    Volumes are extracted from the 3D watershed partition.
    Radii are strictly sampled from the 1D centerline to prevent underestimation.
    """

    edt_bg = ndi.distance_transform_edt(bin_img)
    
    for u, v, key, data in graph.edges(keys=True, data=True):
        
        # Isolate the Outer Path for Radii and Length calculations
        path = graph[u][v][key]['path']
        path_len = len(path)
        
        # Enforce min -> max ordering to align with the path array
        node_s, node_e = (u, v) if u < v else (v, u)
        
        # Radii of junctions (0 if not a junction)
        r_s = graph.nodes[node_s]['radius'] if graph.degree(node_s) > 2 else 0.0
        r_e = graph.nodes[node_e]['radius'] if graph.degree(node_e) > 2 else 0.0

        trim_s, trim_e = graph[u][v][key]['trim_amount']
        
        # Calculate Radii from the Centerline ONLY
        if trim_s + trim_e >= path_len:
            # Swallowed by junctions; sample the midpoint
            if path_len > 0:
                mid_idx = path_len // 2
                val = float(edt_bg[tuple(path[mid_idx])])
            else:
                val = float((r_s + r_e) / 2.0)
            outer_mean, outer_min, outer_max = val, val, val
        else:
            outer_path = path[trim_s : path_len - trim_e]
            edt_vals = edt_bg[tuple(outer_path.T)]
            
            outer_mean = float(np.mean(edt_vals))
            outer_min = float(np.min(edt_vals))
            outer_max = float(np.max(edt_vals))
            
        graph[u][v][key]['mean_radius'] = outer_mean
        graph[u][v][key]['max_radius'] = outer_max
        graph[u][v][key]['min_radius'] = outer_min
        
        # Calculate True Outer Length
        full_length = graph[u][v][key]['length']
        outer_length = max(0.0, float(full_length - r_s - r_e))
        graph[u][v][key]['outer_length'] = outer_length
        

def assign_centerline_radii(graph, bin_img, smooth_sigma=1.0):
    """
    Extracts, corrects, and assigns a 1D array of radii to the centerline pixels.
    Fixes junction inflation by clamping internal segments, and fixes grid 
    staircasing using a fast 1D Gaussian filter.
    """

    edt_bg = ndi.distance_transform_edt(bin_img)

    for u, v, _, data in graph.edges(keys=True, data=True):
        path = data['path']
        path_len = len(path)
        
        if path_len == 0:
            data['radii'] = np.array([], dtype=np.float32)
            continue
            
        # Direct EDT Sampling
        # Extract the raw radius at every pixel along the path
        radii = edt_bg[tuple(path.T)].astype(np.float32)
        
        # Junction Inflation Correction (The Clamping Strategy)
        trim_s, trim_e = data['trim_amount']

        # Define anchor points that must remain unchanged during smoothing
        anchors = [trim_s, path_len - trim_e - 1]
        if trim_s > 0:
            anchors.append(0)
        if trim_e > 0:
            anchors.append(path_len - 1)

        radii_s = gaussian_filter_with_anchors(radii, anchors, sigma=smooth_sigma)
            
        # Assign to the edge
        data['radii'] = radii_s

def tortuosity(
        graph: nx.MultiGraph, 
        scale: float, 
        pix_size: tuple | np.ndarray | None = None,
        use_valid: bool = True
        ):
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

    Returns:
    -------
    float
        The tortuosity of the blood vessels.
    """

    pix_size = pix_size if pix_size is not None else graph.graph.get("pix_size", None)
    if pix_size is None:
        pix_size = np.ones(len(graph.graph["img_shape"]), dtype=float)

    length_threshold = scale/2**0.5
    def f(x): return sum(x)/len(x)
    avg_func = f

    return tort_func(graph, scale, length_threshold, graph_reduction_func=avg_func, 
                     path_reduction_func=avg_func,
                     pix_size=pix_size, use_valid=use_valid)