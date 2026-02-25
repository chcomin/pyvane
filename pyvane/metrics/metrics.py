"""Blood vessel morphometry analysis."""

import networkx as nx
import numpy as np
import scipy.ndimage as ndi

from pyvane.metrics.tortuosity import tortuosity as tort_func
from pyvane.util.misc import gaussian_filter_with_anchors


def avg_func(x):
    """Average function."""
    return sum(x)/len(x)

def _get_pix_size(
        graph: nx.MultiGraph, 
        img_shape: tuple | np.ndarray, 
        pix_size: tuple | np.ndarray | None = None
        ) -> np.ndarray:
    """Determines the pixel size to be used for metric calculations.

    Args:
        graph: The input graph. Used to access the graph's ``pix_size`` attribute if
            ``pix_size`` is not provided.
        img_shape: Size of the image per dimension. Used for validating the length of
            ``pix_size`` if provided.
        pix_size: Optional physical size of each pixel per dimension. If None, the graph's
            ``pix_size`` attribute is used if available; otherwise, a default of ones is
            used.

    Returns:
        Numpy array containing the pixel size per dimension.
    """

    pix_size = pix_size if pix_size is not None else graph.graph.get("pix_size", None)
    if pix_size is None:
        pix_size = np.ones(len(img_shape), dtype=float)
    pix_size = np.array(pix_size)

    return pix_size

def path_length(path, pix_size, img_roi=None):
    """Calculates the arc-length of a parametric sequence of pixels.

    Args:
        path: Sequence of coordinate tuples representing a path or curve.
        pix_size: Physical size per pixel, one value per axis.
        img_roi: Optional region-of-interest mask. Only path segments where the mask
            equals 1 are included in the total.

    Returns:
        Total arc-length of the path.
    """

    dpath = np.diff(path, axis=0)*pix_size
    dlengths = np.sqrt(np.sum(dpath**2, axis=1))

    if img_roi is not None:
        is_inside_roi = img_roi[tuple(zip(*path))]==1
        dlengths = dlengths[is_inside_roi[1:]]

    path_length = np.sum(dlengths)

    return path_length

def xy_roi_to_volume(img_roi, num_planes):
    """Converts a 2D image to 3D by tiling it ``num_planes`` times along axis 0.

    Args:
        img_roi: 2D image to tile.
        num_planes: Number of repetitions along the first axis.

    Returns:
        3D array of shape (num_planes, H, W).
    """

    return np.tile(img_roi, (num_planes, 1, 1))

def labeled_roi(img_roi):
    """Returns a label image of connected components in ``img_roi``.

    If the maximum value of ``img_roi`` equals 1, connected components are labelled
    using a full connectivity structure. Otherwise, ``img_roi`` is returned unchanged.

    Args:
        img_roi: Binary image with values 0 and 1.

    Returns:
        Label image where each connected component has a unique integer ID.
    """

    if img_roi.max()==1:
        img_roi, _ = ndi.label(img_roi, np.ones((3,)*img_roi.ndim))

    return img_roi

def measure_rois(graph, img_roi, mea_funcs, to_volume=False, scale_factor=1):
    """Applies measurement functions to characterise the graph within each region of interest.

    Args:
        graph: The graph to quantify.
        img_roi: Binary image containing regions of interest for quantification.
        mea_funcs: List of functions with signature ``func(graph, img_roi, scale_factor)``
            used to characterise the graph.
        to_volume: If False, ``img_roi`` is used as-is. If True, a 2D ROI is promoted to
            3D by replicating it to match the graph's first-axis size.
        scale_factor: Conversion constant for returning measurements at a specific physical
            scale (m, cm, mm, um, etc).

    Returns:
        Dictionary mapping each connected-component index to a list of measured values,
        one per function in ``mea_funcs``.
    """

    img_roi = labeled_roi(img_roi)
    if to_volume:
        num_planes = graph.graph["shape"][0]
        img_roi = xy_roi_to_volume(img_roi, num_planes)

    structure = np.ones(img_roi.ndim*(3,))

    img_label, _ = ndi.label(img_roi, structure)      # Connected components
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
    """Calculates the physical volume of an image or a region of interest.

    Args:
        pix_size: Physical size of each pixel per dimension.
        img_shape: Number of pixels per dimension.
        img_roi: Optional binary mask. If provided, only pixels where the mask equals 1
            are counted.
        scale_factor: Conversion constant for the desired physical unit.

    Returns:
        Volume in physical units.
    """

    pix_size = np.array(pix_size)
    img_shape = np.array(img_shape)

    if img_roi is None:
        volume = np.prod(img_shape*pix_size*scale_factor)
    else:
        volume = np.sum(img_roi)*np.prod(pix_size*scale_factor)

    return volume

def total_length(graph, pix_size, img_roi=None):
    """Calculates the total arc-length of all edges in the graph.

    Args:
        graph: The input graph. Edges must have a ``path`` attribute.
        pix_size: Physical size of each pixel per dimension.
        img_roi: Optional binary mask. Only path segments inside the mask are counted.

    Returns:
        Total arc-length summed over all edges.
    """

    paths = nx.get_edge_attributes(graph, "path").values()
    total_length = 0
    for path in paths:
        total_length += path_length(path, pix_size, img_roi)

    return total_length

def num_branch_points(graph, img_roi=None):
    """Counts the number of branching points (nodes with degree >= 3) in the graph.

    Args:
        graph: The input graph. Nodes must have a ``center`` attribute.
        img_roi: Optional binary mask. Only nodes whose centre lies inside the mask
            are counted.

    Returns:
        Number of branching points.
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
    """Calculates blood vessel density as total vessel length divided by image volume.

    Args:
        graph: The input graph.
        img_shape: Size of the image per dimension.
        pix_size: Physical size of each pixel per dimension. Defaults to the graph's
            ``pix_size`` attribute, or ones if not set.
        img_roi: Optional binary mask. Only vessels inside the mask are included.
        scale_factor: Conversion constant for the desired physical unit.

    Returns:
        Blood vessel density (length / volume).

    Raises:
        ValueError: If ``pix_size`` length does not match ``img_shape``, or if
            ``img_roi`` shape does not match ``img_shape``.
    """

    pix_size = _get_pix_size(graph, img_shape, pix_size)

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
    """Calculates the density of branching points as count divided by image volume.

    Args:
        graph: The input graph.
        img_shape: Size of the image per dimension.
        pix_size: Physical size of each pixel per dimension. Defaults to the graph's
            ``pix_size`` attribute, or ones if not set.
        img_roi: Optional binary mask. Only branching points inside the mask are counted.
        scale_factor: Conversion constant for the desired physical unit.

    Returns:
        Branching-point density (count / volume).

    Raises:
        ValueError: If ``pix_size`` length does not match ``img_shape``, or if
            ``img_roi`` shape does not match ``img_shape``.
    """

    pix_size = _get_pix_size(graph, img_shape, pix_size)

    if len(pix_size) != len(img_shape):
        raise ValueError("The length of pix_size must match the length of img_shape.")
    
    if img_roi is not None and img_roi.shape != img_shape:
        raise ValueError("The shape of img_roi must match the shape of img_shape.")

    num_bp = num_branch_points(graph, img_roi)
    volume = img_volume(pix_size, img_shape, img_roi, scale_factor)
    bp_density = num_bp/volume

    return bp_density

def assign_edge_radius(graph, bin_img, pix_size: tuple | np.ndarray | None = None):
    """Calculates and assigns radius statistics to every edge in the graph.

    Computes the Euclidean distance transform of ``bin_img``, then for each edge samples
    mean, min, and max radius values from the outer path (excluding pixels inside junction
    spheres). Also assigns the outer edge length.

    Args:
        graph: The graph to annotate in-place. Edges must have ``path`` and
            ``trim_amount`` attributes; nodes must have a ``radius`` attribute.
        bin_img: Binary foreground mask used to compute the distance transform.
        pix_size: Physical size of each pixel per dimension. Defaults to the graph's
            ``pix_size`` attribute, or ones if not set.
    """

    pix_size = np.mean(_get_pix_size(graph, bin_img.shape, pix_size))

    edt_bg = ndi.distance_transform_edt(bin_img)
    
    for u, v, key, _ in graph.edges(keys=True, data=True):
        
        # Isolate the Outer Path for Radii and Length calculations
        path = graph[u][v][key]["path"]
        path_len = len(path)
        
        # Enforce min -> max ordering to align with the path array
        node_s, node_e = (u, v) if u < v else (v, u)
        
        # Radii of junctions (0 if not a junction)
        r_s = graph.nodes[node_s]["radius"] if graph.degree(node_s) > 2 else 0.0
        r_e = graph.nodes[node_e]["radius"] if graph.degree(node_e) > 2 else 0.0

        trim_s, trim_e = graph[u][v][key]["trim_amount"]
        
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
            
        graph[u][v][key]["mean_radius"] = outer_mean*pix_size
        graph[u][v][key]["max_radius"] = outer_max*pix_size
        graph[u][v][key]["min_radius"] = outer_min*pix_size
        
        # Calculate True Outer Length
        full_length = graph[u][v][key]["length"]
        outer_length = max(0.0, float(full_length - r_s - r_e))
        graph[u][v][key]["outer_length"] = outer_length 

def assign_centerline_radii(
        graph, 
        bin_img=None, 
        edt_bg=None, 
        smooth_sigma=1.0, 
        pix_size: tuple | np.ndarray | None = None
        ):
    """Extracts and assigns a smoothed 1D radius array to each edge's centerline.

    Samples the Euclidean distance transform at every path pixel, corrects junction
    inflation by clamping internal segments, and applies a Gaussian filter with anchor
    points to reduce staircase artefacts.

    Args:
        graph: The graph to annotate in-place. Edges must have ``path`` and
            ``trim_amount`` attributes.
        bin_img: Binary foreground mask. Used to compute the EDT when ``edt_bg`` is
            not provided.
        edt_bg: Precomputed Euclidean distance transform. Takes precedence over
            ``bin_img`` when provided.
        smooth_sigma: Standard deviation for the 1D Gaussian smoothing filter.
        pix_size: Physical size of each pixel per dimension. Defaults to the graph's
            ``pix_size`` attribute, or ones if not set.

    Raises:
        ValueError: If neither ``bin_img`` nor ``edt_bg`` is provided.
    """

    if bin_img is None and edt_bg is None:
        raise ValueError("Either bin_img or edt_bg must be provided.")

    if edt_bg is None:
        edt_bg = ndi.distance_transform_edt(bin_img)

    pix_size = np.mean(_get_pix_size(graph, edt_bg.shape, pix_size))

    for _, _, _, data in graph.edges(keys=True, data=True):
        path = data["path"]
        path_len = len(path)
        
        if path_len == 0:
            data["radii"] = np.array([], dtype=np.float32)
            continue
            
        # Direct EDT Sampling
        # Extract the raw radius at every pixel along the path
        radii = edt_bg[tuple(path.T)].astype(np.float32)
        
        # Junction Inflation Correction (The Clamping Strategy)
        trim_s, trim_e = data["trim_amount"]

        # Define anchor points that must remain unchanged during smoothing
        anchors = []
        if trim_s + trim_e < path_len:
            anchors.extend([trim_s, path_len - trim_e - 1])
        if trim_s > 0:
            anchors.append(0)
        if trim_e > 0:
            anchors.append(path_len - 1)

        radii_s = gaussian_filter_with_anchors(radii, anchors, sigma=smooth_sigma)
            
        # Assign to the edge
        data["radii"] = radii_s*pix_size

def tortuosity(
        graph: nx.MultiGraph, 
        scale: float, 
        pix_size: tuple | np.ndarray | None = None,
        use_valid: bool = True
        ):
    """Calculates the mean tortuosity of blood vessels in the graph.

    Args:
        graph: The input graph. Edges must have a ``path`` attribute.
        scale: Length scale for tortuosity computation. Smaller values capture fine
            local curvature; larger values reflect only broad directional changes.
        pix_size: Physical size of each pixel per dimension. Defaults to the graph's
            ``pix_size`` attribute, or ones if not set.
        use_valid: If True, path regions near terminations and bifurcations are
            excluded from the calculation.

    Returns:
        Mean tortuosity across all sufficiently long edges.
    """

    pix_size = _get_pix_size(graph, graph.graph["img_shape"], pix_size)

    length_threshold = scale/2**0.5

    return tort_func(graph, scale, length_threshold, graph_reduction_func=avg_func, 
                     path_reduction_func=avg_func,
                     pix_size=pix_size, use_valid=use_valid)