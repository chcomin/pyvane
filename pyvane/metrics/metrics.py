"""Blood vessel morphometry analysis.

Edge geometry can be measured in two ways, which write the same edge
attributes:
- "voxel": the voxel path of the skeleton and the 3D distance transform
  (``assign_edge_radius``, ``assign_centerline_radii``);
- "smoothed": paths smoothed along their length, leaf tips extended to the label surface and
  in-plane radii (``pyvane.metrics.smoothed``). Less biased for length, tortuosity and
  radius, in particular in labels that the axial blur makes taller than wide.

``measure_edges`` records the method in ``graph.graph["geometry"]``, and the graph metrics
(``vessel_density``, ``tortuosity``) use the matching edge geometry unless told otherwise:
the voxel ``path`` for "voxel" (and for graphs that were not measured), the smoothed
``points`` for "smoothed".
"""

import networkx as nx
import numpy as np
import scipy.ndimage as ndi

from pyvane.metrics import smoothed
from pyvane.metrics.tortuosity import tortuosity as tort_func
from pyvane.util.misc import gaussian_filter_with_anchors

GEOMETRIES = ("voxel", "smoothed")


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

def _get_geometry(graph: nx.MultiGraph, geometry: str | None = None) -> str:
    """The edge geometry to use: ``geometry`` if given, otherwise the one recorded by
    ``measure_edges`` in ``graph.graph["geometry"]``, otherwise "voxel"."""

    geometry = geometry if geometry is not None else graph.graph.get("geometry", "voxel")
    if geometry not in GEOMETRIES:
        raise ValueError(f'Unknown geometry "{geometry}"; use one of {GEOMETRIES}.')
    return geometry

def _edge_polyline_attr(geometry: str) -> str:
    """Edge attribute holding the polyline of the given geometry."""

    return "points" if geometry == "smoothed" else "path"

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
        # Points with non-integer coordinates (smoothed paths) take their nearest voxel
        voxels = np.round(np.asarray(path)).astype(int)
        is_inside_roi = img_roi[tuple(voxels.T)]==1
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

def total_length(graph, pix_size, img_roi=None, geometry: str | None = None):
    """Calculates the total arc-length of all edges in the graph.

    Args:
        graph: The input graph. Edges must have a ``path`` attribute (and ``points`` for
            the "smoothed" geometry).
        pix_size: Physical size of each pixel per dimension.
        img_roi: Optional binary mask. Only path segments inside the mask are counted.
        geometry: "voxel" (edge ``path``) or "smoothed" (edge ``points``, see
            ``measure_edges``). Defaults to ``graph.graph["geometry"]``, or "voxel".

    Returns:
        Total arc-length summed over all edges.
    """

    attr = _edge_polyline_attr(_get_geometry(graph, geometry))
    paths = nx.get_edge_attributes(graph, attr).values()
    if len(paths) != graph.number_of_edges():
        raise ValueError(f'Every edge needs a "{attr}" attribute; see measure_edges.')
    total_length = 0
    for path in paths:
        total_length += path_length(path, pix_size, img_roi)

    return total_length

def num_branch_points(graph, img_roi=None):
    """Counts the number of branching points (nodes with degree >= 3) in the graph.

    Nodes flagged as probable crossings of two vessels (node attribute ``crossing``, set by
    the "scaled" graph creation) are not branching points.

    Args:
        graph: The input graph. Nodes must have a ``center`` attribute.
        img_roi: Optional binary mask. Only nodes whose centre lies inside the mask
            are counted.

    Returns:
        Number of branching points.
    """

    degrees = dict(graph.degree())
    degree_three_nodes = [(node, degree) for node, degree in degrees.items()
                          if degree >= 3 and not graph.nodes[node].get("crossing", False)]
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
        scale_factor: float = 1.0,
        geometry: str | None = None,
        ):
    """Calculates blood vessel density as total vessel length divided by image volume.

    Args:
        graph: The input graph.
        img_shape: Size of the image per dimension.
        pix_size: Physical size of each pixel per dimension. Defaults to the graph's
            ``pix_size`` attribute, or ones if not set.
        img_roi: Optional binary mask. Only vessels inside the mask are included.
        scale_factor: Conversion constant for the desired physical unit.
        geometry: Edge geometry for the length, "voxel" or "smoothed" (see
            ``total_length``). Defaults to ``graph.graph["geometry"]``, or "voxel".

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

    length = total_length(graph, pix_size, img_roi, geometry)*scale_factor
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
        use_valid: bool = True,
        geometry: str | None = None,
        ):
    """Calculates the mean tortuosity of blood vessels in the graph.

    This is a local tortuosity at the given scale, not the
    per-edge ``tortuosity`` attribute (arc length / end-to-end distance) written by
    ``measure_edges``.

    Args:
        graph: The input graph. Edges must have a ``path`` attribute (and ``points`` for
            the "smoothed" geometry).
        scale: Length scale for tortuosity computation. Smaller values capture fine
            local curvature; larger values reflect only broad directional changes.
        pix_size: Physical size of each pixel per dimension. Defaults to the graph's
            ``pix_size`` attribute, or ones if not set.
        use_valid: If True, path regions near terminations and bifurcations are
            excluded from the calculation.
        geometry: Edge geometry, "voxel" (edge ``path``) or "smoothed" (edge ``points``).
            Defaults to ``graph.graph["geometry"]``, or "voxel".

    Returns:
        Mean tortuosity across all sufficiently long edges.
    """

    pix_size = _get_pix_size(graph, graph.graph["img_shape"], pix_size)

    length_threshold = scale/2**0.5

    return tort_func(graph, scale, length_threshold, graph_reduction_func=avg_func, 
                     path_reduction_func=avg_func,
                     pix_size=pix_size, use_valid=use_valid,
                     path_attr=_edge_polyline_attr(_get_geometry(graph, geometry)))
def measure_edges(
        graph: nx.MultiGraph, 
        bin_img: np.ndarray, 
        method: str = "voxel", 
        pix_size: tuple | np.ndarray | None = None,
        smooth_sigma: float = 1.0,
        ):
    """Measures the geometry of every edge, in place, with the "voxel" or "smoothed" method.

    Both methods write the same edge attributes, in physical units (for "voxel", radii use
    the mean of ``pix_size``, as an isotropic image is assumed; for "smoothed", the in-plane
    pixel size):
    - ``mean_radius``, ``min_radius``, ``max_radius``: radius of the vessel away from the
      junctions;
    - ``radii``: radius at each voxel of the edge ``path``;
    - ``points``: (n, 3) polyline of the vessel, in voxels, from the center of min(u, v) to
      the center of max(u, v);
    - ``measured_length``: arc length of ``points``;
    - ``tortuosity``: ``measured_length`` / end-to-end distance (NaN for self-loops).

    Methods:
    - "voxel": radii from the 3D distance transform (``assign_edge_radius`` and
      ``assign_centerline_radii``, which also writes ``outer_length``); ``points`` is the
      voxel path with the node centers.
    - "smoothed": radii from the in-plane distance transform (not inflated by the axial
      blur); ``points`` is smoothed along the path, with leaf tips extended to the label
      surface (``pyvane.metrics.smoothed``). Needs a 3D image.

    The method is recorded in ``graph.graph["geometry"]``, which ``vessel_density`` and
    ``tortuosity`` use by default.

    Args:
        graph: Graph from ``create_graph`` (either method).
        bin_img: The binary image the graph was made from.
        method: "voxel" or "smoothed".
        pix_size: Physical size of each pixel per dimension. Defaults to the graph's
            ``pix_size`` attribute, or ones if not set.
        smooth_sigma: Smoothing of the "voxel" ``radii`` (see ``assign_centerline_radii``).
    """

    if method not in GEOMETRIES:
        raise ValueError(f'Unknown method "{method}"; use one of {GEOMETRIES}.')
    pix_size = _get_pix_size(graph, bin_img.shape, pix_size)

    if method == "smoothed":
        smoothed.measure_edges(graph, bin_img, pix_size=pix_size)
    else:
        assign_edge_radius(graph, bin_img, pix_size)
        assign_centerline_radii(graph, bin_img, smooth_sigma=smooth_sigma, pix_size=pix_size)
        for u, v, key, data in graph.edges(keys=True, data=True):
            points = smoothed.edge_centerline(graph, u, v, key)
            length = smoothed.path_length(points, pix_size)
            end_to_end = smoothed.path_length(points[[0, -1]], pix_size)
            data["points"] = points
            data["measured_length"] = length
            data["tortuosity"] = (length / end_to_end if u != v and end_to_end > 1e-9
                                  else float("nan"))
    graph.graph["geometry"] = method
