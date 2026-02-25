"""Utility functions for graph processing, including path tracing and arc-length calculations."""

import networkx as nx
import numpy as np
from numba import njit


@njit
def bresenham_3d(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Computes all integer voxel coordinates along the 3D Bresenham line between two points.

    Args:
        p1: Starting point as a 1D array of length 3 in [z, y, x] order.
        p2: Ending point as a 1D array of length 3 in [z, y, x] order.

    Returns:
        Array of shape (N, 3) containing integer voxel coordinates, including both
        endpoints.
    """
    z1, y1, x1 = p1[0], p1[1], p1[2]
    z2, y2, x2 = p2[0], p2[1], p2[2]
    
    dz = abs(z2 - z1)
    dy = abs(y2 - y1)
    dx = abs(x2 - x1)
    
    sz = 1 if z1 < z2 else -1
    sy = 1 if y1 < y2 else -1
    sx = 1 if x1 < x2 else -1
    
    max_len = max(dz, dy, dx) + 1
    line = np.zeros((max_len, 3), dtype=np.int32)
    
    if dx >= dy and dx >= dz:
        err_1 = 2 * dy - dx
        err_2 = 2 * dz - dx
        for i in range(max_len):
            line[i, 0], line[i, 1], line[i, 2] = z1, y1, x1
            if x1 == x2: 
                break
            if err_1 > 0:
                y1 += sy
                err_1 -= 2 * dx
            if err_2 > 0:
                z1 += sz
                err_2 -= 2 * dx
            err_1 += 2 * dy
            err_2 += 2 * dz
            x1 += sx
            
    elif dy >= dx and dy >= dz:
        err_1 = 2 * dx - dy
        err_2 = 2 * dz - dy
        for i in range(max_len):
            line[i, 0], line[i, 1], line[i, 2] = z1, y1, x1
            if y1 == y2: 
                break
            if err_1 > 0:
                x1 += sx
                err_1 -= 2 * dy
            if err_2 > 0:
                z1 += sz
                err_2 -= 2 * dy
            err_1 += 2 * dx
            err_2 += 2 * dz
            y1 += sy
            
    else:
        err_1 = 2 * dy - dz
        err_2 = 2 * dx - dz
        for i in range(max_len):
            line[i, 0], line[i, 1], line[i, 2] = z1, y1, x1
            if z1 == z2: 
                break
            if err_1 > 0:
                y1 += sy
                err_1 -= 2 * dz
            if err_2 > 0:
                x1 += sx
                err_2 -= 2 * dz
            err_1 += 2 * dy
            err_2 += 2 * dx
            z1 += sz
            
    return line

@njit
def _compute_arc_length_numba(
        path: np.ndarray, center_u: np.ndarray, center_v: np.ndarray) -> float:
    """Computes the arc-length of a path including the gaps to bounding node centers.

    Bridges the distance from ``center_u`` to the first path pixel and from the last
    path pixel to ``center_v``. Zero-pixel paths (direct node-to-node edges) are
    handled as a special case.

    Args:
        path: Array of shape (N, D) of path pixel coordinates.
        center_u: Coordinate array of length D for the source node center.
        center_v: Coordinate array of length D for the target node center.

    Returns:
        Total arc-length as a float.
    """
    D = center_u.shape[0]
    n_points = path.shape[0]
    
    # Base case: 0-pixel path (direct connection between nodes)
    if n_points == 0:
        dist_sq = 0.0
        for j in range(D):
            diff = center_u[j] - center_v[j]
            dist_sq += diff * diff
        return np.sqrt(dist_sq)
        
    length = 0.0
    
    # Distance from center_u to the start of the path
    dist_sq = 0.0
    for j in range(D):
        diff = center_u[j] - path[0, j]
        dist_sq += diff * diff
    length += np.sqrt(dist_sq)
    
    # Distance strictly along the path array
    for i in range(1, n_points):
        dist_sq = 0.0
        for j in range(D):
            diff = path[i, j] - path[i-1, j]
            dist_sq += diff * diff
        length += np.sqrt(dist_sq)
        
    # Distance from the end of the path to center_v
    dist_sq = 0.0
    for j in range(D):
        diff = path[n_points-1, j] - center_v[j]
        dist_sq += diff * diff
    length += np.sqrt(dist_sq)
        
    return length

def compute_arc_length(path: np.ndarray, u: int, v: int, graph: nx.MultiGraph) -> float:
    """Computes the arc-length of an edge, bridging the gaps to node centers.

    Args:
        path: Array of shape (N, D) of the edge's path pixel coordinates.
        u: Source node identifier.
        v: Target node identifier.
        graph: Graph containing node attributes; nodes must have a ``center`` attribute.

    Returns:
        Total arc-length of the edge as a float.
    """

    center_u = np.array(graph.nodes[u]["center"], dtype=np.int32)
    center_v = np.array(graph.nodes[v]["center"], dtype=np.int32)

    return _compute_arc_length_numba(path, center_u, center_v)    

def calculate_edge_lengths(graph: nx.MultiGraph) -> None:
    """Computes and assigns arc-length to every edge in the graph in-place.

    Args:
        graph: Graph whose edges are annotated with a ``length`` attribute.
    """
    for u, v, key, data in graph.edges(keys=True, data=True):
        graph[u][v][key]["length"] = compute_arc_length(data["path"], u, v, graph)

@njit
def get_euclidean_trim_indices(
        path: np.ndarray,
        center_u: np.ndarray,
        center_v: np.ndarray,
        r_u: float,
        r_v: float,
        ) -> tuple[int, int]:
    """Computes the trim indices that exclude path segments inside junction spheres.

    Walks cumulatively from each node center along the path until the arc-length
    exceeds the respective node radius. Handles both 2D and 3D paths.

    Args:
        path: Array of shape (N, D) of path pixel coordinates.
        center_u: Coordinate array of the source node center.
        center_v: Coordinate array of the target node center.
        r_u: Radius of the source junction sphere.
        r_v: Radius of the target junction sphere.

    Returns:
        A tuple (trim_u, trim_v) of non-negative integers such that
        ``path[trim_u : N - trim_v]`` is the exposed outer path segment.
    """
    n_points = path.shape[0]
    D = path.shape[1]

    if n_points == 0:
        return 0, 0

    # --- Forward pass for trim_u ---
    trim_u = 0
    dist_u = 0.0
    
    # Distance from center_u to path[0]
    dist_sq = 0.0
    for j in range(D):
        diff = center_u[j] - path[0, j]
        dist_sq += diff * diff
    dist_u += np.sqrt(dist_sq)

    if dist_u > r_u:
        trim_u = 0
    else:
        for i in range(1, n_points):
            dist_sq = 0.0
            for j in range(D):
                diff = path[i, j] - path[i-1, j]
                dist_sq += diff * diff
            dist_u += np.sqrt(dist_sq)

            if dist_u > r_u:
                trim_u = i
                break
        else:
            trim_u = n_points # Entire path is swallowed by r_u

    # --- Backward pass for trim_v ---
    trim_v = 0
    dist_v = 0.0
    
    # Distance from center_v to path[-1]
    dist_sq = 0.0
    for j in range(D):
        diff = center_v[j] - path[n_points-1, j]
        dist_sq += diff * diff
    dist_v += np.sqrt(dist_sq)

    if dist_v > r_v:
        trim_v = 0
    else:
        for i in range(n_points - 1, 0, -1):
            dist_sq = 0.0
            for j in range(D):
                diff = path[i, j] - path[i-1, j]
                dist_sq += diff * diff
            dist_v += np.sqrt(dist_sq)

            if dist_v > r_v:
                trim_v = n_points - i
                break
        else:
            trim_v = n_points # Entire path is swallowed by r_v

    return trim_u, trim_v

def get_outer_path_radii(
        graph: nx.MultiGraph,
        u: int,
        v: int,
        key: int,
        r_u: float,
        r_v: float,
        edt_bg: np.ndarray,
        ) -> tuple[float, float]:
    """Samples mean and minimum radius statistics from the exposed outer path of an edge.

    Trims path segments inside junction spheres and queries the Euclidean distance
    transform only on the remaining exposed section.

    Args:
        graph: Graph containing node and edge attributes.
        u: Source node identifier.
        v: Target node identifier.
        key: Edge key in the multigraph.
        r_u: Radius of the source node.
        r_v: Radius of the target node.
        edt_bg: Euclidean distance transform of the background.

    Returns:
        A tuple (mean_radius, min_radius) sampled from the outer path.
    """
    path = graph[u][v][key]["path"]
    path_len = len(path)
    if path_len == 0:
        val = (r_u + r_v) / 2.0
        return val, val
    
    node_start, node_end = (u, v) if u < v else (v, u)
    r_start, r_end = (r_u, r_v) if u < v else (r_v, r_u)
        
    center_start = np.array(graph.nodes[node_start]["center"], dtype=np.int32)
    center_end = np.array(graph.nodes[node_end]["center"], dtype=np.int32)
    
    # Use exact Euclidean trimming
    trim_start, trim_end = get_euclidean_trim_indices(
        path, center_start, center_end, r_start, r_end)
    
    # If the path is entirely swallowed by the two junctions
    if trim_start + trim_end >= path_len:
        mid_idx = path_len // 2
        val = edt_bg[tuple(path[mid_idx])]
        return val, val
        
    # Sample only the exposed "core" path
    outer_path = path[trim_start : path_len - trim_end]
    edt_vals = edt_bg[tuple(outer_path.T)]
    
    return np.mean(edt_vals), np.min(edt_vals)

def map_centerline_labels_to_edges(
        labeled_volume: np.ndarray, id_cl_map: dict) -> tuple[np.ndarray, dict]:
    """Collapses per-pixel centerline labels into per-edge labels.

    Replaces each per-pixel label in ``labeled_volume`` with a single shared label for
    the entire edge, so all pixels belonging to the same edge share one ID.

    Args:
        labeled_volume: Integer array with one unique label per centerline pixel, as
            produced by ``map_foreground_to_graph``.
        id_cl_map: Mapping from per-pixel label IDs to edge and centerline-index data.

    Returns:
        A tuple (labeled_elems, id_edge_map). ``labeled_elems`` is a new integer array
        with per-edge labels. ``id_edge_map`` maps each new label ID to the
        corresponding edge identifier tuple (u, v, key).
    """

    # Create map from edge_id to list of label_ids (multiple labels usually belong to the same edge)
    edge_ids_map = {}
    for label_id, edge_data in id_cl_map.items():
        edge_id = edge_data["edge"]
        if edge_id not in edge_ids_map:
            edge_ids_map[edge_id] = []    
        edge_ids_map[edge_id].append(label_id)

    # Initialize mapping array where index == value (Nodes inherently map to themselves)
    max_id = labeled_volume.max()
    mapping = np.arange(max_id + 1, dtype=np.int32)
    
    # Find node IDs (for calculating current_id)
    ids = set(np.unique(labeled_volume).tolist())
    ids.discard(0)
    node_ids = ids - set(id_cl_map.keys())

    # Update mapping array for edges
    current_id = max(node_ids) + 1 if node_ids else 1    
    id_edge_map = {}
    
    for edge_id, label_ids in edge_ids_map.items():
        for label_id in label_ids:
            mapping[label_id] = current_id # Instant 1D update
        id_edge_map[current_id] = edge_id
        current_id += 1

    # 4. Instantly project to 3D volume
    labeled_elems = mapping[labeled_volume]

    return labeled_elems, id_edge_map

def construct_attribute_volume(
        graph: nx.MultiGraph, 
        labeled_volume: np.ndarray, 
        id_cl_map: dict | None = None,
        id_edge_map: dict | None = None, 
        edge_attr: str | None = None,
        id_node_map: dict | None = None, 
        node_attr: str | None = None, 
        ) -> np.ndarray:
    """Constructs a volume where each voxel holds the value of a graph attribute.

    Uses a lookup table indexed by ``labeled_volume`` to map each label ID to the
    specified edge or node attribute. Background voxels (label 0) are set to NaN.

    Args:
        graph: The graph containing the attribute values.
        labeled_volume: Integer array assigning each foreground voxel to a label ID.
        id_cl_map: Per-pixel centerline label map (mutually exclusive with
            ``id_edge_map``).
        id_edge_map: Per-edge label map (mutually exclusive with ``id_cl_map``).
        edge_attr: Name of the edge attribute to map. Required when ``id_edge_map`` or
            ``id_cl_map`` is provided.
        id_node_map: Per-node label map.
        node_attr: Name of the node attribute to map. Required when ``id_node_map`` is
            provided.

    Returns:
        Float32 array with the same shape as ``labeled_volume`` where each voxel holds
        the attribute value of its assigned graph element, or NaN for background.

    Raises:
        ValueError: If required arguments are missing or mutually exclusive arguments
            are both provided.
    """
    
    if id_node_map is None and id_edge_map is None and id_cl_map is None:
        raise ValueError("At least one of id_node_map, id_edge_map, or id_cl_map must be provided.")
    
    if id_cl_map is not None and id_edge_map is not None:
        raise ValueError("id_cl_map and id_edge_map are mutually exclusive.")
    
    if id_node_map is not None and node_attr is None:
        raise ValueError("node_attr must be specified if id_node_map is provided.")
    
    if (id_edge_map is not None or id_cl_map is not None) and edge_attr is None:
        raise ValueError("edge_attr must be specified if id_edge_map or id_cl_map is provided.")
    
    # Build a lookup table up to the max ID found in the volume
    max_id = labeled_volume.max()
    lookup = np.zeros(max_id + 1, dtype=np.float32)

    # Set Background to NaN
    lookup[:] = np.nan

    if id_node_map is not None:
        for label_id, node_id in id_node_map.items():
            val = graph.nodes[node_id][node_attr]
            lookup[label_id] = val

    if id_edge_map is not None:
        for label_id, edge_data in id_edge_map.items():
            u, v, key = edge_data
            val = graph[u][v][key][edge_attr]            
            lookup[label_id] = val

    elif id_cl_map is not None:
        for label_id, edge_data in id_cl_map.items():
            u, v, key = edge_data["edge"]
            idx = edge_data["abs_idx"]
            val = graph[u][v][key][edge_attr][idx]
            lookup[label_id] = val

    # Project it to the full 3D volume
    attribute_volume = lookup[labeled_volume]

    return attribute_volume