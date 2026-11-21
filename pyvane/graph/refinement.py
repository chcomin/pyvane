"""Functions for refining the graph topology by removing artifacts and simplifying structures."""

from typing import Callable

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

from pyvane.graph.label_geometry import best_pairing, inplane_edt, label_elongation, paths_close
from pyvane.graph.util import (
    bresenham_3d,
    compute_arc_length,
    get_euclidean_trim_indices,
    get_outer_path_radii,
)
from pyvane.util.misc import scalar


def remove_degree_two_nodes(graph: nx.MultiGraph, keep_rings: bool = True) -> bool:
    """Iteratively removes all degree-2 nodes by merging them into their incident edges.

    Each removal concatenates the two incident paths through the degree-2 node's center,
    respecting path orientation and accumulating arc-length. Repeats until no degree-2
    nodes remain.

    Args:
        graph: The graph to simplify in-place.
        keep_rings: If True, isolated self-loop nodes (pure rings) are preserved.

    Returns:
        True if at least one node was removed, False if the graph was unchanged.
    """

    graph_changed = False
    iteration_changed = True
    
    while iteration_changed:
        iteration_changed = False

        deg_2_nodes = [n for n in graph.nodes() if graph.degree(n) == 2]

        for n in deg_2_nodes:
            if n not in graph or graph.degree(n) != 2:
                continue

            # Check for pure isolated ring 
            if graph.number_of_edges(n, n) == 1 and len(list(graph.neighbors(n))) == 1:
                if not keep_rings:
                    graph.remove_node(n) # Remove the node and its self-loop edge
                    graph_changed = True
                    iteration_changed = True
                continue # Skip merging for this node since it's a pure ring

            edges = list(graph.edges(n, keys=True, data=True))
            
            # Since degree is 2 and it's not a single self-loop, there must be exactly 2 
            # distinct edges.
            _, u, _, d1 = edges[0]
            _, w, _, d2 = edges[1]

            path1 = d1["path"]
            path2 = d2["path"]
            
            # center_n needs an expanded dimension to vstack cleanly
            center_n = np.atleast_2d(graph.nodes[n]["center"])

            # Orient path1 so it strictly flows u -> n
            ordered_path1 = path1 if u < n else path1[::-1]

            # Orient path2 so it strictly flows n -> w
            ordered_path2 = path2 if n < w else path2[::-1]

            # Stack into a single continuous path: u -> n -> w
            arrays_to_stack = []
            if len(ordered_path1) > 0: 
                arrays_to_stack.append(ordered_path1)
            arrays_to_stack.append(center_n)
            if len(ordered_path2) > 0: 
                arrays_to_stack.append(ordered_path2)
            
            new_path = np.vstack(arrays_to_stack)

            # Rule: Path must flow from min(node_index) to max(node_index)
            if u > w:
                new_path = new_path[::-1]

            # Calculate the new accumulated length and add the edge
            new_length = compute_arc_length(new_path, min(u, w), max(u, w), graph)
            graph.add_edge(
                scalar(u), scalar(w), path=new_path.astype(np.int32), length=scalar(new_length))

            # Deleting the node automatically deletes the two old edges
            graph.remove_node(n)

            graph_changed = True
            iteration_changed = True

    return graph_changed 


def _extract_trash_path(
        graph: nx.MultiGraph, 
        u: int, 
        v: int, 
        key: int, 
        edt_bg: np.ndarray
        ) -> np.ndarray:
    """Extracts the exposed outer portion of an edge for use as a trash marker.

    Trims path segments that penetrate into junction spheres so that trash markers
    do not carve indentations into parent junctions during watershed flooding.

    Args:
        graph: The graph containing the edge.
        u: Source node identifier.
        v: Target node identifier.
        key: Edge key in the multigraph.
        edt_bg: Euclidean distance transform of the background.

    Returns:
        Array of pixel coordinates for the exposed path segment, or an empty array if
        the entire path is swallowed by the junction spheres.
    """
    path = graph[u][v][key]["path"]
    path_len = len(path)
    
    if path_len == 0:
        return np.array([])
    
    node_s, node_e = (u, v) if u < v else (v, u)
        
    # Treat endpoints (degree 1) as having radius 0 so we don't trim the tips
    r_s = edt_bg[tuple(graph.nodes[node_s]["center"])] if graph.degree(node_s) > 1 else 0.0
    r_e = edt_bg[tuple(graph.nodes[node_e]["center"])] if graph.degree(node_e) > 1 else 0.0
    
    center_s = np.array(graph.nodes[node_s]["center"], dtype=np.int32)
    center_e = np.array(graph.nodes[node_e]["center"], dtype=np.int32)

    trim_s, trim_e = get_euclidean_trim_indices(path, center_s, center_e, r_s, r_e)
    
    if trim_s + trim_e >= path_len:
        return np.array([]) 
        
    return path[trim_s : path_len - trim_e].copy()

def remove_false_anastomoses(
        graph: nx.MultiGraph, 
        edt_bg: np.ndarray, 
        radius_ratio_threshold: float = 0.25, 
        length_threshold: float = 5.0
        ) -> tuple[bool, list[np.ndarray]]:
    """Removes thin, short H-bridge edges that spuriously connect two thicker vessels.

    An edge is removed when both its minimum radius relative to the parent vessel radii
    falls below ``radius_ratio_threshold`` and its exposed length is shorter than
    ``length_threshold``. Long capillary connections are preserved.

    Args:
        graph: The graph to modify in-place.
        edt_bg: Euclidean distance transform of the background.
        radius_ratio_threshold: Fraction of the smaller parent radius below which the
            bridge minimum radius is considered thin.
        length_threshold: Maximum exposed outer length for a thin bridge to be removed.

    Returns:
        A tuple (changed, trash_paths) where ``changed`` is True if the topology was
        altered and ``trash_paths`` is a list of removed pixel coordinate arrays.
    """

    if radius_ratio_threshold <= 0 or length_threshold <= 0:
        return False, []

    edges_to_delete = []
    trash_paths = []
    
    for u, v, key, data in list(graph.edges(keys=True, data=True)):
        if u == v:
            continue  # Ignore self-loops
            
        if graph.degree(u) > 2 and graph.degree(v) > 2:
            r_u = edt_bg[tuple(graph.nodes[u]["center"])]
            r_v = edt_bg[tuple(graph.nodes[v]["center"])]

            _, min_radius = get_outer_path_radii(graph, u, v, key, r_u, r_v, edt_bg)
                
            is_thin = min_radius < radius_ratio_threshold * min(r_u, r_v)
            is_short = max(0.0, data["length"] - (r_u + r_v)) < length_threshold
            
            if is_thin and is_short:
                edges_to_delete.append((u, v, key))
                
    if edges_to_delete:
        for u, v, key in edges_to_delete:
            if graph.has_edge(u, v, key):
                trash_path = _extract_trash_path(graph, u, v, key, edt_bg)
                if len(trash_path) > 0:
                    trash_paths.append(trash_path)
                graph.remove_edge(u, v, key)
        return True, trash_paths
        
    return False, []

def resolve_multi_edges(
        graph: nx.MultiGraph, 
        edt_bg: np.ndarray, 
        threshold: float | None = 0.0
        ) -> tuple[bool, list[np.ndarray]]:
    """Removes short duplicate edges between the same pair of nodes.

    For each node pair with multiple edges, the longest edge is always preserved. Other
    edges shorter than ``threshold`` are removed. Pass ``threshold=None`` to remove all
    but the longest edge regardless of length.

    Args:
        graph: The graph to modify in-place.
        edt_bg: Euclidean distance transform of the background.
        threshold: Duplicate edges shorter than this value are removed. Pass None to
            remove all duplicates; pass 0 to skip multi-edge resolution entirely.

    Returns:
        A tuple (changed, trash_paths) where ``changed`` is True if any edges were
        removed and ``trash_paths`` is a list of removed pixel coordinate arrays.
    """

    if threshold == 0:
        return False, []

    changed = False
    trash_paths = []
    
    # Identify pairs of nodes that have multiple edges between them
    multi_pairs = set()
    for u, v in graph.edges():
        if u != v and graph.number_of_edges(u, v) > 1:
            multi_pairs.add(tuple(sorted((u, v))))

    for u, v in multi_pairs:
        edges_dict = graph[u][v]
        
        # Identify the largest edge to protect it
        max_len = -1
        max_key = None
        for key, data in edges_dict.items():
            if data["length"] > max_len:
                max_len = data["length"]
                max_key = key

        keys_to_remove = []
        for key, data in edges_dict.items():
            if key == max_key:
                continue # Always protect the largest edge
            
            if threshold is None:
                keys_to_remove.append(key)
            elif threshold == 0:
                pass # Keep all
            elif data["length"] < threshold:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            trash_path = _extract_trash_path(graph, u, v, key, edt_bg)
            if len(trash_path) > 0:
                trash_paths.append(trash_path)
            graph.remove_edge(u, v, key)
            changed = True

    return changed, trash_paths

def resolve_self_loops(
        graph: nx.MultiGraph, 
        edt_bg: np.ndarray, 
        threshold: float | None = 0.0
        ) -> tuple[bool, list[np.ndarray]]:
    """Removes short self-loop edges from the graph.

    Applies the same length heuristic as ``resolve_multi_edges``.

    Args:
        graph: The graph to modify in-place.
        edt_bg: Euclidean distance transform of the background.
        threshold: Self-loops shorter than this value are removed. Pass None to remove
            all self-loops; pass 0 to skip self-loop resolution entirely.

    Returns:
        A tuple (changed, trash_paths) where ``changed`` is True if any edges were
        removed and ``trash_paths`` is a list of removed pixel coordinate arrays.
    """

    if threshold == 0:
        return False, []

    changed = False
    trash_paths = []
    
    # Identify nodes that have self-loops
    nodes_with_loops = [n for n in graph.nodes() if graph.number_of_edges(n, n) > 0]

    for n in nodes_with_loops:
        edges_dict = graph[n][n]

        keys_to_remove = []
        for key, data in edges_dict.items():
            if threshold is None:
                keys_to_remove.append(key)
            elif threshold == 0:
                pass
            elif data["length"] < threshold:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            trash_path = _extract_trash_path(graph, n, n, key, edt_bg)
            if len(trash_path) > 0:
                trash_paths.append(trash_path)
            graph.remove_edge(n, n, key)
            changed = True

    return changed, trash_paths

def _evaluate_branch_metrics(
        graph: nx.MultiGraph, 
        u: int, 
        v: int, 
        key: int, 
        edt_bg: np.ndarray
        ) -> tuple[float, float, float, float]:
    """Computes pruning metrics for a candidate leaf-branch edge.

    Zeroes internal radii for endpoint nodes and samples only the exposed outer path
    when computing radii-based ratios.

    Args:
        graph: The graph containing the edge.
        u: Source node identifier.
        v: Target node identifier.
        key: Edge key in the multigraph.
        edt_bg: Euclidean distance transform of the background.

    Returns:
        A tuple (bulge_len, bulge_size, bulge_ratio, elongation). Values are -1.0 for
        non-leaf edges where a metric is undefined.
    """
    deg_u = graph.degree(u)
    deg_v = graph.degree(v)
    
    # Identify radii, enforcing r=0 for endpoints
    r_u = edt_bg[tuple(graph.nodes[u]["center"])] if deg_u > 1 else 0.0
    r_v = edt_bg[tuple(graph.nodes[v]["center"])] if deg_v > 1 else 0.0
    
    # Calculate average radius along the path
    mean_radius, _ = get_outer_path_radii(graph, u, v, key, r_u, r_v, edt_bg)
        
    if mean_radius <= 0:
        return 0.0, 0.0, 0.0, 0.0
        
    length = graph[u][v][key]["length"]
    diameter = 2.0 * mean_radius
    
    # Elongation
    outer_len = max(0.0, float(length - r_u - r_v))
    elongation = outer_len / diameter
    
    # Bulge Size (Only valid for branch-to-leaf connections)
    if (deg_u == 1 and deg_v > 1) or (deg_u > 1 and deg_v == 1):
        # r_parent is the radius > 0, r_tip is the radius from the endpoint
        r_parent = r_u if deg_u > 1 else r_v
        
        n_e = u if deg_u == 1 else v
        r_tip = edt_bg[tuple(graph.nodes[n_e]["center"])]
        
        bulge_len = max(0.0, float(length - r_parent + r_tip))
        bulge_ratio = bulge_len / (r_parent + 1e-6)
        bulge_size = bulge_len / mean_radius
    else:
        # Not a leaf branch
        bulge_len = -1.0 
        bulge_size = -1.0 
        bulge_ratio = -1.0 
        
    return bulge_len, bulge_size, bulge_ratio, elongation

def prune_branches(
        graph: nx.MultiGraph, 
        edt_bg: np.ndarray, 
        bulge_len_threshold: float = 4.0, 
        bulge_size_threshold: float = 2.0,
        bulge_ratio_threshold: float = 0.5,
        elongation_threshold: float = 1.0, 
        ) -> tuple[bool, list[np.ndarray]]:
    """Iteratively prunes leaf branches that are likely segmentation artifacts.

    Leaf branches at each branching node are evaluated with multiple heuristics.
    Topology is protected by always preserving at least two edges at each junction.

    Args:
        graph: The graph to modify in-place.
        edt_bg: Euclidean distance transform of the background.
        bulge_len_threshold: Leaf branches with bulge length below this value are pruned.
        bulge_size_threshold: Leaf branches with bulge size below this value are absorbed.
        bulge_ratio_threshold: Leaf branches with bulge ratio below this value are pruned.
        elongation_threshold: Leaf branches with elongation below this value are absorbed.

    Returns:
        A tuple (changed, trash_paths) where ``changed`` is True if any edges were
        removed and ``trash_paths`` lists pixel coordinate arrays of truly pruned paths.
    """

    if (bulge_len_threshold <= 0 and bulge_size_threshold <= 0 and bulge_ratio_threshold <= 0 and 
        elongation_threshold <= 0):
        return False, []

    graph_changed = False
    iteration_changed = True
    trash_paths = []
    
    while iteration_changed:
        iteration_changed = False
        edges_to_delete = []
        
        for node in list(graph.nodes()):
            if node not in graph or graph.degree(node) <= 2:
                continue
                
            incident_edges = list(graph.edges(node, keys=True))
            potentially_deletable = []
            
            for u, v, key in incident_edges:
                other_node = v if u == node else u
                
                # Check if it's a leaf branch
                if graph.degree(other_node) == 1:
                    *bulge_stats, elongation = _evaluate_branch_metrics(graph, u, v, key, edt_bg)
                    bulge_len, bulge_size, bulge_ratio = bulge_stats
                    
                    is_artifact = False
                    to_trash = False

                    # Check for pure artifact (Trash)
                    if bulge_len < bulge_len_threshold or bulge_ratio < bulge_ratio_threshold:
                        is_artifact = True
                        to_trash = True
                    # Check for anatomical surface bump (Absorb)
                    elif bulge_size < bulge_size_threshold or elongation < elongation_threshold:
                        is_artifact = True
                        to_trash = False

                    if is_artifact:
                        potentially_deletable.append((u, v, key, bulge_len, to_trash))
                        
            if not potentially_deletable:
                continue
                
            # Voreen Topology Protection Logic (Keep at least 2 edges)
            total_edges = len(incident_edges)
            num_kept = total_edges - len(potentially_deletable)
            
            if num_kept >= 2:
                for item in potentially_deletable:
                    edges_to_delete.append(item)
            else:
                potentially_deletable.sort(key=lambda x: x[3], reverse=True)
                slice_idx = 1 if num_kept == 1 else 2
                for item in potentially_deletable[slice_idx:]:
                    edges_to_delete.append(item)
                    
        if edges_to_delete:
            iteration_changed = True
            graph_changed = True
            
            for u, v, key, _, to_trash in edges_to_delete:
                if graph.has_edge(u, v, key):
                    if to_trash:
                        trash_path = _extract_trash_path(graph, u, v, key, edt_bg)
                        if len(trash_path) > 0:
                            trash_paths.append(trash_path)
                    graph.remove_edge(u, v, key)
                    
            orphans = [n for n in graph.nodes() if graph.degree(n) == 0]
            graph.remove_nodes_from(orphans)
            
            remove_degree_two_nodes(graph)
                
    return graph_changed, trash_paths

def _bridge_from_center(
        center: np.ndarray, ordered_path: np.ndarray, is_2d: bool) -> np.ndarray:
    """Prepends to a path the voxels that join it to a node center.

    The joining voxels are the Bresenham line from ``center`` to ``ordered_path[0]``,
    excluding both of its endpoints, so that the result still excludes the node center.

    Args:
        center: Coordinates of the node center, of length 2 or 3.
        ordered_path: Array of shape (N, D) oriented so that it starts next to the node.
            Must not be empty.
        is_2d: Whether the coordinates are 2D.

    Returns:
        Array of shape (M, D) with M >= N.
    """
    if is_2d:
        c_3d = np.array([0, center[0], center[1]], dtype=np.int32)
        p_3d = np.array([0, ordered_path[0][0], ordered_path[0][1]], dtype=np.int32)
    else:
        c_3d = np.asarray(center, dtype=np.int32)
        p_3d = np.asarray(ordered_path[0], dtype=np.int32)

    bridge = bresenham_3d(c_3d, p_3d)
    if is_2d:
        bridge = bridge[:, 1:]

    segment = bridge[1:-1]
    if len(segment) == 0:
        return ordered_path
    return np.vstack([segment, ordered_path])

def _line_between_centers(
        center_a: np.ndarray, center_b: np.ndarray, is_2d: bool) -> np.ndarray:
    """Returns the voxels of the Bresenham line from ``center_a`` to ``center_b``, excluding
    both centers. The result may be empty."""
    if is_2d:
        a_3d = np.array([0, center_a[0], center_a[1]], dtype=np.int32)
        b_3d = np.array([0, center_b[0], center_b[1]], dtype=np.int32)
    else:
        a_3d = np.asarray(center_a, dtype=np.int32)
        b_3d = np.asarray(center_b, dtype=np.int32)

    line = bresenham_3d(a_3d, b_3d)[1:-1]
    if is_2d:
        line = line[:, 1:]
    return line

def _is_collapsible_bridge(graph: nx.MultiGraph, u: int, v: int) -> bool:
    """Checks, on the current graph, whether the edge u-v is a candidate bridge.

    Both nodes must exist and be branching nodes (degree > 2), they must be joined by exactly
    one edge, and ``v`` must not have a self-loop.
    """
    if u == v or u not in graph or v not in graph:
        return False
    if graph.degree(u) <= 2 or graph.degree(v) <= 2:
        return False
    if graph.number_of_edges(u, v) != 1:
        return False
    return not graph.has_edge(v, v)

def _oriented_points(graph: nx.MultiGraph, node: int, other: int, key: int) -> np.ndarray:
    """Points of the edge (node, other, key) from the first path voxel next to ``node`` to
    the center of ``other`` (paths are stored from the smaller to the larger node id)."""
    n_dims = len(graph.nodes[node]["center"])
    path = np.asarray(graph[node][other][key]["path"], float).reshape(-1, n_dims)
    if node > other:
        path = path[::-1]
    other_center = np.asarray(graph.nodes[other]["center"], float)[None]
    return np.vstack([path, other_center])

def _first_point_beyond(points: np.ndarray, origin: np.ndarray, distance: float) -> int:
    """Index of the first point at least ``distance`` from ``origin`` (the last if none is)."""
    beyond = np.nonzero(np.linalg.norm(points - origin, axis=1) >= distance)[0]
    return int(beyond[0]) if len(beyond) else len(points) - 1

def _branch_direction(
        graph: nx.MultiGraph, node: int, other: int, key: int, origin: np.ndarray,
        probe_distance: float) -> np.ndarray:
    """Unit direction of the edge (node, other, key) away from ``origin``: from its first point
    at least ``probe_distance`` away to its first point at least twice as far. Measured along
    the branch, it does not depend on where the node center sits (e.g. between two crossing
    vessels at different depths). Falls back to the direction from ``origin`` for short
    edges."""
    points = _oriented_points(graph, node, other, key)
    near = _first_point_beyond(points, origin, probe_distance)
    far = _first_point_beyond(points, origin, 2 * probe_distance)
    start = points[near] if far > near else origin
    direction = points[far] - start
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else direction

def _is_x_crossing(
        graph: nx.MultiGraph, edges: list, origin: np.ndarray, edt_bg: np.ndarray,
        crossing_angle: float, probe_factor: float = 2.0) -> bool:
    """Whether four edges leaving ``origin`` pair into two lines straighter than
    ``crossing_angle``. Directions are probed ``probe_factor`` x the radius at ``origin``
    away from it."""
    return _crossing_pairs(graph, edges, origin, edt_bg, crossing_angle, probe_factor) is not None

def _crossing_pairs(
        graph: nx.MultiGraph, edges: list, origin: np.ndarray, edt_bg: np.ndarray,
        crossing_angle: float, probe_factor: float = 2.0) -> tuple | None:
    """The two pairs of ``edges`` (indices) forming straight lines through ``origin``, or None
    if the four edges are not an X crossing (see ``_is_x_crossing``)."""
    origin = np.asarray(origin, float)
    probe_distance = probe_factor * max(float(edt_bg[tuple(origin.astype(int))]), 1.0)
    directions = np.array([_branch_direction(graph, node, other, key, origin, probe_distance)
                           for node, other, key in edges])
    if np.any(np.linalg.norm(directions, axis=1) == 0):
        return None
    angle, pairs = best_pairing(directions)
    return pairs if angle >= crossing_angle else None

def _bridge_is_x_crossing(
        graph: nx.MultiGraph, u: int, v: int, center: np.ndarray, edt_bg: np.ndarray,
        crossing_angle: float) -> bool:
    """Whether the bridge u-v is the middle of an X crossing: u and v have degree 3 and their
    four other branches pair into two lines through ``center`` straighter than
    ``crossing_angle``."""
    if graph.degree(u) != 3 or graph.degree(v) != 3:
        return False
    outer_edges = [(node, other, key) for node in (u, v)
                   for _, other, key in graph.edges(node, keys=True) if other not in (u, v)]
    if len(outer_edges) != 4:
        return False
    return _is_x_crossing(graph, outer_edges, center, edt_bg, crossing_angle)

def flag_crossings(
        graph: nx.MultiGraph, edt_bg: np.ndarray, crossing_angle: float = 150.0,
        probe_factor: float = 2.0) -> int:
    """Flags the nodes where two vessels cross (in place).

    A node of degree 4 without self-loops whose four edges pair into two lines straighter than
    ``crossing_angle`` (degrees) gets ``crossing=True`` and ``crossing_pairs``, the two pairs
    of neighbors that continue each other. Where two vessels cross at nearly the same depth,
    their labels merge and the graph joins them; the flag lets analyses leave such nodes out
    of the branch points. Flags of nodes that no longer qualify are removed.

    Args:
        graph: The graph to modify in-place.
        edt_bg: Euclidean distance transform of the background.
        crossing_angle: Smallest angle (degrees) between the two branches of each line.
        probe_factor: Branch directions are measured this many node radii away from the node.

    Returns:
        The number of flagged nodes.
    """
    n_flagged = 0
    for node in graph.nodes:
        graph.nodes[node].pop("crossing", None)
        graph.nodes[node].pop("crossing_pairs", None)
        if graph.degree(node) != 4 or graph.has_edge(node, node):
            continue
        edges = [(node, other, key) for _, other, key in graph.edges(node, keys=True)]
        center = np.asarray(graph.nodes[node]["center"], float)
        pairs = _crossing_pairs(graph, edges, center, edt_bg, crossing_angle, probe_factor)
        if pairs is None:
            continue
        neighbors = [other for _, other, _ in edges]
        graph.nodes[node]["crossing"] = True
        graph.nodes[node]["crossing_pairs"] = [(neighbors[i], neighbors[j]) for i, j in pairs]
        n_flagged += 1
    return n_flagged

def _path_centrality(path: np.ndarray, edt_bg: np.ndarray) -> float:
    """Mean distance to the background along a path (0 for an empty path)."""
    path = np.asarray(path, int)
    return float(np.mean(edt_bg[tuple(path.T)])) if len(path) else 0.0

def merge_tunnels(
        graph: nx.MultiGraph, edt_bg: np.ndarray, edt_xy: np.ndarray, elongation: float,
        tunnel_width: float = 3.0) -> tuple[bool, list[np.ndarray]]:
    """Merges the edges that run around a hole in the label into one (in place).

    A hole through a vessel's label (a tunnel) makes its skeleton split into two paths
    around the hole: two edges between the same nodes, or a self-loop. Unlike
    ``resolve_multi_edges`` and ``resolve_self_loops``, which use the edge length, a pair of
    paths is a tunnel when they stay close to each other everywhere, in units of the local
    label width:

    - for each pair of nodes joined by several edges, the most central edge (largest mean
      distance to the background) is kept, and every other edge whose path stays within
      ``tunnel_width`` label widths of it is removed;
    - a self-loop whose two halves stay within ``tunnel_width`` label widths of each other is
      removed.

    Real loops (two vessels joining the same junctions, rings) stay apart by more than a
    vessel width and are kept. The label width is 2 x the in-plane distance to the background
    + 1 voxel (``label_geometry.paths_close``); z distances are divided by ``elongation``, the
    ratio of the label's half-height to its in-plane radius.

    Only for 3D graphs.

    Args:
        graph: The graph to modify in-place.
        edt_bg: Euclidean distance transform of the background.
        edt_xy: In-plane distance transform (``label_geometry.inplane_edt``).
        elongation: Label elongation (``label_geometry.label_elongation``).
        tunnel_width: Largest distance between the two paths, in local label widths.

    Returns:
        A tuple (changed, trash_paths) where ``changed`` is True if any edge was removed and
        ``trash_paths`` lists the removed paths outside the junction spheres.
    """
    if tunnel_width <= 0:
        return False, []
    elongation = max(1.0, float(elongation))
    to_remove = []
    pairs = {(min(u, v), max(u, v)) for u, v in graph.edges() if u != v}
    for u, v in pairs:
        if graph.number_of_edges(u, v) < 2:
            continue
        keys = sorted(graph[u][v],
                      key=lambda key: -_path_centrality(graph[u][v][key]["path"], edt_bg))
        main_path = np.asarray(graph[u][v][keys[0]]["path"]).reshape(-1, 3)
        for key in keys[1:]:
            path = np.asarray(graph[u][v][key]["path"]).reshape(-1, 3)
            if paths_close(main_path, path, edt_xy, elongation, tunnel_width):
                to_remove.append((u, v, key))
    for node in [n for n in graph.nodes if graph.has_edge(n, n)]:
        for key, data in graph[node][node].items():
            path = np.asarray(data["path"]).reshape(-1, 3)
            half = len(path) // 2
            if paths_close(path[:half], path[half:], edt_xy, elongation, tunnel_width):
                to_remove.append((node, node, key))
    trash_paths = []
    for u, v, key in to_remove:
        if graph.has_edge(u, v, key):
            trash_path = _extract_trash_path(graph, u, v, key, edt_bg)
            if len(trash_path) > 0:
                trash_paths.append(trash_path)
            graph.remove_edge(u, v, key)
    graph.remove_nodes_from([n for n in list(graph.nodes) if graph.degree(n) == 0])
    return bool(to_remove), trash_paths

def collapse_internal_bridges(
        graph: nx.MultiGraph, 
        edt_bg: np.ndarray, 
        collapse_length_ratio_threshold: float = 0.25,
        crossing_angle: float = 150.0,
        crossing_length_ratio: float = 0.0,
        ) -> bool:
    """Collapses short internal bridges between adjacent branching junctions.

    When two high-degree nodes are connected by an edge whose length is less than
    ``collapse_length_ratio_threshold`` times the sum of their radii, the edge is
    collapsed and the two nodes are merged into one. Iterates until stable.

    A bridge u-v is only collapsed when u and v are joined by exactly one edge and v has no
    self-loop. These conditions are checked on the current graph right before each
    collapse, since earlier collapses in the same pass change the graph. When a bridge is
    collapsed, u moves to the midpoint of u and v, all edges of u (including its self-loops)
    are re-joined to the new center, and all edges of v are transferred to u.

    Args:
        graph: The graph to modify in-place.
        edt_bg: Euclidean distance transform of the background.
        collapse_length_ratio_threshold: An edge is collapsed if its length divided by
            the sum of the connected node radii is below this value.
        crossing_angle: Smallest angle (degrees) of the two lines of an X crossing: a
            bridge between two nodes of degree 3 is an X crossing that the skeleton split into
            >-< if the four other branches pair into two lines straighter than this.
        crossing_length_ratio: If larger than ``collapse_length_ratio_threshold``, X
            crossings are also collapsed when the bridge length divided by the sum of the
            node radii is below this value (vessels crossing a few radii apart in z, whose
            labels merge).

    Returns:
        True if at least one bridge was collapsed, False otherwise.
    """
    if collapse_length_ratio_threshold <= 0 and crossing_length_ratio <= 0:
        return False

    graph_changed = False
    iteration_changed = True    

    is_2d = graph.graph.get("is_2d", False)
    
    while iteration_changed:
        iteration_changed = False

        cand_bridge_edges = [
            (u, v) for u, v in graph.edges() if _is_collapsible_bridge(graph, u, v)
        ]

        for u, v in cand_bridge_edges:
            # Earlier collapses in this pass may have changed the graph.
            if not _is_collapsible_bridge(graph, u, v):
                continue

            bridge_length = next(iter(graph[u][v].values()))["length"]

            # Get radii of the two branching nodes
            r_u = edt_bg[tuple(graph.nodes[u]["center"])]
            r_v = edt_bg[tuple(graph.nodes[v]["center"])]
            
            # Criterion: Junctions geometrically overlap into a single body, or the bridge is
            # the middle of an X crossing (checked below)
            junction_limit = (r_u + r_v) * collapse_length_ratio_threshold
            crossing_limit = (r_u + r_v) * crossing_length_ratio
            if bridge_length >= max(junction_limit, crossing_limit):
                continue

            # Compute new merged center for u
            c_u = np.array(graph.nodes[u]["center"])
            c_v = np.array(graph.nodes[v]["center"])
            new_center = np.round((c_u + c_v) / 2.0).astype(np.int32)

            # beyond the junction limit, only X crossings of two degree-3 nodes are collapsed
            if bridge_length >= junction_limit and not _bridge_is_x_crossing(
                    graph, u, v, new_center, edt_bg, crossing_angle):
                continue

            graph.nodes[u]["center"] = new_center

            # Re-join the self-loops of u to the new center, at both ends of the path.
            self_loops = list(graph[u][u].items()) if graph.has_edge(u, u) else []
            for loop_key, loop_data in self_loops:
                path = loop_data["path"]
                if len(path) > 0:
                    path = _bridge_from_center(new_center, path, is_2d)
                    path = _bridge_from_center(new_center, path[::-1], is_2d)[::-1]
                graph.add_edge(
                    u,
                    u,
                    key=loop_key,
                    path=path.astype(np.int32),
                    length=scalar(compute_arc_length(path, u, u, graph)),
                )

            # Re-join the edges of u and transfer the edges of v to u. The bridge u-v is
            # dropped together with v. v has no self-loops (checked above).
            for ref_node in (u, v):
                for neighbor in list(graph.neighbors(ref_node)):
                    if neighbor in (u, v):
                        continue

                    for edge_key, edge_data in list(graph[ref_node][neighbor].items()):
                        path = edge_data["path"]

                        if len(path) > 0:
                            # Orient path so it flows ref_node -> neighbor
                            ordered_path = path if ref_node < neighbor else path[::-1]
                            new_path = _bridge_from_center(new_center, ordered_path, is_2d)
                            # Re-orient to flow min(node) -> max(node)
                            if u > neighbor:
                                new_path = new_path[::-1]
                        else:
                            # Direct node-to-node edge: join the two centers with a line.
                            new_path = _line_between_centers(
                                new_center, graph.nodes[neighbor]["center"], is_2d)
                            if u > neighbor:
                                new_path = new_path[::-1]

                        new_length = compute_arc_length(new_path, u, neighbor, graph)
                        # Edges of u keep their key; edges of v get a new one.
                        graph.add_edge(
                            u,
                            neighbor,
                            key=edge_key if ref_node == u else None,
                            path=new_path.astype(np.int32),
                            length=scalar(new_length),
                        )

            # Delete v (also removes the bridge and the original edges of v)
            graph.remove_node(v)
            graph_changed = True
            iteration_changed = True

    return graph_changed

def filter_components(
        graph: nx.MultiGraph, 
        comp_size_threshold: int = 0, 
        comp_length_threshold: float = 0.0
        ) -> None:
    """Removes small connected components from the graph in-place.

    A threshold of 0 means that criterion is not applied.

    Args:
        graph: The graph to filter in-place.
        comp_size_threshold: Minimum number of nodes required to keep a component.
        comp_length_threshold: Minimum total edge-length required to keep a component.
    """

    if comp_size_threshold == 0 and comp_length_threshold == 0.0:
        return

    # networkx connected_components yields sets of nodes
    components = list(nx.connected_components(graph))
    nodes_to_remove = []
    
    for comp in components:
        # Filter by number of nodes
        if len(comp) < comp_size_threshold:
            nodes_to_remove.extend(comp)
            continue # Already marked for deletion, skip length check
            
        # Filter by total physical length
        # Extract the subgraph to isolate this component's edges
        subgraph = graph.subgraph(comp)
        
        # Sum the pre-calculated lengths of all edges in this component
        total_length = sum(
            data["length"] for u, v, key, data in subgraph.edges(keys=True, data=True)
        )
        
        if total_length < comp_length_threshold:
            nodes_to_remove.extend(comp)
                
    graph.remove_nodes_from(nodes_to_remove)

def refine_graph(
        condensed_graph: nx.MultiGraph,
        bin_img: np.ndarray,
        bulge_len_threshold: float = 4.0,
        bulge_size_threshold: float = 0.0,
        bulge_ratio_threshold: float = 0.0,
        elongation_threshold: float = 0.0,
        multi_edge_threshold: float | None = 0.0, 
        self_loop_threshold: float | None = 0.0, 
        bridge_radius_ratio_threshold: float = 0.5,
        bridge_length_threshold: float = 3.0,
        collapse_length_ratio_threshold: float = 1.0,
        comp_size_threshold: int = 0, 
        comp_length_threshold: float = 1.0, 
        keep_rings: bool = True,
        step_callback: Callable[[str, nx.MultiGraph, bool, list[np.ndarray]], None] | None = None,
        edt_bg: np.ndarray | None = None,
        z_spur_params: dict | None = None,
        tunnel_width: float = 0.0,
        crossing_collapse_ratio: float = 0.0,
        crossing_angle: float = 0.0,
        ) -> tuple[nx.MultiGraph, np.ndarray, list[np.ndarray]]:
    """Simplifies the graph topology iteratively until stable. 
    
    The function has many threshold parameters setting the heuristics for graph refinement. Most
    of them are optional. bulge_len_threshold is usually the most important. It sets the maximum 
    allowed bulge length for leaf branches. 

    Pruning thresholds (bulge_*_threshold and elongation_threshold):
    These thresholds are applied for removing leaf branches that are likely to be artifacts. 
    The bulge length is defined as the numerator of the bulge_size equation in [1]. The bulge size 
    is given by bulge_len / mean_radius, where mean_radius is the average radius of the branch. 
    The bulge ratio is given by bulge_len / r_parent, where r_parent is the radius of the parent 
    vessel of the branch at the branching point.
    The elongation is given by the outer length (defined in [1]) divided by 2x the mean radius. 

    Graph cleaning thresholds (multi_edge_threshold, self_loop_threshold, comp_size_threshold, 
    comp_length_threshold):
    These thresholds are applied for removing multi-edges, self-loops, and small components.
    
    Bridge and crossing thresholds (bridge_radius_ratio_threshold, bridge_length_threshold, 
    collapse_length_ratio_threshold):
    These thresholds are applied for identifying and removing false anastomoses (H-bridges), 
    which are two unrelated but close vessels that end up connected by a short edge, and for
    collapsing X crosses that usually become two nodes connected by a short edge due to how
    skeletonization works. 

    Optional steps for labels that the axial blur makes taller than wide (3D only; all off by
    default):
    - z-spur pruning (``z_spur_params``): leaf branches without a counterpart in the graph of
      the label projected along z are removed (``pyvane.graph.guided.prune_z_spurs``), after
      ``prune_branches`` in every pass.
    - tunnel merging (``tunnel_width``): edges that run around a hole in the label are merged
      into one (``merge_tunnels``), after ``collapse_internal_bridges`` in every pass.
    - X-crossing collapse (``crossing_collapse_ratio``): bridges between two degree-3 nodes
      whose other branches form an X are collapsed up to a longer bridge length (see
      ``collapse_internal_bridges``).
    - crossing flags (``crossing_angle``): at the end, nodes where two vessels cross get
      ``crossing=True`` (``flag_crossings``).
    Values chosen on synthetic stacks with ground truth (widefield 10x and 20x, confocal 10x;
    exact masks and realistic labels): ``z_spur_params={}`` (the defaults of
    ``prune_z_spurs``), ``tunnel_width=3``, ``crossing_angle=150``,
    ``crossing_collapse_ratio=2.5``.

    Args:
        condensed_graph: The initial graph to be refined.
        bin_img: The original binary image, used for EDT calculations.
        bulge_len_threshold: Min allowed bulge length for leaf branches.
        bulge_size_threshold: Min allowed bulge size for leaf branches.
        bulge_ratio_threshold:  Min allowed bulge ratio for leaf branches.
        elongation_threshold: Min allowed elongation for leaf branches.
        multi_edge_threshold: When two nodes have multiple edges between them, edges smaller than 
            this threshold are removed, unless all edges are smaller than the threshold, in which 
            case the largest edge is kept. Set to None to remove all multiedges, which is useful for
            constructing a simple graph when necessary.
        self_loop_threshold: Self-loops smaller than this threshold are removed. Set to None to 
            remove all self-loops.
        bridge_radius_ratio_threshold: H-bridge removal. The bridge_radius_ratio is given by the
            minimum radius of the bridge divided by the minimum radius of the two parent vessels.
            Thus, thin bridges are candidates for removal.
        bridge_length_threshold: H-bridge removal. Besides being thin, the bridge must also be 
            smaller than this threshold to be removed.
        collapse_length_ratio_threshold: X-cross collapse. If the edge length divided by the
            sum of the radii of the connected nodes is smaller than this threshold, the edge is 
            collapsed into a single node. Note that this transformation might have unintended 
            consequences on the topology, so use with caution.
        comp_size_threshold: Min number of nodes for a connected component to be kept.
        comp_length_threshold: Min total edge length for a component to be kept.
        keep_rings: Whether to preserve pure ring structures (degree 2 nodes with a single 
            self-loop). Note that comp_*_threshold can also remove rings and have precedence over 
            this parameter.
        step_callback: Optional function called after every step as
            ``step_callback(step, graph, changed, trash_paths)``, where ``step`` is one of
            "degree_two", "multi_edges", "self_loops", "false_anastomoses", "prune",
            "z_spurs" (if enabled), "collapse", "tunnels" (if enabled) and "components". It is
            meant for inspection and testing (e.g. checking the graph with ``pyvane.graph.validate.check_graph``); it must not modify
            the graph.
        edt_bg: Optional precomputed ``distance_transform_edt(bin_img)``, to save its
            computation when the same image is refined several times.
        z_spur_params: If given, enables z-spur pruning. Keyword arguments of
            ``pyvane.graph.guided.projection_guide`` (``length_threshold_2d``,
            ``direction_length``) and ``prune_z_spurs`` (``match_distance``, ``xy_keep``,
            ``face_distance``), in units of the in-plane vessel radius; use ``{}`` for the
            defaults.
        tunnel_width: If positive, enables tunnel merging: the largest distance between the
            two paths around a hole, in local label widths (see ``merge_tunnels``).
        crossing_collapse_ratio: ``crossing_length_ratio`` of ``collapse_internal_bridges``:
            X crossings split into >-< are collapsed up to this bridge length / radii ratio
            (0: like any other bridge). The X test uses ``crossing_angle``, or 150 degrees if
            it is 0.
        crossing_angle: If positive, nodes of degree 4 whose edges pair into two lines
            straighter than this angle (degrees) are flagged ``crossing`` at the end
            (``flag_crossings``).

    References:
    [1] Drees, Dominik, et al. "Scalable robust graph and feature extraction for arbitrary vessel 
    networks in large volumetric datasets." BMC bioinformatics 22.1 (2021): 346.
    """
    graph = condensed_graph.copy()
    is_2d = graph.graph.get("is_2d", False) or np.ndim(bin_img) == 2
    if is_2d and (z_spur_params is not None or tunnel_width > 0):
        raise ValueError("z-spur pruning and tunnel merging need a 3D image.")

    if edt_bg is None:
        edt_bg = distance_transform_edt(bin_img)
    if z_spur_params is not None:
        # Imported here because pyvane.graph.guided imports this module
        from pyvane.graph.guided import projection_guide, prune_z_spurs
        guide_keys = ("length_threshold_2d", "direction_length")
        guide = projection_guide(
            bin_img, **{k: v for k, v in z_spur_params.items() if k in guide_keys})
        prune_keys = ("match_distance", "xy_keep", "direction_length", "face_distance")
        prune_params = {k: v for k, v in z_spur_params.items() if k in prune_keys}
    if z_spur_params is not None or tunnel_width > 0:
        edt_xy = inplane_edt(np.asarray(bin_img, bool))
    if tunnel_width > 0:
        path_voxels = [np.asarray(d["path"], int).reshape(-1, 3)
                       for *_, d in graph.edges(data=True)]
        skeleton_voxels = np.vstack(path_voxels) if path_voxels else np.zeros((0, 3), int)
        elongation = label_elongation(np.asarray(bin_img, bool), edt_xy, skeleton_voxels)
    x_angle = crossing_angle if crossing_angle > 0 else 150.0
    # Keep track of pixels from paths that are removed.
    global_trash_paths = []

    def notify(step, changed, trash):
        if step_callback is not None:
            step_callback(step, graph, changed, trash)

    topology_changed = True
    while topology_changed:
        topology_changed = False

        changed = remove_degree_two_nodes(graph, keep_rings = keep_rings)
        if changed:
            topology_changed = True
        notify("degree_two", changed, [])

        changed, trash = resolve_multi_edges(graph, edt_bg, multi_edge_threshold)
        if changed:
            global_trash_paths.extend(trash)
            topology_changed = True
        notify("multi_edges", changed, trash)
            
        changed, trash = resolve_self_loops(graph, edt_bg, self_loop_threshold)
        if changed:
            global_trash_paths.extend(trash)
            topology_changed = True
        notify("self_loops", changed, trash)

        changed, trash = remove_false_anastomoses(
            graph, 
            edt_bg, 
            radius_ratio_threshold = bridge_radius_ratio_threshold, 
            length_threshold = bridge_length_threshold
            )
        if changed:
            global_trash_paths.extend(trash)
            topology_changed = True
        notify("false_anastomoses", changed, trash)

        changed, trash = prune_branches(
            graph, 
            edt_bg=edt_bg, 
            bulge_len_threshold = bulge_len_threshold, 
            bulge_size_threshold = bulge_size_threshold,
            bulge_ratio_threshold = bulge_ratio_threshold,
            elongation_threshold = elongation_threshold,
            )
        if changed:
            global_trash_paths.extend(trash)
            topology_changed = True
        notify("prune", changed, trash)

        if z_spur_params is not None:
            changed, trash = prune_z_spurs(
                graph, edt_bg, edt_xy, guide, keep_rings=keep_rings, **prune_params)
            if changed:
                global_trash_paths.extend(trash)
                topology_changed = True
            notify("z_spurs", changed, trash)

        changed = collapse_internal_bridges(
            graph, 
            edt_bg, 
            collapse_length_ratio_threshold = collapse_length_ratio_threshold,
            crossing_angle = x_angle,
            crossing_length_ratio = crossing_collapse_ratio,
            )
        if changed:
            topology_changed = True
        notify("collapse", changed, [])

        if tunnel_width > 0:
            changed, trash = merge_tunnels(graph, edt_bg, edt_xy, elongation, tunnel_width)
            if changed:
                global_trash_paths.extend(trash)
                topology_changed = True
            notify("tunnels", changed, trash)

    n_before = graph.number_of_nodes()
    filter_components(graph, comp_size_threshold, comp_length_threshold)
    notify("components", graph.number_of_nodes() != n_before, [])

    if crossing_angle > 0:
        flag_crossings(graph, edt_bg, crossing_angle)

    return graph, edt_bg, global_trash_paths

