"""Functions for refining the graph topology by removing artifacts and simplifying structures."""

from typing import Callable

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

from pyvane.graph.border import mark_border
from pyvane.graph.label_geometry import (
    label_elongation,
    paths_close,
    vessel_radius,
    vessel_radius_map,
)
from pyvane.graph.util import (
    bresenham_3d,
    compute_arc_length,
    get_euclidean_trim_indices,
    get_outer_path_radii,
)
from pyvane.util.misc import scalar

# Defaults of the knobs, chosen on synthetic stacks with ground truth (see refine_graph)
SPUR_REACH = 2.5
JUNCTION_REACH = 1.5
TUNNEL_WIDTH = 3.0


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
        radius_map: np.ndarray,
        radius_ratio_threshold: float = 0.7, 
        length_ratio_threshold: float = 1.0
        ) -> tuple[bool, list[np.ndarray]]:
    """Removes thin, short H-bridge edges that spuriously connect two thicker vessels.

    An edge between two junctions is removed when its minimum radius is below
    ``radius_ratio_threshold`` times the smaller junction radius and its exposed length
    (outside the two junctions) is below ``length_ratio_threshold`` times the smaller
    junction radius. Long capillary connections are preserved.

    Args:
        graph: The graph to modify in-place.
        edt_bg: Euclidean distance transform of the background (for the removed paths).
        radius_map: Vessel radius map (``label_geometry.vessel_radius_map``).
        radius_ratio_threshold: Fraction of the smaller junction radius below which the
            bridge minimum radius is considered thin.
        length_ratio_threshold: Largest exposed length of a removed bridge, in units of the
            smaller junction radius.

    Returns:
        A tuple (changed, trash_paths) where ``changed`` is True if the topology was
        altered and ``trash_paths`` is a list of removed pixel coordinate arrays.
    """

    if radius_ratio_threshold <= 0 or length_ratio_threshold <= 0:
        return False, []

    edges_to_delete = []
    trash_paths = []
    
    for u, v, key, data in list(graph.edges(keys=True, data=True)):
        if u == v:
            continue  # Ignore self-loops
            
        if graph.degree(u) > 2 and graph.degree(v) > 2:
            r_u = vessel_radius(radius_map, graph.nodes[u]["center"])
            r_v = vessel_radius(radius_map, graph.nodes[v]["center"])

            _, min_radius = get_outer_path_radii(graph, u, v, key, r_u, r_v, radius_map)
                
            r_min = min(r_u, r_v)
            is_thin = min_radius < radius_ratio_threshold * r_min
            is_short = max(0.0, data["length"] - (r_u + r_v)) < length_ratio_threshold * r_min
            
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

def _path_centrality(path: np.ndarray, edt_bg: np.ndarray) -> float:
    """Mean distance to the background along a path (0 for an empty path)."""
    path = np.asarray(path, int)
    return float(np.mean(edt_bg[tuple(path.T)])) if len(path) else 0.0

def merge_tunnels(
        graph: nx.MultiGraph, edt_bg: np.ndarray, radius_map: np.ndarray, elongation: float,
        tunnel_width: float = 3.0) -> tuple[bool, list[np.ndarray]]:
    """Merges the edges that run around a hole in the label into one (in place).

    A hole through a vessel's label (a tunnel) makes its skeleton split into two paths
    around the hole: two edges between the same nodes, or a self-loop. Rather than by the
    edge length, a pair of paths is a tunnel when they stay close to each other everywhere,
    in units of the local label width:

    - for each pair of nodes joined by several edges, the most central edge (largest mean
      distance to the background) is kept, and every other edge whose path stays within
      ``tunnel_width`` label widths of it is removed;
    - a self-loop whose two halves stay within ``tunnel_width`` label widths of each other is
      removed.

    Real loops (two vessels joining the same junctions, rings) stay apart by more than a
    vessel width and are kept. The label width is 2 x the vessel radius + 1 voxel
    (``label_geometry.paths_close``); in 3D, z distances are divided by ``elongation``, the
    ratio of the label's half-height to its in-plane radius.

    Args:
        graph: The graph to modify in-place.
        edt_bg: Euclidean distance transform of the background (path centrality and the
            removed paths).
        radius_map: Vessel radius map (``label_geometry.vessel_radius_map``).
        elongation: Label elongation (``label_geometry.label_elongation``); ignored in 2D.
        tunnel_width: Largest distance between the two paths, in local label widths.

    Returns:
        A tuple (changed, trash_paths) where ``changed`` is True if any edge was removed and
        ``trash_paths`` lists the removed paths outside the junction spheres.
    """
    if tunnel_width <= 0:
        return False, []
    elongation = max(1.0, float(elongation))
    n_dims = radius_map.ndim
    to_remove = []
    pairs = {(min(u, v), max(u, v)) for u, v in graph.edges() if u != v}
    for u, v in pairs:
        if graph.number_of_edges(u, v) < 2:
            continue
        keys = sorted(graph[u][v],
                      key=lambda key: -_path_centrality(graph[u][v][key]["path"], edt_bg))
        main_path = np.asarray(graph[u][v][keys[0]]["path"]).reshape(-1, n_dims)
        for key in keys[1:]:
            path = np.asarray(graph[u][v][key]["path"]).reshape(-1, n_dims)
            if paths_close(main_path, path, radius_map, elongation, tunnel_width):
                to_remove.append((u, v, key))
    for node in [n for n in graph.nodes if graph.has_edge(n, n)]:
        for key, data in graph[node][node].items():
            path = np.asarray(data["path"]).reshape(-1, n_dims)
            half = len(path) // 2
            if paths_close(path[:half], path[half:], radius_map, elongation, tunnel_width):
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
        radius_map: np.ndarray, 
        collapse_length_ratio_threshold: float = 1.5,
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
        radius_map: Vessel radius map (``label_geometry.vessel_radius_map``).
        collapse_length_ratio_threshold: An edge is collapsed if its length divided by
            the sum of the connected node radii is below this value.

    Returns:
        True if at least one bridge was collapsed, False otherwise.
    """
    if collapse_length_ratio_threshold <= 0:
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
            r_u = vessel_radius(radius_map, graph.nodes[u]["center"])
            r_v = vessel_radius(radius_map, graph.nodes[v]["center"])
            
            # Criterion: Junctions geometrically overlap into a single body
            if bridge_length >= (r_u + r_v) * collapse_length_ratio_threshold:
                continue

            # Compute new merged center for u
            c_u = np.array(graph.nodes[u]["center"])
            c_v = np.array(graph.nodes[v]["center"])
            new_center = np.round((c_u + c_v) / 2.0).astype(np.int32)

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

def to_simple_graph(graph: nx.MultiGraph) -> None:
    """Turns a multigraph into a simple graph in-place: every self-loop is removed and, for
    each pair of nodes joined by several edges, only the longest edge is kept.

    This is a conversion for analyses that need a simple graph, not a refinement heuristic:
    the removed edges may be real vessels (e.g. two vessels joining the same junctions).
    Degree-2 nodes left by the removals are merged into their edges.

    Args:
        graph: The graph to modify in-place.
    """
    graph.remove_edges_from(list(nx.selfloop_edges(graph, keys=True)))
    pairs = {(min(u, v), max(u, v)) for u, v in graph.edges() if graph.number_of_edges(u, v) > 1}
    for u, v in pairs:
        keys = sorted(graph[u][v], key=lambda key: graph[u][v][key]["length"], reverse=True)
        graph.remove_edges_from([(u, v, key) for key in keys[1:]])
    remove_degree_two_nodes(graph, keep_rings=False)

def refine_graph(
        condensed_graph: nx.MultiGraph,
        bin_img: np.ndarray,
        spur_reach: float = SPUR_REACH,
        junction_reach: float = JUNCTION_REACH,
        tunnel_width: float = TUNNEL_WIDTH,
        keep_rings: bool = True,
        step_callback: Callable[[str, nx.MultiGraph, bool, list[np.ndarray]], None] | None = None,
        edt_bg: np.ndarray | None = None,
        radius_map: np.ndarray | None = None,
        ) -> tuple[nx.MultiGraph, np.ndarray, list[np.ndarray]]:
    """Simplifies the skeleton graph with one decision per structure type, until stable.

    Every distance is relative to the vessel radius (``label_geometry.vessel_radius``: the
    in-plane distance to the background in 3D, the distance transform in 2D), so the same
    values work at any voxel size. Each knob trades precision and recall for one structure
    type; a value of 0 turns its step off:

    - ``spur_reach`` (leaves, ``guided.prune_leaves``): a leaf branch is removed if it
      reaches less than this many parent radii out of its parent vessel in the image plane
      (from the parent centerline, in units of the mean radius along it). In 3D, a leaf is
      kept anyway if a tip of the graph of the label projected along z (pruned with the same
      ``spur_reach``) matches it, or if it ends on the top or bottom face. At the end, in
      3D, the end of each leaf that runs off the vessel axis along z (onto the end cap of a
      tall label) is cut, and leaves matched by a projected tip are extended in XY to it
      (``guided.place_leaf_tips``). Higher: fewer spurs, more short real branches lost.
    - ``junction_reach`` (split junctions, ``collapse_internal_bridges``): two junctions
      joined by an edge shorter than this many times the sum of their radii are merged.
      Higher: fewer junctions split in two, more close real junctions merged.
    - ``tunnel_width`` (tunnels, ``merge_tunnels``): edges between the same nodes (or the two
      halves of a self-loop) that stay within this many local label widths of each other run
      around a hole in the label and are merged into one. Higher: fewer label holes, more
      tight real loops and touching vessels merged.

    Thin, short bridges between two vessels (contacts) are always removed
    (``remove_false_anastomoses`` with its default thresholds).

    Vessels cut by the XY faces of the image (``pyvane.graph.border``): the thinning stops the
    skeleton up to one radius before the face, so a cut vessel can look like a short leaf.
    Leaf branches whose mask reaches an XY face and whose end points to it are not removed by
    the leaf rule, since they may be vessels leaving the image. At the end, every leaf tip
    whose mask reaches an XY face gets ``border=True`` (other nodes ``border=False``) and is
    moved to the face, its path extended by the straight segment; every edge gets
    ``border_fraction``, the fraction of its path where the mask reaches an XY face. The z
    faces are not treated this way.

    The defaults were chosen on synthetic stacks with ground truth (widefield 10x and 20x,
    confocal 10x; exact masks and labels made like the real ones): the heuristics that did not
    improve the graphs were removed, and the knob values minimize the error of the
    branch-point and length densities. For labels that are thick relative to the vessels and
    little blurred along z (e.g. widefield 20x), a lower ``junction_reach`` (0.5-1) was
    better.

    Args:
        condensed_graph: The initial graph to be refined.
        bin_img: The original binary image, used for EDT calculations.
        spur_reach: Leaves (see above), in parent radii.
        junction_reach: Split junctions (see above), in sums of the two junction radii.
        tunnel_width: Tunnels (see above), in local label widths.
        keep_rings: Whether to preserve pure ring structures (degree 2 nodes with a single
            self-loop).
        step_callback: Optional function called after every step as
            ``step_callback(step, graph, changed, trash_paths)``, where ``step`` is one of
            "degree_two", "false_anastomoses", "leaves", "collapse" and "tunnels". It is meant
            for inspection and testing (e.g. checking the graph with
            ``pyvane.graph.validate.check_graph``); it must not modify the graph.
        edt_bg: Optional precomputed ``distance_transform_edt(bin_img)``, to save its
            computation when the same image is refined several times.
        radius_map: Optional precomputed ``label_geometry.vessel_radius_map(bin_img)``, for
            the same reason.

    Returns:
        The refined graph, the distance transform of the background and the paths removed
        (outside the junction spheres), for ``map_foreground_to_graph``.
    """
    # Imported here because pyvane.graph.guided imports this module
    from pyvane.graph.guided import place_leaf_tips, projection_guide, prune_leaves

    graph = condensed_graph.copy()
    bin_img = np.asarray(bin_img, bool)
    is_2d = graph.graph.get("is_2d", False) or bin_img.ndim == 2

    if edt_bg is None:
        edt_bg = distance_transform_edt(bin_img)
    if radius_map is None:
        radius_map = vessel_radius_map(bin_img)
    # the projected graph is pruned with the same spur_reach: a branch is a branch by its reach,
    # whatever its direction
    guide = (projection_guide(bin_img, prune_ratio_2d=spur_reach)
             if spur_reach > 0 and not is_2d else None)
    elongation = 1.0
    if tunnel_width > 0 and not is_2d:
        path_voxels = [np.asarray(d["path"], int).reshape(-1, 3)
                       for *_, d in graph.edges(data=True)]
        skeleton_voxels = np.vstack(path_voxels) if path_voxels else np.zeros((0, 3), int)
        elongation = label_elongation(bin_img, radius_map, skeleton_voxels)
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

        changed, trash = remove_false_anastomoses(graph, edt_bg, radius_map)
        if changed:
            global_trash_paths.extend(trash)
            topology_changed = True
        notify("false_anastomoses", changed, trash)

        if spur_reach > 0:
            changed, trash = prune_leaves(graph, edt_bg, radius_map, guide,
                                          spur_reach=spur_reach, keep_rings=keep_rings)
            if changed:
                global_trash_paths.extend(trash)
                topology_changed = True
            notify("leaves", changed, trash)

        changed = collapse_internal_bridges(graph, radius_map, junction_reach)
        if changed:
            topology_changed = True
        notify("collapse", changed, [])

        changed, trash = merge_tunnels(graph, edt_bg, radius_map, elongation, tunnel_width)
        if changed:
            global_trash_paths.extend(trash)
            topology_changed = True
        notify("tunnels", changed, trash)

    if guide is not None:
        place_leaf_tips(graph, radius_map, guide)
    mark_border(graph, radius_map)
    return graph, edt_bg, global_trash_paths
