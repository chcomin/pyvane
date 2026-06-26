"""Functions for refining the graph topology by removing artifacts and simplifying structures."""

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

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
            new_length = compute_arc_length(new_path, u, w, graph)
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

def collapse_internal_bridges(
        graph: nx.MultiGraph, 
        edt_bg: np.ndarray, 
        collapse_length_ratio_threshold: float = 0.25
        ) -> bool:
    """Collapses short internal bridges between adjacent branching junctions.

    When two high-degree nodes are connected by an edge whose length is less than
    ``collapse_length_ratio_threshold`` times the sum of their radii, the edge is
    collapsed and the two nodes are merged into one. Iterates until stable.

    Args:
        graph: The graph to modify in-place.
        edt_bg: Euclidean distance transform of the background.
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

        cand_bridge_edges = []
        for u, v, key, data in graph.edges(keys=True, data=True):
            if u != v and graph.degree(u) > 2 and graph.degree(v) > 2:
                 # Exclude bridges where v has a self-loop and bridges with multiple edges.
                 if not graph.has_edge(v, v) and graph.number_of_edges(u, v) == 1:
                    cand_bridge_edges.append((u, v, key, data))

        for u, v, key, data in cand_bridge_edges:    
            if u not in graph or v not in graph:
                continue # One of the nodes was removed in a previous iteration
            if graph.degree(u) <= 2 or graph.degree(v) <= 2:
                # This condition should never be true since a bridge removal only increases
                # the degree of a node
                raise ValueError(
                    f"Inconsistent state for edge ({u}, {v}, {key}): degree(u)={graph.degree(u)}, "
                    f"degree(v)={graph.degree(v)}"
                    )

            # Get radii of the two branching nodes
            r_u = edt_bg[tuple(graph.nodes[u]["center"])]
            r_v = edt_bg[tuple(graph.nodes[v]["center"])]
            
            # Criterion: Junctions geometrically overlap into a single body
            if data["length"] < (r_u + r_v) * collapse_length_ratio_threshold:
                
                # Compute new merged center for u
                c_u = np.array(graph.nodes[u]["center"])
                c_v = np.array(graph.nodes[v]["center"])
                new_center = np.round((c_u + c_v) / 2.0).astype(np.int32)
                graph.nodes[u]["center"] = new_center
                
                # Bridge the gaps and re-wire all edges attached to v, transferring them to u
                for ref_node in [u, v]:
                    for neighbor in list(graph.neighbors(ref_node)):
                        if neighbor in (u, v):
                            # Drop the internal bridge itself. Warning!! This also avoids 
                            # transfering self-loops and multiedges between u and v. So, we 
                            # consider that they are also artifacts.
                            continue 
                            
                        for k, edge_data in list(graph[ref_node][neighbor].items()):
                            path = edge_data["path"]
                            
                            # Prepare 3D coordinates for Numba Bresenham
                            if is_2d:
                                c_new_3d = np.array(
                                    [0, new_center[0], new_center[1]], dtype=np.int32)
                            else:
                                c_new_3d = new_center.astype(np.int32)
                                
                            # Orient path so it flows ref_node -> neighbor
                            ordered_path = path if ref_node < neighbor else path[::-1]
                            
                            if len(ordered_path) > 0:
                                if is_2d:
                                    p_start_3d = np.array(
                                        [0, ordered_path[0][0], ordered_path[0][1]], dtype=np.int32)
                                else:
                                    p_start_3d = ordered_path[0].astype(np.int32)
                                
                                # Bridge the gap inside the junction
                                bridge = bresenham_3d(c_new_3d, p_start_3d)
                                if is_2d:
                                    bridge = bridge[:, 1:]
                                    
                                segment = bridge[1:-1]
                                arrays_to_stack = []
                                if len(segment) > 0: 
                                    arrays_to_stack.append(segment)
                                arrays_to_stack.append(ordered_path)

                                new_path = np.vstack(arrays_to_stack)
                            else:
                                new_path = path
                                
                            # Re-orient correctly to flow min(node) -> max(node)
                            if u > neighbor:
                                new_path = new_path[::-1]
                                
                            # Add the transferred edge to u
                            new_length = compute_arc_length(new_path, u, neighbor, graph)
                            key = None
                            if ref_node == u:
                                key = k
                            graph.add_edge(
                                u, 
                                neighbor, 
                                key=key, 
                                path=new_path.astype(np.int32), 
                                length=scalar(new_length)
                                )
                        
                # 3. Delete v (automatically cleans up the old bridge and edges)
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
        keep_rings: bool = True
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

    References:
    [1] Drees, Dominik, et al. "Scalable robust graph and feature extraction for arbitrary vessel 
    networks in large volumetric datasets." BMC bioinformatics 22.1 (2021): 346.
    """
    graph = condensed_graph.copy()

    edt_bg: np.ndarray = distance_transform_edt(bin_img)
    # Keep track of pixels from paths that are removed.
    global_trash_paths = []

    topology_changed = True
    while topology_changed:
        topology_changed = False

        if remove_degree_two_nodes(graph, keep_rings = keep_rings):
            topology_changed = True

        changed, trash = resolve_multi_edges(graph, edt_bg, multi_edge_threshold)
        if changed:
            global_trash_paths.extend(trash)
            topology_changed = True
            
        changed, trash = resolve_self_loops(graph, edt_bg, self_loop_threshold)
        if changed:
            global_trash_paths.extend(trash)
            topology_changed = True

        changed, trash = remove_false_anastomoses(
            graph, 
            edt_bg, 
            radius_ratio_threshold = bridge_radius_ratio_threshold, 
            length_threshold = bridge_length_threshold
            )
        if changed:
            global_trash_paths.extend(trash)
            topology_changed = True

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

        if collapse_internal_bridges(
            graph, 
            edt_bg, 
            collapse_length_ratio_threshold = collapse_length_ratio_threshold
            ):
            topology_changed = True

    filter_components(graph, comp_size_threshold, comp_length_threshold)

    return graph, edt_bg, global_trash_paths

