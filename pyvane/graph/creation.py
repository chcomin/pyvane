"""Module for creating a graph from a binary skeleton image."""
import networkx as nx
import numpy as np
from numba import njit, prange
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from skimage.segmentation import watershed

from pyvane.graph.refinement import refine_graph, remove_degree_two_nodes
from pyvane.graph.util import (
    bresenham_3d,
    calculate_edge_lengths,
    get_euclidean_trim_indices,
)
from pyvane.util.misc import scalar


@njit(parallel=True)
def classify_pixels(img_padded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classifies pixels in a 3D binary skeleton into node (V) and path (P) types.

    A path pixel has exactly two neighbors that are not adjacent to each other. All other
    foreground pixels are classified as node pixels.

    Args:
        img_padded: 3D binary array padded by 1 on all sides to avoid boundary checks.

    Returns:
        A tuple (mask_P, mask_V) of boolean arrays with the same shape as ``img_padded``.
        ``mask_P`` marks path pixels and ``mask_V`` marks node pixels.
    """
    Z, Y, X = img_padded.shape
    mask_V = np.zeros_like(img_padded, dtype=np.bool_)
    mask_P = np.zeros_like(img_padded, dtype=np.bool_)
    
    # 26-connected neighborhood offsets
    offsets = np.zeros((26, 3), dtype=np.int32)
    idx = 0
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                offsets[idx, 0] = dz
                offsets[idx, 1] = dy
                offsets[idx, 2] = dx
                idx += 1
                
    for z in prange(1, Z - 1):
        for y in range(1, Y - 1):
            for x in range(1, X - 1):
                if not img_padded[z, y, x]:
                    continue
                
                # Count neighbors and record their positions
                neighbors_z = np.zeros(26, dtype=np.int32)
                neighbors_y = np.zeros(26, dtype=np.int32)
                neighbors_x = np.zeros(26, dtype=np.int32)
                count = 0
                
                for i in range(26):
                    dz, dy, dx = offsets[i]
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if img_padded[nz, ny, nx]:
                        neighbors_z[count] = nz
                        neighbors_y[count] = ny
                        neighbors_x[count] = nx
                        count += 1
                
                # Classification Logic
                if count != 2:
                    mask_V[z, y, x] = True
                else:
                    # Check if the two neighbors are adjacent to each other
                    dz = abs(neighbors_z[0] - neighbors_z[1])
                    dy = abs(neighbors_y[0] - neighbors_y[1])
                    dx = abs(neighbors_x[0] - neighbors_x[1])
                    
                    if max(dz, dy, dx) <= 1: # Chebyshev distance <= 1 means adjacent
                        mask_V[z, y, x] = True
                    else:
                        mask_P[z, y, x] = True
                        
    return mask_P, mask_V

@njit
def trace_path(path_coords: np.ndarray) -> np.ndarray:
    """Traces an unordered array of path coordinates into an ordered sequence.

    Finds an endpoint (degree-1 pixel) and walks the chain to produce a sorted
    coordinate array from one end to the other.

    Args:
        path_coords: Array of shape (N, 3) containing unordered path pixel coordinates.

    Returns:
        Array of shape (N, 3) with the same coordinates reordered along the path.
    """
    n_points = path_coords.shape[0]
    if n_points <= 1:
        return path_coords
        
    # Find the endpoint by checking degrees within this specific segment
    degrees = np.zeros(n_points, dtype=np.int32)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # Chebyshev distance
            dz = abs(path_coords[i, 0] - path_coords[j, 0])
            dy = abs(path_coords[i, 1] - path_coords[j, 1])
            dx = abs(path_coords[i, 2] - path_coords[j, 2])
            if max(dz, dy, dx) <= 1:
                degrees[i] += 1
                degrees[j] += 1
                
    # Start at the first point with exactly 1 neighbor
    start_idx = 0
    for i in range(n_points):
        if degrees[i] == 1:
            start_idx = i
            break
            
    # Trace the path
    ordered = np.zeros_like(path_coords)
    visited = np.zeros(n_points, dtype=np.bool_)
    
    curr_idx = start_idx
    ordered[0] = path_coords[curr_idx]
    visited[curr_idx] = True
    
    for step in range(1, n_points):
        for j in range(n_points):
            if not visited[j]:
                dz = abs(path_coords[curr_idx, 0] - path_coords[j, 0])
                dy = abs(path_coords[curr_idx, 1] - path_coords[j, 1])
                dx = abs(path_coords[curr_idx, 2] - path_coords[j, 2])
                
                if max(dz, dy, dx) <= 1:
                    ordered[step] = path_coords[j]
                    visited[j] = True
                    curr_idx = j
                    break
                    
    return ordered

def group_coordinates_by_label(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Extracts voxel coordinates for each labelled region in a single pass.

    Args:
        labels: 3D integer array where each non-zero voxel value identifies its label.

    Returns:
        Dictionary mapping each label ID to an (N, 3) array of voxel coordinates.
    """
    # Get 1D indices of all non-zero elements
    flat_indices = np.flatnonzero(labels)
    
    # Get the label values for these indices
    valid_labels = labels.ravel()[flat_indices]
    
    # Sort indices by their label
    sort_order = np.argsort(valid_labels)
    sorted_labels = valid_labels[sort_order]
    sorted_flat_indices = flat_indices[sort_order]
    
    # Find the boundaries where the label ID changes
    _, split_indices = np.unique(sorted_labels, return_index=True)
    
    # Split the sorted flat indices into a list of arrays
    grouped_flat_indices = np.split(sorted_flat_indices, split_indices[1:])
    unique_labels = sorted_labels[split_indices]
    
    # Convert flat indices back to 3D coordinates and store in dict
    shape = labels.shape
    grouped_coords = {}
    for label_id, flat_idx_array in zip(unique_labels, grouped_flat_indices):
        # unravel_index returns a tuple of arrays (Z_arr, Y_arr, X_arr)
        # we stack them to get an (N, 3) array
        coords = np.column_stack(np.unravel_index(flat_idx_array, shape))
        grouped_coords[label_id] = coords
        
    return grouped_coords

def graph_from_skeleton(skel_img: np.ndarray, keep_rings: bool = True) -> nx.MultiGraph:
    """Extracts an exact MultiGraph from a binary skeleton image.

    Classifies skeleton pixels into nodes and paths, traces each path component, and
    connects them to form graph edges. Handles both 2D and 3D arrays.

    Args:
        skel_img: 2D or 3D binary skeleton array.
        keep_rings: If True, isolated loops with no junction pixels are preserved as
            self-loop edges. If False, these ring components are removed.

    Returns:
        A MultiGraph representing the skeleton topology. Each node has a ``pixels``
        attribute with its constituent pixel coordinates and each edge has a ``path``
        attribute with ordered pixel coordinates.
    """

    img_shape = skel_img.shape
    is_2d = skel_img.ndim == 2
    if is_2d:
        skel_img = skel_img[np.newaxis, ...] # Promote to 3D
        
    img_padded = np.pad(skel_img, 1, mode="constant", constant_values=0)
    
    # Find Node (V) and Path (P) pixels
    mask_P_padded, mask_V_padded = classify_pixels(img_padded)
    
    # Remove padding to match original image coordinates
    mask_P = mask_P_padded[1:-1, 1:-1, 1:-1]
    mask_V = mask_V_padded[1:-1, 1:-1, 1:-1]
    
    struct_26 = generate_binary_structure(3, 3)

    # Pure Ring Fix
    # If a path component does not touch ANY V-pixel, it's a pure ring.
    # We force one pixel to become a V-pixel to break the loop.
    labels_P, num_P = label(mask_P, structure=struct_26) # type: ignore
    if num_P > 0:
        dilated_V = binary_dilation(mask_V, structure=struct_26)
        node_coords_dict = group_coordinates_by_label(labels_P)
        for _, coords in node_coords_dict.items():
            if not np.any(dilated_V[tuple(coords.T)]):
                if keep_rings:
                    # Pick the first pixel of the ring and turn it into a Node
                    first_pixel = coords[0]
                    mask_V[tuple(first_pixel)] = True
                    mask_P[tuple(first_pixel)] = False
                else:
                    mask_P[tuple(coords.T)] = False
                
    # Node & Path Identification
    labels_V, _ = label(mask_V, structure=struct_26) # type: ignore
    labels_P, _ = label(mask_P, structure=struct_26) # type: ignore
    
    graph = nx.MultiGraph()
    graph.graph = {
        "img_shape": img_shape,
        "is_2d": is_2d,
    }
    
    # Add Nodes
    node_coords_dict = group_coordinates_by_label(labels_V)
    for label_id, node_coords in node_coords_dict.items():
        if is_2d:
            node_coords = node_coords[:, 1:] # Drop Z coordinate for 2D
        
        graph.add_node(scalar(label_id), pixels=node_coords.astype(np.int32))

    label_shape: tuple[int, int, int] = labels_V.shape # type: ignore
        
    # Topology Linking (Edge Construction)
    path_coords_dict = group_coordinates_by_label(labels_P)
    for _, path_coords in path_coords_dict.items():
        ordered_path = trace_path(path_coords)
        
        # Find which nodes the endpoints touch
        end_A = ordered_path[0]
        end_B = ordered_path[-1]
        
        # Look at the 26-neighborhood of the endpoints in labels_V
        neighbors_A = []
        neighbors_B = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    # Bounds check (handled safely by numpy slice padding if we were careful, 
                    # but manual check here against shape is safer)
                    nzA, nyA, nxA = end_A[0]+dz, end_A[1]+dy, end_A[2]+dx
                    if (
                        0 <= nzA < label_shape[0] and 
                        0 <= nyA < label_shape[1] and 
                        0 <= nxA < label_shape[2]
                        ):
                        val_A = labels_V[nzA, nyA, nxA]
                        if val_A > 0: 
                            neighbors_A.append(val_A)
                        
                    nzB, nyB, nxB = end_B[0]+dz, end_B[1]+dy, end_B[2]+dx
                    if (
                        0 <= nzB < label_shape[0] and 
                        0 <= nyB < label_shape[1] and 
                        0 <= nxB < label_shape[2]
                        ):
                        val_B = labels_V[nzB, nyB, nxB]
                        if val_B > 0: 
                            neighbors_B.append(val_B)

        # Some sanity checks
        if not neighbors_A:
            raise ValueError(f"Start of path coords {ordered_path} do touch any nodes")
        
        if not neighbors_B:
            raise ValueError(f"End of path coords {ordered_path} do touch any nodes")
        
        neighbors = set(neighbors_A + neighbors_B)

        if len(neighbors) > 2:
            raise ValueError(
                f"Path coords {ordered_path} do not touch exactly one or two nodes")
        
        neighbors = list(neighbors)
        if len(neighbors) == 1:
            # Pure ring case: both endpoints touch the same node
            node_A = node_B = neighbors[0]
        elif len(path_coords) == 1:
            # Single-pixel path with two neighbors, node_A and node_B are arbitrarily assigned
            node_A, node_B = neighbors
        else:
            # Multipixel path with two neighbors
            node_A, node_B = neighbors_A[0], neighbors_B[0]
        
        # Enforce ordering: Smallest node index to largest node index
        if node_A > node_B:
            node_A, node_B = node_B, node_A
            ordered_path = ordered_path[::-1] # Reverse path to match node order
            
        if is_2d:
            ordered_path = ordered_path[:, 1:]
            
        graph.add_edge(scalar(node_A), scalar(node_B), path=ordered_path.astype(np.int32))

    return graph

def remove_clusters(
        full_graph: nx.MultiGraph, 
        remove_degree_two: bool = True, 
        keep_rings: bool = True
        ) -> nx.MultiGraph:
    """Collapses multi-pixel node clusters to single centroid nodes.

    For each node the centroid of all constituent pixels becomes the representative
    coordinate. Bresenham segments bridge the gap between centroids and adjacent path
    endpoints. Optionally merges any resulting degree-2 nodes.

    Args:
        full_graph: The raw graph produced by ``graph_from_skeleton``.
        remove_degree_two: If True, degree-2 nodes are merged into their incident edges
            after cluster collapse.
        keep_rings: If True, isolated loop nodes are preserved during degree-2 removal.

    Returns:
        A new MultiGraph with a ``center`` attribute on every node and a ``length``
        attribute on every edge.
    """
    graph = full_graph.copy()
    is_2d = graph.graph.get("is_2d", False)
    
    # Calculate and set the centroid for each node
    for node, data in graph.nodes(data=True):
        pixels = data["pixels"]
        # np.mean correctly handles (N, 2) or (N, 3) arrays natively
        centroid = np.round(np.mean(pixels, axis=0)).astype(np.int32)
        graph.nodes[node]["center"] = centroid
        
    # Bridge the gaps for all edges
    for u, v, key, data in graph.edges(keys=True, data=True):
        path = data["path"]
        center_u = graph.nodes[u]["center"]
        center_v = graph.nodes[v]["center"]
        
        # Prepare 3D coordinates for the Numba kernel
        if is_2d:
            c_u_3d = np.array([0, center_u[0], center_u[1]], dtype=np.int32)
            c_v_3d = np.array([0, center_v[0], center_v[1]], dtype=np.int32)
        else:
            c_u_3d = center_u.astype(np.int32)
            c_v_3d = center_v.astype(np.int32)
            
        # Handle the edge case of an empty path
        if len(path) == 0:
            full_bridge = bresenham_3d(c_u_3d, c_v_3d)
            if is_2d:
                full_bridge = full_bridge[:, 1:] # Strip Z-axis
            graph[u][v][key]["path"] = full_bridge[1:-1] # Exclude the centroids
            continue
            
        # Get path endpoints and pad if 2D
        if is_2d:
            p_start_3d = np.array([0, path[0][0], path[0][1]], dtype=np.int32)
            p_end_3d = np.array([0, path[-1][0], path[-1][1]], dtype=np.int32)
        else:
            p_start_3d = path[0].astype(np.int32)
            p_end_3d = path[-1].astype(np.int32)
            
        # Generate the discrete connecting bridges
        bridge_u = bresenham_3d(c_u_3d, p_start_3d)
        bridge_v = bresenham_3d(p_end_3d, c_v_3d)
        
        # Slice back to 2D if necessary
        if is_2d:
            bridge_u = bridge_u[:, 1:]
            bridge_v = bridge_v[:, 1:]
            
        # Drop the overlapping coordinate to avoid repeating the endpoint
        segment_u = bridge_u[1:-1]
        segment_v = bridge_v[1:-1]
        
        # Concatenate into the new, continuous path
        arrays_to_stack = []
        if len(segment_u) > 0: 
            arrays_to_stack.append(segment_u)
        arrays_to_stack.append(path)
        if len(segment_v) > 0: 
            arrays_to_stack.append(segment_v)
        
        new_path = np.vstack(arrays_to_stack)
        
        # Update the edge data
        graph[u][v][key]["path"] = new_path

    calculate_edge_lengths(graph)

    if remove_degree_two:
        remove_degree_two_nodes(graph, keep_rings = keep_rings)
        
    return graph

def assign_node_radii_and_outer_props(graph: nx.MultiGraph, edt_bg: np.ndarray) -> None:
    """Calculates and stores radius and trim information for all nodes and edges.

    For each node, the Euclidean distance transform value at the node's centroid is
    stored as ``radius``. For each edge, the indices at which the path exits the
    bounding junction spheres are stored as ``trim_amount``, along with the
    exposed ``outer_length``.

    Args:
        graph: The graph to annotate in-place. Nodes must have a ``center`` attribute
            and edges must have a ``path`` attribute.
        edt_bg: Euclidean distance transform of the background, with the same spatial
            shape as the original image.
    """

    for n in graph.nodes():
        center = tuple(graph.nodes[n]["center"])
        graph.nodes[n]["radius"] = float(edt_bg[center])
        
    for u, v, _, data in graph.edges(keys=True, data=True):
        path = data["path"]
        path_len = len(path)
        
        if path_len == 0:
            data["trim_amount"] = (0, 0)
            data["outer_length"] = 0.0
            continue
            
        # Enforce min -> max ordering to perfectly align with the path array
        node_start, node_end = (u, v) if u < v else (v, u)
        
        # Determine the effective trimming radius (0.0 if not a junction)
        r_start = graph.nodes[node_start]["radius"] if graph.degree(node_start) > 2 else 0.0
        r_end = graph.nodes[node_end]["radius"] if graph.degree(node_end) > 2 else 0.0
        
        center_start = np.array(graph.nodes[node_start]["center"], dtype=np.int32)
        center_end = np.array(graph.nodes[node_end]["center"], dtype=np.int32)
        
        trim_start, trim_end = get_euclidean_trim_indices(
            path, center_start, center_end, r_start, r_end
        )
        
        # Store the indices on the edge dictionary
        data["trim_amount"] = (int(trim_start), int(trim_end))
        data["outer_length"] = max(0.0, float(path_len - r_start - r_end))

def map_foreground_to_graph(
        graph: nx.MultiGraph, 
        bin_img: np.ndarray, 
        edt_bg: np.ndarray, 
        edges_only: bool = True, 
        compactness: float = 0.0,
        trash_paths: list[np.ndarray] | None = None, 
        ) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, dict]:
    """Partitions the foreground volume by assigning each voxel to a graph element.

    Computes a geodesic Voronoi partition of the binary image and maps each region to
    either a node (junction) or an edge pixel via watershed flooding.

    Args:
        graph: The refined graph whose nodes and edges define partition seeds.
        bin_img: Binary foreground mask of the same shape as the original image.
        edt_bg: Euclidean distance transform of the background.
        edges_only: If True, only edge pixels are used as seeds; junction volumes are
            not separately labelled. If False, degree->2 nodes are also labelled.
        compactness: Compactness parameter passed to the watershed algorithm.
        trash_paths: List of pixel coordinate arrays for removed path segments, used
            to suppress their influence during watershed flooding.

    Returns:
        If ``edges_only`` is True, a tuple (labeled_volume, id_cl_map). Otherwise, a
        tuple (labeled_volume, id_cl_map, id_node_map). ``labeled_volume`` assigns each
        foreground voxel a partition ID. ``id_cl_map`` maps label IDs to edge and
        centerline-index data. ``id_node_map`` maps label IDs to node identifiers.
    """
    marker_volume = np.zeros_like(bin_img, dtype=np.int32)
    
    id_node_map = {}
    id_cl_map = {}
    
    TRASH_ID = -1
    
    # Burn Trash Markers
    if trash_paths:
        for t_path in trash_paths:
            if len(t_path) > 0:
                marker_volume[tuple(t_path.T)] = TRASH_ID
                
    current_id = 1
    
    # Keep a temporary reverse lookup to assign internal paths to node IDs
    node_to_id = {}
    
    if not edges_only:
        # Burn Node Markers (Junction Centers)
        for n in graph.nodes():
            if graph.degree(n) > 2:
                id_node_map[current_id] = n
                node_to_id[n] = current_id
                
                center = tuple(graph.nodes[n]["center"])
                marker_volume[center] = current_id
                current_id += 1
            
    # Process Edges (Inner Paths vs. Outer Paths)
    for u, v, key, data in graph.edges(keys=True, data=True):
        path = data["path"]
        path_len = len(path)
        
        if path_len == 0:
            continue
            
        # Enforce min -> max ordering to align mathematically with the path array
        node_start, node_end = (u, v) if u < v else (v, u)
        edge_id = (node_start, node_end, key)
        
        trim_start, trim_end = data["trim_amount"]
        if edges_only:
            trim_start, trim_end = 0, 0

        # Scenario A: Edge is entirely swallowed by junctions
        if trim_start + trim_end >= path_len:
            mid_idx = path_len // 2
            if graph.degree(node_start) > 2 and mid_idx > 0:
                marker_volume[tuple(path[:mid_idx].T)] = node_to_id[node_start]
            if graph.degree(node_end) > 2:
                marker_volume[tuple(path[mid_idx:].T)] = node_to_id[node_end]
                
        # Scenario B: Normal edge with an exposed outer path
        else:
            # Burn the inner paths using the existing Junction IDs
            if trim_start > 0 and graph.degree(node_start) > 2:
                marker_volume[tuple(path[:trim_start].T)] = node_to_id[node_start]
            if trim_end > 0 and graph.degree(node_end) > 2:
                marker_volume[tuple(path[path_len - trim_end:].T)] = node_to_id[node_end]
            
            # Burn the outer path: EVERY PIXEL gets a unique ID
            # We map it to its exact index 'i' within the edge's original path array
            for i in range(trim_start, path_len - trim_end):
                coord = tuple(path[i].tolist())
                marker_volume[coord] = current_id
                id_cl_map[current_id] = {
                    "edge": edge_id,
                    "abs_idx": i,                 
                    "outer_idx": i - trim_start,  
                    "coord": coord                
                }
                current_id += 1

    # Execute Topographical Flooding
    topography = -edt_bg
    labeled_volume = watershed(
        topography, marker_volume, mask=bin_img, connectivity=3, compactness=compactness) # type: ignore
    
    labeled_volume[labeled_volume == TRASH_ID] = 0

    if edges_only:
        return labeled_volume, id_cl_map
    
    return labeled_volume, id_cl_map, id_node_map

def create_graph(
    skel_img: np.ndarray, 
    bin_img: np.ndarray, 
    length_threshold: float = 5.0,
    keep_rings: bool = True, 
    full_output: bool = False
    ) -> nx.MultiGraph | tuple[nx.MultiGraph, np.ndarray, list[np.ndarray]]:
    """Main function to create a graph from a skeleton image and its corresponding binary mask.
    
    The function performs the following steps:
    1. Extraction of an initial graph from the skeleton. This graph contains all information 
    about nodes, edges, and their paths. That is, it is a pure topological graph with no pruning 
    or simplification. 
    2. A well-known problem with skeletonization is that it produces islands of connected pixels. 
    This is particularly problematic at junctions, but even L corners end up being represented
    as a node since the pixels have three neighbors. To address this, a cluster removal step
    is performed to collapse these islands to single centroids and to remove the spurious degree-2 
    nodes generated by the previous step. 
    3. The resulting graph is then refined using a battery of anatomical and topological criteria 
    to remove spurious branches and to absorb small surface bumps. This step is crucial for 
    cleaning up the graph and improving the quality of the final output. The refinement step
    has many parameters, this function assumes some sane default values. If more controls is
    desired, the user can call the individual functions separately and pass in custom parameters.

    Note: An isotropic image is assumed.

    Args:
        skel_img: 2D or 3D binary array representing the skeletonized image.
        bin_img: 2D or 3D binary array representing the original foreground mask.
        length_threshold: Minimum length for edges, in voxels, to be retained during refinement.
        keep_rings: Whether to keep pure rings (isolated loops) in the graph.
        full_output: If True, also returns the Euclidean distance transform and the list
            of paths removed during refinement.

    Returns:
        The refined graph, or a tuple (graph, edt_bg, trash_paths) when ``full_output``
        is True. ``edt_bg`` is the Euclidean distance transform of the background;
        ``trash_paths`` lists pixel coordinate arrays removed during refinement.
    """

    # Create initial graph that represents all skeleton pixels as accurately as possible
    init_graph = graph_from_skeleton(skel_img, keep_rings = keep_rings)

    # Remove clusters and degree-2 nodes to get a cleaner graph topology before refinement.
    cond_graph = remove_clusters(init_graph, remove_degree_two = True, keep_rings = keep_rings)

    # Refine the graph topology
    simple_graph, edt_bg, trash_paths = refine_graph(
        cond_graph, 
        bin_img,
        bulge_len_threshold = length_threshold,
        bulge_size_threshold = 0.0,
        bulge_ratio_threshold = 0.0,
        elongation_threshold = 0.0,
        multi_edge_threshold = length_threshold, 
        self_loop_threshold = length_threshold, 
        bridge_radius_ratio_threshold = 0.5,
        bridge_length_threshold = 3.0,
        collapse_length_ratio_threshold = 1.0,
        comp_size_threshold = 0, 
        comp_length_threshold = 1.0, 
        keep_rings = keep_rings
        )

    assign_node_radii_and_outer_props(simple_graph, edt_bg)
    
    if not full_output:
        return simple_graph
    
    return simple_graph, edt_bg, trash_paths

def create_graph_with_mapping(
    skel_img: np.ndarray, 
    bin_img: np.ndarray, 
    length_threshold: float = 5.0,
    edges_only: bool = True,
    keep_rings: bool = True, 
    ) -> tuple[nx.MultiGraph, np.ndarray, dict] | tuple[nx.MultiGraph, np.ndarray, dict, dict]:
    """Creates a graph and also returns a mapping from the foreground pixels to the graph structure.
    
    Please see `create_graph` for details on the graph creation process. This function extends 
    that by also returning a labeled volume where each foreground pixel is assigned a unique integer
    ID corresponding to either a node or an edge in the graph. 

    Args:
        skel_img: 2D or 3D binary array representing the skeletonized image.
        bin_img: 2D or 3D binary array representing the original foreground mask.
        length_threshold: Minimum length for edges to be retained during refinement.
        edges_only: If True, pixels are only mapped to graph edges. If False, pixels are
            also mapped to junction nodes when they are closer to a junction than to an
            edge.
        keep_rings: Whether to preserve pure ring structures in the graph.

    Returns:
        A tuple (graph, labeled_volume, id_cl_map) when ``edges_only`` is True, or
        (graph, labeled_volume, id_cl_map, id_node_map) when False. ``labeled_volume``
        assigns each foreground voxel a unique label. ``id_cl_map`` maps label IDs to
        edge and centerline-index information. ``id_node_map`` maps label IDs to node
        identifiers (only present when ``edges_only`` is False).
    """ 

    simple_graph, edt_bg, trash_paths = create_graph(
        skel_img = skel_img, 
        bin_img = bin_img,
        length_threshold = length_threshold,
        keep_rings = keep_rings, 
        full_output = True
    )

    labeled_volume, *maps = map_foreground_to_graph(
        graph = simple_graph, 
        bin_img = bin_img, 
        edt_bg = edt_bg, 
        edges_only = edges_only, 
        compactness = 0.1,
        trash_paths = trash_paths
        )
    
    output = (simple_graph, labeled_volume, maps[0])
    if not edges_only:
        output += (maps[1],)

    return output