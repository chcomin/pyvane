"""Utility functions for graph processing, including path tracing and arc-length calculations."""

import numpy as np
from numba import njit


@njit
def bresenham_3d(p1, p2):
    """Computes the 3D Bresenham line between two points.
    Expects p1 and p2 to be 1D numpy arrays of length 3: [z, y, x].
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
def _compute_arc_length_numba(path, center_u, center_v):
    """Computes the arc-length of a discrete path array, seamlessly bridging 
    the gaps to the bounding node centers. Handles 0-pixel paths efficiently.
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

def compute_arc_length(path, u, v, graph):
    """Wrapper to compute arc-length using numba-optimized function, extracting node centers 
    from the graph.
    """

    center_u = np.array(graph.nodes[u]["center"], dtype=np.int32)
    center_v = np.array(graph.nodes[v]["center"], dtype=np.int32)

    return _compute_arc_length_numba(path, center_u, center_v)    

def calculate_edge_lengths(graph):
    """Iterates through the graph and assigns the arc-length to every edge."""
    for u, v, key, data in graph.edges(keys=True, data=True):
        graph[u][v][key]["length"] = compute_arc_length(data["path"], u, v, graph)

@njit
def get_euclidean_trim_indices(path, center_u, center_v, r_u, r_v):
    """Calculates the exact array indices to trim by tracking cumulative 
    Euclidean distance (arc-length) from the junction centers.
    Handles 2D and 3D paths dynamically.
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

def get_outer_path_radii(graph, u, v, key, r_u, r_v, edt_bg):
    """Slices off the internal junction penetration to sample the true exposed vessel.
    Returns (mean_radius, min_radius).
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
