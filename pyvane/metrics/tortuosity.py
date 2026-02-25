"""Calculation of blood vessel tortuosity."""

import networkx as nx
import numpy as np


def avg_func(x):
    """Average function."""

    return sum(x)/len(x)

def parametric_line_fit(path):
    """Fits a straight line to the points in ``path`` using arc-length parameterisation.

    Returns equally-spaced (along the original path) points projected onto the
    best-fit straight line.

    Args:
        path: List of coordinate tuples representing the path.

    Returns:
        List of 1D arrays, one per spatial coordinate axis, each with the same length
        as ``path``, containing the fitted line coordinates.
    """

    path = np.array(path)

    dpath = np.diff(path, axis=0)
    dlength = np.sqrt(np.sum(dpath**2, axis=1))
    s = np.array([0, *(np.cumsum(dlength)/sum(dlength)).tolist()])

    fitted_coord = []
    for coord in path.T:
        slope, intercept = np.polyfit(s, coord, 1)
        fitted_coord.append(slope*s + intercept)

    return fitted_coord

def point_path_dist(point, path):
    """Computes the Euclidean distance from a single point to every point in a path.

    Args:
        point: A single coordinate tuple.
        path: Array of shape (N, D) containing path coordinates.

    Returns:
        Array of length N with the distance from ``point`` to each path point.
    """

    return np.sqrt(np.sum((path-point)**2, axis=1))

def vector_norm(vector):
    """Computes the Euclidean norm of a vector.

    Args:
        vector: The input vector.

    Returns:
        The Euclidean norm as a float.
    """

    return np.sqrt(np.sum(vector**2))

def points_line_dist(points, v1, v2):
    """Computes the perpendicular distance from each point to a line defined by two points.

    Args:
        points: Single coordinate tuple or array of shape (N, D).
        v1: First point defining the line.
        v2: Second point defining the line.

    Returns:
        Array of perpendicular distances, one per input point.
    """

    points, v1, v2 = np.array(points), np.array(v1), np.array(v2)

    if points.ndim==1:
        points = points[None]

    # Line versor
    v12 = v2 - v1
    n12 = v12/vector_norm(v12)

    # Vector from v1 to points
    v1p = v1 - points

    # Scalar product between v1p and n12 (i.e., size of v1p projected into n12)
    v1pProjn12 = np.sum(v1p*n12, axis=1)
    v1pProjn12 = v1pProjn12.reshape(points.shape[0], 1)

    # Vector from points to the closest point in the line
    vpx = v1p - v1pProjn12*n12

    # Norm of all vectors in vpx
    distances = np.sqrt(np.sum(vpx**2, axis=1))

    return distances

def find_path_segment(path, pixel_index, radius):
    """Extracts the segment of ``path`` within ``radius`` of a reference pixel.

    Args:
        path: List of coordinate tuples representing the path.
        pixel_index: Index of the reference pixel in ``path``.
        radius: Maximum distance from the reference pixel to include.

    Returns:
        Array containing the extracted path segment.
    """

    path = np.array(path)
    pixel = path[pixel_index]

    dist = point_path_dist(pixel, path)
    ind_left, _ = find_left_path_segment_idx(path, pixel_index, radius, dist)
    ind_right, _ = find_right_path_segment_idx(path, pixel_index, radius, dist)

    edge_segment = path[ind_left:ind_right]

    return edge_segment

def find_left_path_segment_idx(path, pixel_index, radius, dist=None):
    """Finds the starting index of the path segment centred on ``pixel_index``.

    Args:
        path: List of coordinate tuples.
        pixel_index: Index of the reference pixel in ``path``.
        radius: Maximum distance from the reference pixel to include.
        dist: Optional precomputed distances from the reference pixel to all path points.

    Returns:
        A tuple (ind_left, is_left_border). ``ind_left`` is the starting index of the
        segment. ``is_left_border`` is True when the segment starts at the first pixel.
    """

    if dist is None:
        pixel = path[pixel_index]
        dist = point_path_dist(pixel, path)
    ind_left = pixel_index
    while ind_left>=0 and dist[ind_left]<=radius:
    #while ind_left>0 and dist[ind_left]<radius:    # To agree with old algorithm
        ind_left -= 1
    ind_left += 1

    if ind_left==0:
        is_left_border = True
    else:
        is_left_border = False

    return ind_left, is_left_border

def find_right_path_segment_idx(path, pixel_index, radius, dist=None):
    """Finds the ending index of the path segment centred on ``pixel_index``.

    Args:
        path: List of coordinate tuples.
        pixel_index: Index of the reference pixel in ``path``.
        radius: Maximum distance from the reference pixel to include.
        dist: Optional precomputed distances from the reference pixel to all path points.

    Returns:
        A tuple (ind_right, is_right_border). ``ind_right`` is one past the last segment
        index (suitable for slicing). ``is_right_border`` is True when the segment ends
        at the last pixel.
    """

    num_pixels = len(path)
    if dist is None:
        pixel = path[pixel_index]
        dist = point_path_dist(pixel, path)
    ind_right = pixel_index
    while ind_right<num_pixels and dist[ind_right]<=radius:
        ind_right += 1
    ind_right -= 1

    if ind_right==num_pixels-1:
        is_right_border = True
    else:
        is_right_border = False

    return ind_right+1, is_right_border

def tortuosity_path_pixel(path, scale, return_valid=True):
    """Calculates the local tortuosity at each pixel of a path.

    Args:
        path: List of coordinate tuples representing the path.
        scale: Scale at which tortuosity is evaluated (segment radius = scale / 2).
        return_valid: If True, pixels near path endpoints that cannot form a full
            neighbourhood are excluded.

    Returns:
        A tuple (path_tortuosity, left_border_index, right_border_index).
        ``path_tortuosity`` is a list of per-pixel tortuosity values.
        ``left_border_index`` and ``right_border_index`` mark the extent of the valid
        region when ``return_valid`` is True.

    Raises:
        ValueError: If the path has fewer than 2 points or ``scale`` is less than 2.
    """

    if len(path)<2 or scale<2:
        raise ValueError("Path must have at least two points for fitting.")

    num_pixels = len(path)
    left_border_index = num_pixels
    right_border_index = 0

    radius = scale/2
    idx_first_pixel, _ = find_right_path_segment_idx(path, 0, radius)
    idx_first_pixel -= 1
    idx_last_pixel, _ = find_left_path_segment_idx(path, num_pixels-1, radius)
    idx_last_pixel += 1
    left_border_index = idx_first_pixel
    right_border_index = idx_last_pixel
    if not return_valid:
        idx_first_pixel = 0
        idx_last_pixel = num_pixels

    path_tortuosity = []
    for pixel_index in range(idx_first_pixel, idx_last_pixel):

        path_segment = find_path_segment(path, pixel_index, scale/2)

        if len(path_segment)<2:
            raise ValueError("Path segment must have at least two points for fitting.")

        line_coords = parametric_line_fit(path_segment)
        v1, v2 = [], []
        for coord in line_coords:
            v1.append(coord[0])
            v2.append(coord[-1])

        distances = points_line_dist(path_segment, v1, v2)

        tortuosity_pixel = np.mean(distances)
        path_tortuosity.append(tortuosity_pixel)

    return path_tortuosity, left_border_index, right_border_index

def tortuosity_path(path, scale, reduction_function=None, use_valid=True):
    """Calculates the tortuosity of a path by aggregating pixel-level values.

    Args:
        path: List of coordinate tuples.
        scale: Scale at which tortuosity is evaluated.
        reduction_function: Function to reduce per-pixel tortuosity to a scalar (e.g.,
            mean). Defaults to the simple average.
        use_valid: If True, excludes pixels near path endpoints from the calculation.

    Returns:
        Scalar tortuosity value, or None if no valid pixels are available.
    """

    if reduction_function is None:
        reduction_function = avg_func

    pixelwise_tortuority, _, _ = tortuosity_path_pixel(path, scale, return_valid=use_valid)
    if len(pixelwise_tortuority)==0:
        path_tort = None
    else:
        path_tort = reduction_function(pixelwise_tortuority)

    return path_tort

def tortuosity(graph, scale, length_threshold, graph_reduction_func=None, path_reduction_func=None,
               pix_size=(1., 1., 1.), use_valid=True):
    """Calculates the mean tortuosity of blood vessels represented by the graph.

    Args:
        graph: The input graph. Edges must have a ``path`` attribute.
        scale: Scale at which tortuosity is evaluated. Smaller values capture fine local
            curvature; larger values reflect only broad directional changes.
        length_threshold: Edges shorter than this value (in physical units) are skipped.
            Edge length is approximated by the number of path points.
        graph_reduction_func: Function to aggregate per-edge tortuosity into a single
            result. Defaults to the simple average.
        path_reduction_func: Function to aggregate per-pixel tortuosity within an edge.
            Defaults to the simple average.
        pix_size: Physical size of the pixels per dimension.
        use_valid: If True, path regions near terminations are excluded.

    Returns:
        Scalar mean tortuosity across all qualifying edges.
    """

    if graph_reduction_func is None:
        graph_reduction_func = avg_func
    if path_reduction_func is None:
        path_reduction_func = avg_func

    pix_size = np.array(pix_size)
    tortuosities = []
    paths = nx.get_edge_attributes(graph, "path").values()
    for path in paths:
        if len(path)>(length_threshold/avg_func(pix_size)):
            path = np.array(path)*pix_size
            tort_val = tortuosity_path(
                path, scale, reduction_function=path_reduction_func, use_valid=use_valid)
            if tort_val is not None:
                tortuosities.append(tort_val)

    return graph_reduction_func(tortuosities)