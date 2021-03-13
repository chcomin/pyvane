"""Calculation of blood vessel tortuosity."""

import numpy as np
import scipy
import networkx as nx

def parametric_line_fit(path):
    """Fit a straight line to the points in `path`. The returned value is an array containing
    points along the fitted line. Note that the points are equally spaced along `path`.

    Parameters
    ----------
    path : list of tuple
        List containing points in the path.

    Returns
    -------
    fitted_coord : ndarray
        Points along the fitted line. The size of the array is the same as the input list `path`.
    """

    path = np.array(path)

    dpath = np.diff(path, axis=0)
    dlength = np.sqrt(np.sum(dpath**2, axis=1))
    s = np.array([0] + (np.cumsum(dlength)/sum(dlength)).tolist())

    fitted_coord = []
    for coord in path.T:
        slope, intercept = scipy.polyfit(s, coord, 1)
        fitted_coord.append(slope*s + intercept)

    return fitted_coord

def point_path_dist(point, path):
    """Calculate all distances between a point and a path.

    Parameters
    ----------
    point : tuple of float
        A point.
    path : list of tuple
        List containing points in the path.

    Returns
    -------
    float
        The requested distances.
    """

    return np.sqrt(np.sum((path-point)**2, axis=1))

def vector_norm(vector):
    """The norm of a vector."""

    return np.sqrt(np.sum(vector**2))

def points_line_dist(points, v1, v2):
    """Calculate the smallest distance for each point in `points` to the line
    defined by the points `v1` and `v2`.

    Parameters
    ----------
    points : list of tuple
        A set of points.
    v1 : tuple
        The first point of a line.
    v2 : tuple
        The second point of a line.

    Returns
    -------
    distances : ndarray
        The requested distances.
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
    """Find the segment in `path` in which all points are a distance smaller than or equal
    to `radius` to the point in `path` defined by `pixel_index`. This is used for extracting
    the segment around a given point in the path for calculating the tortuosity.

    Parameters
    ----------
    path : list of tuple
        List containing points in the path.
    pixel_index : int
        The index of the point in `path` that will be used for extracting the segment.
    radius : float
        The radius used for extracting the segment.

    Returns
    -------
    edge_segment : ndarray
        The extracted segment.
    """

    path = np.array(path)
    pixel = path[pixel_index]

    dist = point_path_dist(pixel, path)
    ind_left, is_left_border = find_left_path_segment_idx(path, pixel_index, radius, dist)
    ind_right, is_right_border = find_right_path_segment_idx(path, pixel_index, radius, dist)

    edge_segment = path[ind_left:ind_right]

    return edge_segment

def find_left_path_segment_idx(path, pixel_index, radius, dist=None):
    """Auxiliary function for `find_path_segment`. Find the starting index of the
    segment that will be extracted from the path.

    Parameters
    ----------
    path : list of tuple
        List containing points in the path.
    pixel_index : int
        The index of the point in `path` that will be used for extracting the segment.
    radius : float
        The radius used for extracting the segment.
    dist : float, optional
        The distances between the point defined by `pixel_index` and each point in `path`.
        It is calculated if not provided.

    Returns
    -------
    ind_left : int
        Starting index of the segment to be extracted.
    is_left_border : bool
        If True, indicates that the segment begins at the first point of `path`.
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
    """Auxiliary function for `find_path_segment`. Find the last index of the
    segment that will be extracted from the path.

    Parameters
    ----------
    path : list of tuple
        List containing points in the path.
    pixel_index : int
        The index of the point in `path` that will be used for extracting the segment.
    radius : float
        The radius used for extracting the segment.
    dist : float, optional
        The distances between the point defined by `pixel_index` and each point in `path`.
        It is calculated if not provided.

    Returns
    -------
    ind_right : int
        Last index +1 of the segment to be extracted. Thus, the index can be used for
        slicing the `path` (e.g., path[:ind_right]).
    is_right_border : bool
        If True, indicates that the segment ends at the last point of `path`.
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
    """Calculate the tortuosity of each point in `path`.

    Parameters
    ----------
    path : list of tuple
        List containing points in the path.
    scale : float
        The scale at which the tortuosity will be calculated. That is, smaller values indicate
        that the tortuosity should be calculated for local changes in direction of the blood vessels,
        while larger values indicate that small changes should be ignored and only large variations
        should be taken into account.
    return_valid : bool
        If True, regions close to terminations and bifurcations of blood vessels are not considered
        in the calculation. This ignored region becomes larger with the `scale` parameter.

    Returns
    -------
    path_tortuosity : list of float
        The tortuosity around each point in `path`.
    left_border_index : int
        If `return_valid` is True, contains the index of the first point for which the tortuosity
        was calculated.
    right_border_index : int
        If `return_valid` is True, contains the index of the last point for which the tortuosity
        was calculated.
    """

    if len(path)<2 or scale<2:
        raise ValueError("Path must have at least two points for fitting.")

    num_pixels = len(path)
    left_border_index = num_pixels
    right_border_index = 0
    found_left_border = False
    found_right_border = False

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
        pixel = path[pixel_index]

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
    """Calculate the tortuosity of a path.

    Parameters
    ----------
    path : list of tuple
        List containing points in the path.
    scale : float
        The scale at which the tortuosity will be calculated. That is, smaller values indicate
        that the tortuosity should be calculated for local changes in direction of the blood vessels,
        while larger values indicate that small changes should be ignored and only large variations
        should be taken into account.
    reduction_function : func
        Function to use for reducing the tortuosity calculated for each point in `path` into a single
        value. For instance, it can be an average or median function.
    use_valid : bool
        If True, regions close to terminations and bifurcations of blood vessels are not considered
        in the calculation. This ignored region becomes larger with the `scale` parameter.

    Returns
    -------
    path_tort
        The tortuosity of the path.
    """

    if reduction_function is None:
        reduction_function = lambda x: sum(x)/len(x)   # Average

    pixelwise_tortuority, _, _ = tortuosity_path_pixel(path, scale, return_valid=use_valid)
    if len(pixelwise_tortuority)==0:
        path_tort = None
    else:
        path_tort = reduction_function(pixelwise_tortuority)

    return path_tort

def tortuosity(graph, scale, length_threshold, graph_reduction_func=None, path_reduction_func=None,
               pix_size=(1., 1., 1.), use_valid=True):
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
    length_threshold : float
        Edges smaller than `length_threshold` will not be included in the calculation. Note that
        `length_threshold` is given in physical units. Also, for compatibility purposes, the length
        of the edges are calculated using the number of points representing the edge instead of the
        arc length. If the edges are represented by adjacent pixels, this should not be a problem.
    graph_reduction_func :
        Function to use for reducing the tortuosity calculated for each edge into a single value.
        For instance, it can be an average or median function.
    path_reduction_func :
        Function to use for reducing the tortuosity calculated for each point in an edge into a
        single value.
    pix_size : tuple of float
        The physical size of the pixels in the image where the graph was generated.
    use_valid : bool
        If True, regions close to terminations and bifurcations of blood vessels are not considered
        in the calculation. This ignored region becomes larger with the `scale` parameter.

    Returns
    -------
    float
        A single value quantifying the tortuosity of the blood vessels.
    """

    avg_func = lambda x: sum(x)/len(x)   # Average
    if graph_reduction_func is None:
        graph_reduction_func = avg_func
    if path_reduction_func is None:
        path_reduction_func = avg_func

    pix_size = np.array(pix_size)
    tortuosities = []
    paths = nx.get_edge_attributes(graph, 'path').values()
    for path in paths:
        if len(path)>(length_threshold/avg_func(pix_size)):
            path = np.array(path)*pix_size
            tort_val = tortuosity_path(path, scale, reduction_function=path_reduction_func, use_valid=use_valid)
            if tort_val is not None:
                tortuosities.append(tort_val)

    return graph_reduction_func(tortuosities)