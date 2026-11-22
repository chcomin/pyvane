"""Vessels cut by the XY faces of the image.

The thinning treats the outside of the image as background, so the skeleton of a vessel cut
by an image face stops up to one vessel radius before the face, and its leaf branch looks like
a short spur. Whether such a branch is a vessel leaving the image or an artifact on a vessel
that touches the face cannot be told from the image. This module

- tells whether the mask of a leaf branch reaches an XY face (``touches_xy_border``): the tip
  is within (``TIP_TOLERANCE_RADII`` x its vessel radius + 1) voxels of the face and the
  straight segment from the tip to the face lies in the mask;
- tells whether a leaf branch points to that face (``points_to_border``): its last
  ``END_RADII`` tip radii make an angle of at most ``BORDER_ANGLE`` degrees with the outward
  normal of the face. Spurs that the skeleton grows along z inside axially blurred vessels
  point along z, not to an XY face;
- marks the graph (``mark_border``): leaf tips whose mask reaches an XY face get
  ``border=True`` and are moved to the face, their path extended by the straight segment; every
  edge gets ``border_fraction``, the fraction of its path voxels whose mask reaches an XY face
  (edges that graze or run along the border have uncertain position and radius there).

The z faces are left out: in axially blurred stacks, the vessels near a z face mostly run along
it, and their labels touch it without being cut by it.

The mask is taken from a vessel radius map (``label_geometry.vessel_radius_map``): positive
inside the mask. Arrays are indexed (z, y, x) or (y, x).
"""

import networkx as nx
import numpy as np

from pyvane.graph.label_geometry import vessel_radius
from pyvane.graph.util import bresenham_3d, compute_arc_length

# A leaf points to the border if its last END_RADII tip radii make an angle of at most
# BORDER_ANGLE degrees with the outward normal of the nearest XY face (chosen on synthetic
# stacks with ground truth: 75 degrees kept too many spurs, 45 degrees fewer real cut vessels)
BORDER_ANGLE = 60.0
END_RADII = 3.0
# A tip reaches an XY face if it is within (TIP_TOLERANCE_RADII x its vessel radius + 1) voxels
# of it: the thinning stops the skeleton about one radius before the face, up to two in labels
# that are taller than wide (2 found slightly more vessels cut by the face than 1 on the
# synthetic stacks)
TIP_TOLERANCE_RADII = 2.0


def nearest_xy_face(voxel, shape) -> tuple[int, tuple, np.ndarray]:
    """The nearest XY face of the image to a voxel.

    Args:
        voxel: (z, y, x) or (y, x) voxel.
        shape: Image shape.

    Returns:
        A tuple (distance in voxels, the voxel of the face closest to ``voxel``, the outward
        unit normal of the face in (y, x)).
    """
    *z, y, x = (int(v) for v in voxel)
    height, width = shape[-2:]
    faces = [(y, (*z, 0, x), (-1.0, 0.0)), (height - 1 - y, (*z, height - 1, x), (1.0, 0.0)),
             (x, (*z, y, 0), (0.0, -1.0)), (width - 1 - x, (*z, y, width - 1), (0.0, 1.0))]
    distance, face_voxel, normal = min(faces)
    return distance, face_voxel, np.array(normal)


def _segment(start, end) -> np.ndarray:
    """Voxels of the straight segment from ``start`` to ``end``, both included."""
    start, end = np.asarray(start, np.int32), np.asarray(end, np.int32)
    if len(start) == 2:
        line = bresenham_3d(np.r_[0, start].astype(np.int32), np.r_[0, end].astype(np.int32))
        return line[:, 1:]
    return bresenham_3d(start, end)


def touches_xy_border(radius_map: np.ndarray, voxel) -> bool:
    """Whether the mask at ``voxel`` reaches an XY face: the voxel is within
    (``TIP_TOLERANCE_RADII`` x its vessel radius + 1) voxels of the face and the straight
    segment to the face lies in the mask."""
    distance, face_voxel, _ = nearest_xy_face(voxel, radius_map.shape)
    if distance > TIP_TOLERANCE_RADII * vessel_radius(radius_map, voxel) + 1:
        return False
    segment = _segment(voxel, face_voxel)
    return bool(np.all(radius_map[tuple(segment.T)] > 0))


def _leaf_points(graph: nx.MultiGraph, tip, parent, data: dict) -> np.ndarray:
    """Points of a leaf branch from the parent center to the tip center (paths are stored from
    the node of smaller index to the larger one)."""
    n_dims = len(graph.nodes[tip]["center"])
    path = np.asarray(data["path"], float).reshape(-1, n_dims)
    if tip < parent:
        path = path[::-1]
    return np.vstack([graph.nodes[parent]["center"], path, graph.nodes[tip]["center"]])


def points_to_border(graph: nx.MultiGraph, tip, parent, data: dict,
                     radius_map: np.ndarray, max_angle: float = BORDER_ANGLE) -> bool:
    """Whether the end of a leaf branch points to the nearest XY face: the vector over its last
    ``END_RADII`` tip radii makes an angle of at most ``max_angle`` degrees with the outward
    normal of the face (an end along z makes an angle near 90 degrees)."""
    tip_voxel = graph.nodes[tip]["center"]
    _, _, normal = nearest_xy_face(tip_voxel, radius_map.shape)
    points = _leaf_points(graph, tip, parent, data)
    back = max(len(points) - 1 - round(END_RADII * vessel_radius(radius_map, tip_voxel)), 0)
    end_vector = points[-1] - points[back]
    norm = np.linalg.norm(end_vector)
    if norm == 0:
        return False
    return bool(end_vector[-2:] @ normal >= np.cos(np.radians(max_angle)) * norm)


def is_border_leaf(graph: nx.MultiGraph, tip, parent, data: dict,
                   radius_map: np.ndarray) -> bool:
    """Whether a leaf branch reaches an XY face and points to it (see the module docstring):
    such a branch may be a vessel leaving the image and should not be pruned."""
    return (touches_xy_border(radius_map, graph.nodes[tip]["center"])
            and points_to_border(graph, tip, parent, data, radius_map))


def _extend_to_face(graph: nx.MultiGraph, tip, radius_map: np.ndarray) -> None:
    """Moves a leaf tip to the nearest XY face and extends its path by the straight segment."""
    ((u, v, key),) = graph.edges(tip, keys=True)
    parent = v if u == tip else u
    data = graph[u][v][key]
    old_center = np.asarray(graph.nodes[tip]["center"], np.int32)
    _, face_voxel, _ = nearest_xy_face(old_center, radius_map.shape)
    face_voxel = np.asarray(face_voxel, np.int32)
    if np.array_equal(face_voxel, old_center):
        return
    n_dims = len(old_center)
    path = np.asarray(data["path"], np.int32).reshape(-1, n_dims)
    if tip < parent:
        path = path[::-1]                                   # parent -> tip
    # the old center and the segment to the face, without the new center
    added = _segment(old_center, face_voxel)[:-1]
    path = np.vstack([path, added]).astype(np.int32)
    if tip < parent:
        path = path[::-1]
    graph.nodes[tip]["center"] = face_voxel
    data["path"] = np.ascontiguousarray(path)
    data["length"] = float(compute_arc_length(data["path"], u, v, graph))


def mark_border(graph: nx.MultiGraph, radius_map: np.ndarray) -> None:
    """Marks the vessels cut by the XY faces (in place).

    Every node gets ``border``: True for the leaf tips whose mask reaches an XY face
    (``touches_xy_border``), which are moved to the face, their path extended by the straight
    segment (the thinning stops the skeleton up to one radius before the face). Every edge gets
    ``border_fraction``: the fraction of its path voxels whose mask reaches an XY face.

    Args:
        graph: The graph to modify in-place.
        radius_map: Vessel radius map of the binary image the graph was made from.
    """
    for node in graph.nodes:
        is_border = (graph.degree(node) == 1 and not graph.has_edge(node, node)
                     and touches_xy_border(radius_map, graph.nodes[node]["center"]))
        graph.nodes[node]["border"] = bool(is_border)
        if is_border:
            _extend_to_face(graph, node, radius_map)
    height, width = radius_map.shape[-2:]
    for *_, data in graph.edges(data=True):
        path = np.asarray(data["path"], np.int32).reshape(-1, radius_map.ndim)
        if len(path) == 0:
            data["border_fraction"] = 0.0
            continue
        # only the voxels near a face need the segment test
        face_distance = np.min([path[:, -2], height - 1 - path[:, -2],
                                path[:, -1], width - 1 - path[:, -1]], axis=0)
        radii = np.maximum(radius_map[tuple(path.T)], 1.0)
        near = np.nonzero(face_distance <= TIP_TOLERANCE_RADII * radii + 1)[0]
        n_touching = sum(touches_xy_border(radius_map, path[i]) for i in near)
        data["border_fraction"] = n_touching / len(path)
