"""The "smoothed" edge measurement (``pyvane.metrics.metrics.measure_edges``): smoothed paths
for length and tortuosity, and in-plane radii.

- Voxel paths are staircases: their arc length overestimates vessel length by 7-11 % and
  tortuosity by 11-14 % on exact masks. Gaussian smoothing of the coordinates along the path
  (1 point) removes this. In labels that the axial blur makes taller than wide, the path also
  wanders up and down inside the label, which needs more smoothing (``auto_sigma``).
- The skeleton of a vessel that ends inside the image stops about one radius before the end
  of the label; leaf edges are continued straight to the label surface.
- The in-plane distance to the background is the vessel radius whatever the vessel's
  orientation (a tilted tube cuts every slice in an ellipse of minor semi-axis r) and is not
  inflated by the axial blur, while the 3D distance is limited by the label's z extent.
"""

import networkx as nx
import numpy as np
from scipy import ndimage as ndi

from pyvane.graph.label_geometry import MAX_RADIUS, inplane_edt, label_elongation
# Smallest radius assigned to a point inside the label
MIN_RADIUS = 0.5


def edge_centerline(graph: nx.MultiGraph, u, v, key) -> np.ndarray:
    """Points of an edge from the center of node min(u, v), along the edge path, to the center
    of node max(u, v), as float voxel coordinates (n, 3)."""
    first_node, last_node = sorted((u, v))
    path = np.asarray(graph[u][v][key]["path"], float).reshape(-1, 3)
    first_center = np.asarray(graph.nodes[first_node]["center"], float)
    last_center = np.asarray(graph.nodes[last_node]["center"], float)
    return np.vstack([first_center[None], path, last_center[None]])


def smooth_path(points: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of the coordinates along a path, with the ends fixed.

    Args:
        points: (n, 3) points.
        sigma: Standard deviation in points; 0 returns the points unchanged.
    """
    points = np.asarray(points, float)
    if sigma <= 0 or len(points) <= 2:
        return points
    # Odd reflection about the ends keeps them in place and the path straight there
    pad = min(int(np.ceil(4 * sigma)), len(points) - 1)
    head = 2 * points[0] - points[pad:0:-1]
    tail = 2 * points[-1] - points[-2:-pad - 2:-1]
    padded = np.vstack([head, points, tail])
    smoothed = ndi.gaussian_filter1d(padded, sigma, axis=0, mode="nearest")
    smoothed = smoothed[pad:pad + len(points)]
    smoothed[0], smoothed[-1] = points[0], points[-1]
    return smoothed


def auto_sigma(elongation: float) -> float:
    """Smoothing (points) for a label of this elongation: max(1, elongation / 2)."""
    return max(1.0, 0.5 * elongation)


def path_length(points: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """Arc length of a polyline, in units of ``spacing``."""
    steps = np.diff(points, axis=0) * np.asarray(spacing, float)
    return float(np.linalg.norm(steps, axis=1).sum())


def point_radii(points: np.ndarray, bin_img: np.ndarray, edt_xy: np.ndarray) -> np.ndarray:
    """In-plane radius at each point (nearest voxel); NaN for points outside the label."""
    voxels = np.round(points).astype(int)
    voxels = tuple(np.clip(voxels, 0, np.array(bin_img.shape) - 1).T)
    radii = np.minimum(edt_xy[voxels], MAX_RADIUS).astype(float)
    return np.where(bin_img[voxels], np.maximum(radii, MIN_RADIUS), np.nan)


def _node_radius(graph: nx.MultiGraph, node, bin_img: np.ndarray, edt_xy: np.ndarray,
                 edge_radii: np.ndarray) -> float:
    """In-plane radius at a node's center. If the center is outside the label, the largest
    radius along the edge is used instead (1 if the edge has none)."""
    center = tuple(np.asarray(graph.nodes[node]["center"], int))
    if bin_img[center]:
        return min(float(edt_xy[center]), MAX_RADIUS)
    if np.isfinite(edge_radii).any():
        return float(np.nanmax(edge_radii))
    return 1.0


def _outside_junctions(graph: nx.MultiGraph, points: np.ndarray, end_nodes, bin_img: np.ndarray,
                       edt_xy: np.ndarray, radii: np.ndarray, elongation: float) -> np.ndarray:
    """Mask of the points that are not inside the junction at either end of the edge.

    A junction (end node of degree > 1) covers the points closer to its center than its
    in-plane radius, with the z distance divided by the elongation (the label is taller than
    wide by that factor)."""
    z_scale = 1.0 / max(1.0, elongation)
    outside = np.ones(len(points), bool)
    for node in end_nodes:
        if graph.degree(node) <= 1:
            continue
        center = np.asarray(graph.nodes[node]["center"], float)
        junction_radius = _node_radius(graph, node, bin_img, edt_xy, radii)
        scaled_offsets = (points - center) * [z_scale, 1.0, 1.0]
        outside &= np.linalg.norm(scaled_offsets, axis=1) > junction_radius
    return outside


def _radius_stats(radii: np.ndarray, outside_junctions: np.ndarray) -> tuple[float, ...]:
    """Mean, min and max radius of the points outside the junctions, or of all points if none
    of those is in the label; NaN if no point is in the label."""
    selected = radii[outside_junctions]
    if not np.isfinite(selected).any():
        selected = radii
    if not np.isfinite(selected).any():
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(selected)), float(np.nanmin(selected)), float(np.nanmax(selected))


def _extend_to_label_surface(points: np.ndarray, bin_img: np.ndarray, max_distance: float,
                             n_direction: int = 5, step: float = 0.5) -> np.ndarray | None:
    """Point where the path, continued straight beyond its last point, leaves the label or the
    image.

    The direction is that from the point ``n_direction`` positions before the end to the last
    point. Returns None if the path cannot be continued (no direction, or the next step is
    already outside the label).

    Args:
        points: (n, 3) path ending at the tip to extend.
        bin_img: The binary image.
        max_distance: Maximum extension (voxels).
        n_direction: Number of points over which the direction is measured.
        step: Step of the march along the direction (voxels).
    """
    tip = points[-1]
    direction = tip - points[max(len(points) - 1 - n_direction, 0)]
    norm = np.linalg.norm(direction)
    if norm < 1e-6 or max_distance <= 0:
        return None
    direction = direction / norm
    shape = np.array(bin_img.shape)
    distance = 0.0
    while distance + step <= max_distance:
        voxel = np.round(tip + (distance + step) * direction).astype(int)
        if np.any(voxel < 0) or np.any(voxel >= shape) or not bin_img[tuple(voxel)]:
            break
        distance += step
    return tip + distance * direction if distance > 0 else None


def _extend_leaf_tips(graph: nx.MultiGraph, points: np.ndarray, first_node, last_node,
                      bin_img: np.ndarray, max_distance: float) -> np.ndarray:
    """Continues the ends of the path that are leaves (degree 1) to the label surface."""
    if graph.degree(last_node) == 1:
        new_end = _extend_to_label_surface(points, bin_img, max_distance)
        if new_end is not None:
            points = np.vstack([points, new_end[None]])
    if graph.degree(first_node) == 1:
        new_start = _extend_to_label_surface(points[::-1], bin_img, max_distance)
        if new_start is not None:
            points = np.vstack([new_start[None], points])
    return points


def measure_edge(graph: nx.MultiGraph, u, v, key, bin_img: np.ndarray, edt_xy: np.ndarray,
                 elongation: float = 1.0, sigma: float = 1.0, extend_tips: bool = True,
                 pix_size=(1.0, 1.0, 1.0)) -> dict:
    """Geometry of one edge; see ``measure_edges`` for the arguments and the returned keys."""
    first_node, last_node = sorted((u, v))
    is_loop = first_node == last_node
    # in-plane radii are distances within a z slice
    radius_scale = float(np.mean(np.asarray(pix_size, float)[1:]))
    points = smooth_path(edge_centerline(graph, u, v, key), sigma)
    point_radius = point_radii(points, bin_img, edt_xy)
    outside = _outside_junctions(graph, points, (first_node, last_node), bin_img, edt_xy,
                                 point_radius, elongation)
    mean_radius, min_radius, max_radius = _radius_stats(point_radius, outside)

    # The skeleton stops about one radius before the end of a vessel
    if extend_tips and not is_loop and np.isfinite(mean_radius):
        points = _extend_leaf_tips(graph, points, first_node, last_node, bin_img,
                                   max_distance=2 * mean_radius)

    path = np.asarray(graph[u][v][key]["path"]).reshape(-1, 3)
    length = path_length(points, pix_size)
    end_to_end = path_length(points[[0, -1]], pix_size)
    if is_loop or end_to_end <= 1e-9:
        tortuosity = float("nan")
    else:
        tortuosity = length / end_to_end
    return {"points": points,
            "radii": point_radii(path, bin_img, edt_xy) * radius_scale,
            "mean_radius": mean_radius * radius_scale,
            "min_radius": min_radius * radius_scale,
            "max_radius": max_radius * radius_scale,
            "measured_length": length,
            "tortuosity": tortuosity}


def measure_edges(graph: nx.MultiGraph, bin_img: np.ndarray, pix_size=(1.0, 1.0, 1.0),
                  edt_xy: np.ndarray | None = None, elongation: float | None = None,
                  sigma: float | str = "auto", extend_tips: bool = True) -> None:
    """Stores the smoothed geometry and in-plane radii of every edge (in place).

    Edge attributes written (the same as the "voxel" measurement):
    - ``points``: (n, 3) float voxels from the center of min(u, v) to the center of
      max(u, v), smoothed, with leaf tips extended to the label surface;
    - ``radii``: in-plane radius at each voxel of ``path`` (NaN off the label);
    - ``mean_radius``, ``min_radius``, ``max_radius``: of the in-plane radius along
      ``points``, away from the junctions (points farther than the junction's in-plane
      radius from its center, z divided by the elongation);
    - ``measured_length``: arc length of ``points``;
    - ``tortuosity``: ``measured_length`` / end-to-end distance (NaN for self-loops).
    Radii and lengths are in units of ``pix_size`` (radii, measured within z slices, use the
    in-plane pixel size). Also sets ``graph.graph["elongation"]``.

    Args:
        graph: Graph with node ``center`` and edge ``path``.
        bin_img: The 3D binary image the graph was made from.
        pix_size: Voxel size.
        edt_xy: In-plane distance transform of ``bin_img`` (``label_geometry.inplane_edt``);
            computed if None.
        elongation: Label elongation (``label_geometry.label_elongation``); taken from
            ``graph.graph["elongation"]`` or measured at the edge paths if None.
        sigma: Smoothing in points, or "auto" (``auto_sigma``).
        extend_tips: Continue leaf edges to the label surface (at most two mean radii).
    """
    if edt_xy is None:
        edt_xy = inplane_edt(bin_img)
    if elongation is None:
        elongation = graph.graph.get("elongation")
    if elongation is None:
        paths = [np.asarray(d["path"]).reshape(-1, 3) for *_, d in graph.edges(data=True)]
        voxels = np.vstack(paths) if paths else np.zeros((0, 3), int)
        elongation = label_elongation(bin_img, edt_xy, voxels.astype(int))
    graph.graph["elongation"] = float(elongation)
    if sigma == "auto":
        sigma = auto_sigma(elongation)
    for u, v, key in graph.edges(keys=True):
        geometry = measure_edge(graph, u, v, key, bin_img, edt_xy, elongation, sigma,
                                extend_tips, pix_size)
        graph[u][v][key].update(geometry)
