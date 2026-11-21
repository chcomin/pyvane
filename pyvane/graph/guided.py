"""Pruning of the spurs that the skeleton grows along z in axially elongated labels, guided by
the graph of the label projected along z ("z-spur pruning").

With a low axial resolution (widefield or confocal 10x), vessels in the label are several
times taller than wide. The skeleton of such a cross-section is not a single centerline: it
grows short branches towards the top and bottom of the vessel, and their lengths overlap those
of real branches, so no length threshold separates them. The projection of the label along z
has no such spurs: a real vessel end or branch is still a tip of the projected graph. So a leaf
branch of the 3D graph is kept only if

- a tip of the projected graph matches it (within ``match_distance`` in XY, pointing the same
  way in XY), or
- it reaches at least ``xy_keep`` out of its parent vessel in XY (it would be a branch of the
  projected graph too; if it has no tip there, it is hidden in the projection, e.g. under
  another vessel), or
- its tip is within ``face_distance`` of the top or bottom face of the image (a vessel cut by
  the image; those cut by the XY borders end in the projected graph too).

The method was developed for confocal stacks of cortical capillaries
(``context/pyvane_handoff/guided_pruning`` in the analysis that introduced it), with distances
in voxels. Here every distance is relative to the vessel size, as in ``pyvane.graph.scaled``:
the unit of a leaf branch is the in-plane radius of the label at its parent junction
(``label_geometry.vessel_radius``), and the unit of the projected graph is the radius of the
projected label. Arrays are indexed (z, y, x).
"""

from typing import NamedTuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from scipy.ndimage import distance_transform_edt

from pyvane.graph.label_geometry import vessel_radius
from pyvane.graph.refinement import _extract_trash_path, remove_degree_two_nodes


class ProjectionGuide(NamedTuple):
    """Tips of the graph of the label projected along z.

    Attributes:
        tip_positions: (n, 2) YX positions of the tips.
        tip_directions: (n, 2) unit YX directions of the branch ends, pointing to the tips.
    """
    tip_positions: np.ndarray
    tip_directions: np.ndarray


def _points_from(graph: nx.MultiGraph, start, end, data: dict) -> np.ndarray:
    """Points of an edge from the center of ``start`` to the center of ``end`` (paths are
    stored from the node of smaller index to the larger one)."""
    n_dims = len(graph.nodes[start]["center"])
    path = np.asarray(data["path"], float).reshape(-1, n_dims)
    if end < start:
        path = path[::-1]
    return np.vstack([graph.nodes[start]["center"], path, graph.nodes[end]["center"]]).astype(float)


def _branch_end(graph: nx.MultiGraph, tip, parent, data: dict,
                direction_length: float) -> tuple[np.ndarray, np.ndarray]:
    """YX position of a tip and the unit YX direction of the last ``direction_length`` path
    points of its branch (voxels), pointing to the tip."""
    points = _points_from(graph, parent, tip, data)
    tip_position = points[-1, -2:]
    back_index = max(len(points) - 1 - round(direction_length), 0)
    direction = tip_position - points[back_index, -2:]
    norm = np.linalg.norm(direction)
    return tip_position, (direction / norm if norm else direction)


def projection_guide(bin_img: np.ndarray, prune_ratio_2d: float = 1.0,
                     direction_length: float = 3.0) -> ProjectionGuide:
    """Tips of the graph of the label projected along z (a pixel is foreground if any voxel
    of its column is).

    Args:
        bin_img: 3D binary image (z, y, x).
        prune_ratio_2d: Pruning of the projected graph, which removes the pixel noise of the
            2D skeleton: ``bulge_ratio_threshold`` of ``refine_graph`` (leaf branches reaching
            less than this many parent radii are removed); the other ``refine_graph``
            parameters keep their defaults.
        direction_length: Length of the branch end used for its direction, in units of the
            radius of the projected label at the branch's parent node.
    """
    # Imported here because pyvane.graph.creation imports pyvane.graph.refinement
    from pyvane.graph.creation import graph_from_skeleton, remove_clusters
    from pyvane.graph.refinement import refine_graph
    from pyvane.skeletonization import lee

    projection = np.asarray(bin_img, bool).any(axis=0)
    skeleton_2d = lee.skeletonize(projection.astype(np.uint8))
    edt_2d = distance_transform_edt(projection)
    condensed_2d = remove_clusters(graph_from_skeleton(skeleton_2d))
    graph_2d, _, _ = refine_graph(condensed_2d, projection, edt_bg=edt_2d,
                                  bulge_ratio_threshold=prune_ratio_2d)
    positions, directions = [], []
    for node in graph_2d.nodes:
        if graph_2d.degree(node) != 1:
            continue
        ((u, v, data),) = graph_2d.edges(node, data=True)
        parent = v if u == node else u
        parent_radius = vessel_radius(edt_2d, graph_2d.nodes[parent]["center"])
        position, direction = _branch_end(graph_2d, node, parent, data,
                                          direction_length * parent_radius)
        positions.append(position)
        directions.append(direction)
    return ProjectionGuide(np.array(positions, float).reshape(-1, 2),
                           np.array(directions, float).reshape(-1, 2))


def _junction_leaves(graph: nx.MultiGraph) -> list[tuple]:
    """(tip, parent, edge data) of the leaf branches attached to a node of degree >= 3."""
    leaves = []
    for u, v, data in graph.edges(data=True):
        degree_u, degree_v = graph.degree(u), graph.degree(v)
        if min(degree_u, degree_v) == 1 and max(degree_u, degree_v) >= 3:
            leaves.append((u, v, data) if degree_u == 1 else (v, u, data))
    return leaves


def _parent_radius(graph: nx.MultiGraph, parent, radius_map: np.ndarray) -> float:
    """Distance unit of a leaf branch: the in-plane radius at its parent junction."""
    return vessel_radius(radius_map, graph.nodes[parent]["center"])


def _match_tips(graph: nx.MultiGraph, radius_map: np.ndarray, guide: ProjectionGuide,
                kept_tips: set, claimed_guide_tips: set, match_distance: float,
                direction_length: float) -> None:
    """Matches the leaf branches not kept yet to the guide tips not claimed yet, greedily,
    nearest first. A pair needs the two branch ends to point the same way in XY (non-negative
    cosine). Distances are in units of the leaf's parent radius. Updates ``kept_tips`` (3D tip
    nodes) and ``claimed_guide_tips`` (guide indices)."""
    leaves = [leaf for leaf in _junction_leaves(graph) if leaf[0] not in kept_tips]
    if not leaves or not len(guide.tip_positions):
        return
    radii = np.array([_parent_radius(graph, parent, radius_map) for _, parent, _ in leaves])
    ends = [_branch_end(graph, tip, parent, data, direction_length * radius)
            for (tip, parent, data), radius in zip(leaves, radii)]
    leaf_positions = np.array([position for position, _ in ends])
    leaf_directions = np.array([direction for _, direction in ends])
    candidates = []
    guide_tree = cKDTree(guide.tip_positions)
    near_tips = guide_tree.query_ball_point(leaf_positions, match_distance * radii)
    for i, near in enumerate(near_tips):
        for j in near:
            if j not in claimed_guide_tips and leaf_directions[i] @ guide.tip_directions[j] >= 0:
                distance = np.linalg.norm(leaf_positions[i] - guide.tip_positions[j])
                candidates.append((distance, i, j))
    for _, i, j in sorted(candidates):
        tip = leaves[i][0]
        if tip not in kept_tips and j not in claimed_guide_tips:
            kept_tips.add(tip)
            claimed_guide_tips.add(j)


def _xy_reach(graph: nx.MultiGraph, tip, parent, data: dict, radius_map: np.ndarray) -> float:
    """How far a leaf branch reaches out of its parent vessel in XY (voxels): the largest XY
    distance of its points to the parent center, minus the parent radius, plus the tip radius
    (pyvane's bulge length, measured in XY). Radii are in-plane radii at the node centers."""
    points = _points_from(graph, parent, tip, data)
    reach = np.hypot(*(points[:, 1:] - points[0, 1:]).T).max()
    parent_radius = _parent_radius(graph, parent, radius_map)
    tip_radius = vessel_radius(radius_map, graph.nodes[tip]["center"])
    return float(reach - parent_radius + tip_radius)


def prune_z_spurs(graph: nx.MultiGraph, edt_bg: np.ndarray, radius_map: np.ndarray,
                  guide: ProjectionGuide, match_distance: float = 3.0, xy_keep: float = 4.0,
                  direction_length: float = 3.0, face_distance: float = 0.0,
                  keep_rings: bool = True) -> tuple[bool, list[np.ndarray]]:
    """Removes the leaf branches that have no counterpart in the projected graph (in place).

    Distances are in units of the in-plane radius of the label at the leaf's parent junction.
    Repeats until nothing changes:
    1. Each guide tip is matched to at most one leaf branch: the one whose tip is the nearest
       in XY, within ``match_distance``, and whose end points the same way in XY. Matched
       leaves are kept for good.
    2. At every node of degree >= 3, the leaf branches that are not matched are removed,
       except those reaching at least ``xy_keep`` out of their parent in XY and those whose tip
       is within ``face_distance`` of the top or bottom face. A node keeps at least two edges
       (the longest leaves are kept for that), as in ``prune_branches``.
    3. Nodes of degree 2 are merged.
    Only leaf branches are removed.

    Args:
        graph: Graph to modify in place.
        edt_bg: Euclidean distance transform of the background (3D), for the removed paths.
        radius_map: ``label_geometry.vessel_radius_map`` of the binary image.
        guide: ``projection_guide`` of the binary image the graph was made from.
        match_distance: Largest XY distance between a leaf tip and its guide tip.
        xy_keep: Leaves reaching at least this far out of their parent in XY are kept;
            np.inf disables the rule.
        direction_length: Length of the branch end used for its direction.
        face_distance: Leaves whose tip is this close to the top or bottom face are kept (0:
            only tips on the face).
        keep_rings: Passed to ``remove_degree_two_nodes``.

    Returns:
        A tuple (changed, trash_paths), as the other refinement steps: whether any branch was
        removed, and the removed paths outside the junction spheres.
    """
    depth = edt_bg.shape[0]
    kept_tips: set = set()
    claimed_guide_tips: set = set()
    trash_paths = []
    changed = False
    while True:
        _match_tips(graph, radius_map, guide, kept_tips, claimed_guide_tips, match_distance,
                    direction_length)
        edges_to_delete = []
        for node in list(graph.nodes):
            if graph.degree(node) <= 2:
                continue
            incident_edges = list(graph.edges(node, keys=True, data=True))
            radius = _parent_radius(graph, node, radius_map)
            removable = []      # (length, tip, u, v, key)
            for u, v, key, data in incident_edges:
                tip = v if u == node else u
                if graph.degree(tip) != 1 or tip in kept_tips:
                    continue
                tip_z = graph.nodes[tip]["center"][0]
                if (tip_z <= face_distance * radius
                        or tip_z >= depth - 1 - face_distance * radius):
                    continue
                if _xy_reach(graph, tip, node, data, radius_map) >= xy_keep * radius:
                    continue
                removable.append((data["length"], tip, u, v, key))
            if not removable:
                continue
            n_other_edges = len(incident_edges) - len(removable)
            # spare the longest leaves so that the node keeps two edges
            removable.sort(key=lambda leaf: -leaf[0])
            edges_to_delete += removable[max(0, 2 - n_other_edges):]
        if not edges_to_delete:
            break
        for _, tip, u, v, key in edges_to_delete:
            if graph.has_edge(u, v, key):
                trash_path = _extract_trash_path(graph, u, v, key, edt_bg)
                if len(trash_path):
                    trash_paths.append(trash_path)
                graph.remove_edge(u, v, key)
                if tip in graph and graph.degree(tip) == 0:
                    graph.remove_node(tip)
                changed = True
        remove_degree_two_nodes(graph, keep_rings=keep_rings)
    return changed, trash_paths
