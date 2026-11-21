"""Graph creation with every decision scaled to the local vessel size ("scaled" method).

An alternative to ``create_graph`` for labels whose vessels are taller than wide, as
widefield and confocal stacks give them (the axial blur), where the skeleton grows spurs along
z inside every cross-section and splits junctions. The skeleton graph is simplified only by
the decisions below, taken inside the ellipsoid of every junction; none of the heuristics of
``refine_graph`` is used. The ellipsoid's semi-axes are a multiple of the local in-plane
radius of the label, extended along z by the label's elongation (half-height / in-plane
radius, measured once per image). All parameters are therefore relative to the vessel size
and do not depend on the voxel size.

Each decision has its own parameter, so that precision and recall can be traded per kind of
structure:

- ``spur_reach`` (leaves): a leaf branch is kept only if it leaves the ellipsoid of its
  junction scaled by ``spur_reach``. Higher removes more spurs and more short real branches.
- ``junction_reach`` (junctions): junctions inside each other's ellipsoid scaled by
  ``junction_reach`` form one junction region, decided by the label components that cross
  the region's shell. Higher merges more split junctions and more close real junctions.
- ``tunnel_width`` (tunnels): edges that leave a region through the same label component
  towards the same node, and whose paths stay within ``tunnel_width`` label widths of each
  other, are one vessel split by a hole in the label; the most central path is kept. Higher
  merges more holes and more tight real loops or touching vessels.
- ``crossing_angle`` (crossings): a region with four components whose exit directions pair
  into two lines straighter than this angle is kept as one node flagged ``crossing`` (with
  ``crossing_pairs``), so that analyses can leave it out of the branch points.

Automatic values. The defaults were tuned on synthetic stacks with ground truth (widefield
10x and 20x, confocal 10x; exact masks and realistic labels; label elongation 0.7-5). Since
every distance is relative to the local vessel size and the measured elongation, one set of
values serves all of them, except ``spur_reach``, whose best value grows with the elongation:
``spur_reach="auto"`` (default) uses ``default_spur_reach(elongation)``. Alternatively,
``spur_reach="z_excess"`` searches for it per image without ground truth
(``choose_spur_reach``): spurs grown inside tall labels point along z, so the leaves of a good
graph are not more often along z than the vessels themselves are. ``graph_diagnostics``
reports that excess and other statistics, to check a result.

Typical use::

    graph = create_scaled_graph(bin_img)            # defaults
    prep = prepare_scaled(bin_img)                  # or: prepare once, build many times
    graph, stats = scaled_graph_from(prep, spur_reach=2.0, tunnel_width=3.0)
    graph, stats = scaled_graph_from(prep, spur_reach="z_excess")

Note: an isotropic image is assumed (as for ``create_graph``).
"""

import copy
from dataclasses import dataclass
from typing import NamedTuple

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from pyvane.graph.creation import graph_from_skeleton, remove_clusters
from pyvane.graph.refinement import (
    _bridge_from_center,
    _extract_trash_path,
    _line_between_centers,
    remove_degree_two_nodes,
)
from pyvane.graph.label_geometry import (  # noqa: F401  (also part of this module's API)
    MAX_RADIUS,
    best_pairing,
    half_height,
    inplane_edt,
    label_elongation,
    paths_close,
)
from pyvane.graph.util import compute_arc_length
from pyvane.skeletonization import lee

DEFAULTS = {"spur_reach": "auto", "junction_reach": 2.0, "tunnel_width": 2.0,
            "crossing_angle": 150.0}

# Internal constants of scaled_graph_from: how the ellipsoids extend along z (see
# _ellipsoid_axes), the margin added to every semi-axis (voxels) and the maximum number of
# rounds of spur pruning and junction-region decisions
Z_MODE = "add"
MARGIN = 1.0
MAX_ITER = 5

def default_spur_reach(elongation: float) -> float:
    """``spur_reach`` for a label of this elongation: 1.25 + 0.2 x (elongation - 1), with the
    elongation at least 1 (1.25 for round vessels, about 2 for confocal labels 5x taller than
    wide)."""
    return 1.25 + 0.2 * (max(1.0, elongation) - 1.0)


# ----- preparation ------------------------------------------------------------------------

@dataclass
class ScaledInput:
    """Everything that does not depend on the scaled-method parameters, for one binary image.

    Attributes:
        bin_img: The binary image.
        edt: Euclidean distance transform of the background (3D).
        edt_xy: In-plane distance transform (``inplane_edt``).
        graph: Skeleton graph, with clusters of junction voxels merged into one node and
            degree-2 nodes merged into their edges. No edge has been removed.
        elongation: Label elongation (``label_elongation``).
        keep_rings: Whether pure rings (isolated loops) are kept as one node and a self-loop.
    """
    bin_img: np.ndarray
    edt: np.ndarray
    edt_xy: np.ndarray
    graph: nx.MultiGraph
    elongation: float
    keep_rings: bool = True


def prepare_scaled(bin_img: np.ndarray, skel_img: np.ndarray | None = None,
                   skeleton_method: str = "fixed", elongation: float | None = None,
                   keep_rings: bool = True) -> ScaledInput:
    """Skeleton, graph, distance maps and label elongation of a binary image.

    Args:
        bin_img: 3D binary image (z, y, x), isotropic voxels.
        skel_img: Skeleton of ``bin_img``; computed with ``lee.skeletonize`` if None.
        skeleton_method: ``lee.skeletonize`` method, when ``skel_img`` is None.
        elongation: Label elongation; measured if None.
        keep_rings: Whether pure rings (isolated loops) are kept.

    Returns:
        A ``ScaledInput`` to pass to ``scaled_graph_from`` (any number of times).
    """
    bin_img = np.ascontiguousarray(bin_img, bool)
    if bin_img.ndim != 3:
        raise ValueError("The scaled method needs a 3D image (z, y, x).")
    if skel_img is None:
        skel_img = lee.skeletonize(bin_img, method=skeleton_method)
    edt = ndi.distance_transform_edt(bin_img)
    edt_xy = inplane_edt(bin_img)
    graph = remove_clusters(graph_from_skeleton(skel_img, keep_rings=keep_rings),
                            keep_rings=keep_rings)
    remove_degree_two_nodes(graph, keep_rings=keep_rings)
    if elongation is None:
        elongation = label_elongation(bin_img, edt_xy, np.argwhere(skel_img))
    return ScaledInput(bin_img, edt, edt_xy, graph, float(elongation), keep_rings)


# ----- junction ellipsoids ----------------------------------------------------------------

def _ellipsoid_axes(prep: ScaledInput, voxel, reach: float, margin: float,
                    z_mode: str) -> np.ndarray:
    """Semi-axes (z, y, x) of the ellipsoid of a junction at ``voxel``.

    In-plane: reach x local in-plane radius. Along z, the label is ``elongation`` times
    taller: with ``z_mode="add"``, (reach + elongation - 1) x radius (the blurred half-height
    plus the in-plane reach beyond the radius); with ``z_mode="mul"``, reach x elongation x
    radius. ``margin`` is added to every semi-axis.
    """
    radius = max(float(prep.edt_xy[tuple(voxel)]), 1.0)
    elongation = max(1.0, prep.elongation)
    if z_mode == "mul":
        z_axis = reach * elongation * radius
    else:
        z_axis = (reach + elongation - 1.0) * radius
    inplane_axis = reach * radius
    return np.array([z_axis + margin, inplane_axis + margin, inplane_axis + margin])


def _leaf_extent(graph: nx.MultiGraph, junction, tip, edge_data: dict,
                 axes: np.ndarray) -> float:
    """How far a leaf branch reaches from its junction: the largest distance of its points
    to the junction center, in units of the ellipsoid ``axes`` (1 = on the ellipsoid)."""
    center = np.asarray(graph.nodes[junction]["center"], float)
    path = np.asarray(edge_data["path"], float).reshape(-1, 3)
    tip_center = np.asarray(graph.nodes[tip]["center"], float)
    points = np.vstack([path, tip_center[None]])
    scaled_offsets = (points - center) / axes[None]
    return float(np.sqrt((scaled_offsets ** 2).sum(axis=1)).max())


def _record_removed(graph: nx.MultiGraph, u, v, key, prep: ScaledInput,
                    removed_paths: list | None) -> None:
    """Appends the path of an edge about to be removed to ``removed_paths`` (if a list)."""
    if removed_paths is None:
        return
    path = _extract_trash_path(graph, u, v, key, prep.edt)
    if len(path):
        removed_paths.append(path)


def prune_spurs(graph: nx.MultiGraph, prep: ScaledInput, spur_reach: float = 1.25,
                margin: float = 1.0, z_mode: str = "add", keep_rings: bool = True,
                removed_paths: list | None = None) -> int:
    """Removes the leaf branches that do not leave the ellipsoid of their junction.

    A junction keeps at least two edges (the leaves reaching farthest). Repeats until stable.

    Args:
        graph: Graph to modify in place.
        prep: ``prepare_scaled`` output of the image the graph was made from.
        spur_reach: Scale of the ellipsoids.
        margin: Added to every semi-axis (voxels).
        z_mode: See ``_ellipsoid_axes``.
        keep_rings: Passed to ``remove_degree_two_nodes``.
        removed_paths: If a list, the paths of the removed branches are appended to it
            (outside the junction spheres, as ``refine_graph``'s trash paths).

    Returns:
        The number of removed branches.
    """
    n_removed = 0
    changed = True
    while changed:
        changed = False
        edges_to_remove = []
        for node in list(graph.nodes):
            degree = graph.degree(node)
            if degree <= 2:
                continue
            center = np.asarray(graph.nodes[node]["center"], int)
            axes = _ellipsoid_axes(prep, center, spur_reach, margin, z_mode)
            leaves = []     # (extent, u, v, key)
            for u, v, key, data in graph.edges(node, keys=True, data=True):
                other = v if u == node else u
                if other != node and graph.degree(other) == 1:
                    extent = _leaf_extent(graph, node, other, data, axes)
                    leaves.append((extent, u, v, key))
            spurs = [leaf for leaf in leaves if leaf[0] <= 1.0]
            if not spurs:
                continue
            n_left = degree - len(spurs)
            if n_left < 2:
                # spare the spurs reaching farthest, so that the junction keeps two edges
                spurs.sort(key=lambda leaf: -leaf[0])
                spurs = spurs[2 - n_left:]
            edges_to_remove += [(u, v, key) for _, u, v, key in spurs]
        for u, v, key in edges_to_remove:
            if graph.has_edge(u, v, key):
                _record_removed(graph, u, v, key, prep, removed_paths)
                graph.remove_edge(u, v, key)
                n_removed += 1
                changed = True
        if changed:
            graph.remove_nodes_from([n for n in list(graph.nodes) if graph.degree(n) == 0])
            remove_degree_two_nodes(graph, keep_rings=keep_rings)
    return n_removed


# ----- junction regions -------------------------------------------------------------------

def _path_from(graph: nx.MultiGraph, start, end, key) -> np.ndarray:
    """Path of edge (start, end, key) oriented from ``start`` to ``end`` (centers excluded)."""
    path = np.asarray(graph[start][end][key]["path"], int).reshape(-1, 3)
    return path if start <= end else path[::-1]


def _rejoin(graph: nx.MultiGraph, node, other, path_from_node: np.ndarray):
    """Adds an edge node-other whose path starts next to the center of ``node``."""
    center = np.asarray(graph.nodes[node]["center"], np.int32)
    if len(path_from_node):
        path = _bridge_from_center(center, path_from_node.astype(np.int32), False)
        if other == node:
            # self-loop: the other end also starts at the center
            path = _bridge_from_center(center, path[::-1], False)[::-1]
    else:
        other_center = np.asarray(graph.nodes[other]["center"], np.int32)
        path = _line_between_centers(center, other_center, False)
    if node > other:
        path = path[::-1]
    path = np.asarray(path, np.int32).reshape(-1, 3)
    graph.add_edge(node, other, path=path,
                   length=float(compute_arc_length(path, node, other, graph)))


def _junction_regions(graph: nx.MultiGraph, junctions: list, axes: dict) -> list[list]:
    """Groups junctions that are inside each other's ellipsoid (``axes``), transitively."""
    centers = np.array([np.asarray(graph.nodes[n]["center"], float) for n in junctions])
    largest_axis = max(a.max() for a in axes.values())
    parent = list(range(len(junctions)))

    def find_root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(centers).query_pairs(largest_axis):
        offset = centers[j] - centers[i]
        distance_i = np.sqrt(((offset / axes[junctions[i]]) ** 2).sum())
        distance_j = np.sqrt(((offset / axes[junctions[j]]) ** 2).sum())
        if min(distance_i, distance_j) <= 1.0:
            root_i, root_j = find_root(i), find_root(j)
            if root_i != root_j:
                parent[root_i] = root_j
    regions: dict[int, list] = {}
    for i, node in enumerate(junctions):
        regions.setdefault(find_root(i), []).append(node)
    return list(regions.values())


def _region_center(graph: nx.MultiGraph, members: list, bin_img: np.ndarray,
                   edt: np.ndarray) -> np.ndarray:
    """Mean of the member centers; the center of the most central member (largest distance
    to the background) if the mean falls outside the label."""
    member_centers = [graph.nodes[n]["center"] for n in members]
    center = np.round(np.mean(member_centers, axis=0)).astype(int)
    if bin_img[tuple(center)]:
        return center

    def depth(node):
        return edt[tuple(np.asarray(graph.nodes[node]["center"], int))]

    return np.asarray(graph.nodes[max(members, key=depth)]["center"], int)


class _RegionShell:
    """Ellipsoid of a junction region and the label components on its shell (a thin layer
    just outside the ellipsoid), labelled in a crop of the image around the region."""

    def __init__(self, bin_img: np.ndarray, center: np.ndarray, semi_axes: np.ndarray):
        self.center = center
        self.semi_axes = semi_axes
        half_size = np.ceil(semi_axes).astype(int)
        start = np.maximum(center - half_size - 3, 0)
        end = np.minimum(center + half_size + 4, np.array(bin_img.shape))
        crop = bin_img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        zz, yy, xx = np.ogrid[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        distance = np.sqrt(((zz - center[0]) / semi_axes[0]) ** 2
                           + ((yy - center[1]) / semi_axes[1]) ** 2
                           + ((xx - center[2]) / semi_axes[2]) ** 2)
        # at least 1.5 voxels thick along the shortest axis
        thickness = max(1.5 / semi_axes.min(), 0.12)
        shell = crop & (distance >= 1.0) & (distance < 1.0 + thickness)
        self.components, _ = ndi.label(shell, structure=np.ones((3, 3, 3)))
        self.crop_start = start

    def distance(self, points) -> np.ndarray:
        """Distance of points to the center in units of the semi-axes (1 = on the ellipsoid)."""
        offsets = (np.asarray(points, float) - self.center) / self.semi_axes
        return np.sqrt((offsets ** 2).sum(axis=-1))

    def component_at(self, point) -> int | None:
        """Most frequent shell component within 2 voxels of ``point`` (None if there is
        none)."""
        local = np.asarray(point, int) - self.crop_start
        start = np.maximum(local - 2, 0)
        end = np.minimum(local + 3, self.components.shape)
        window = self.components[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        labels = window[window > 0]
        return int(np.bincount(labels).argmax()) if len(labels) else None


class _Exit(NamedTuple):
    """An edge from a region member to a node outside the region."""
    component: int | None       # shell component where the edge leaves (None: it never does)
    far_node: object
    member: object
    key: object
    path: np.ndarray            # from the member, centers excluded
    exit_point: np.ndarray | None
    extent: float               # largest ``_RegionShell.distance`` along the edge


def _region_edges(graph: nx.MultiGraph, members: list, shell: _RegionShell):
    """Edges of the region members: exits (``_Exit``) and internal edges (member to member,
    as (u, v, key, path from u))."""
    member_set = set(members)
    exits, internal = [], []
    for member in members:
        for u, v, key in list(graph.edges(member, keys=True)):
            other = v if u == member else u
            if member != other:
                path = _path_from(graph, member, other, key)
            else:
                path = np.asarray(graph[member][member][key]["path"], int).reshape(-1, 3)
            if other in member_set:
                internal.append((member, other, key, path))
                continue
            other_center = np.asarray(graph.nodes[other]["center"], int)
            path_to_other = np.vstack([path, other_center[None]])
            distances = shell.distance(path_to_other)
            extent = float(distances.max())
            outside = np.nonzero(distances >= 1.0)[0]
            if len(outside) == 0:
                exits.append(_Exit(None, other, member, key, path, None, extent))
                continue
            exit_point = path_to_other[outside[0]]
            exits.append(_Exit(shell.component_at(exit_point), other, member, key, path,
                               exit_point, extent))
    return exits, internal


def _merge_tunnels(exits: list, edt: np.ndarray, edt_xy: np.ndarray, elongation: float,
                   tunnel_width: float) -> tuple[list, list]:
    """Exits through the same shell component towards the same node, whose paths stay within
    ``tunnel_width`` label widths of the most central one (largest mean distance to the
    background), are dropped. Returns the kept and the dropped exits."""
    groups: dict = {}
    for exit_ in exits:
        # exits that never leave the region form a group of their own
        component = exit_.component if exit_.component is not None else ("own", id(exit_))
        groups.setdefault((component, exit_.far_node), []).append(exit_)

    def centrality(exit_):
        return -float(np.mean(edt[tuple(exit_.path.T)])) if len(exit_.path) else 0.0

    kept, dropped = [], []
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group.sort(key=centrality)
        main = group[0]
        kept.append(main)
        for exit_ in group[1:]:
            if tunnel_width > 0 and paths_close(main.path, exit_.path, edt_xy, elongation,
                                                tunnel_width):
                dropped.append(exit_)
            else:
                kept.append(exit_)
    return kept, dropped


def merge_junction_regions(graph: nx.MultiGraph, prep: ScaledInput,
                           junction_reach: float = 2.0, spur_reach: float = 1.25,
                           tunnel_width: float = 2.0, crossing_angle: float = 150.0,
                           margin: float = 1.0, z_mode: str = "add",
                           removed_paths: list | None = None) -> dict:
    """Decides every junction region by the branches that leave it (in place).

    Junctions inside each other's ellipsoid (``junction_reach``) form one region. Every edge
    leaving the region is assigned the label component it crosses on the region's shell. Then:

    - the region becomes one node (internal edges are dropped unless they leave and re-enter);
    - edges through the same shell component towards the same node, whose paths stay within
      ``tunnel_width`` label widths of each other, are one vessel: the most central path
      (highest mean distance to the background) is kept;
    - leaf edges that stay inside the region scaled to ``spur_reach`` are removed;
    - a region with 4 components whose exit directions pair into two lines (both pair angles
      above ``crossing_angle``) is flagged ``crossing``, with ``crossing_pairs``.

    Args:
        graph: Graph to modify in place.
        prep: ``prepare_scaled`` output of the image the graph was made from.
        junction_reach, spur_reach, tunnel_width, crossing_angle: See the module docstring.
        margin: Added to every semi-axis (voxels).
        z_mode: See ``_ellipsoid_axes``.
        removed_paths: If a list, the paths of the removed edges (spurs, tunnel paths,
            internal edges) are appended to it, as in ``prune_spurs``.

    Returns:
        Counts of the decisions: regions_merged, tunnel_paths, spurs, crossings,
        internal_dropped.
    """
    stats = {"regions_merged": 0, "tunnel_paths": 0, "spurs": 0, "crossings": 0,
             "internal_dropped": 0}
    junctions = [n for n in graph.nodes if graph.degree(n) >= 3]
    if not junctions:
        return stats
    junction_axes = {n: _ellipsoid_axes(prep, np.asarray(graph.nodes[n]["center"], int),
                                        junction_reach, margin, z_mode) for n in junctions}
    regions = _junction_regions(graph, junctions, junction_axes)

    # spurs are judged on the region ellipsoid scaled from junction_reach to spur_reach
    spur_extent = spur_reach / junction_reach
    elongation = max(1.0, prep.elongation)
    for members in regions:
        members = [n for n in members if n in graph]
        if not members:
            continue
        center = _region_center(graph, members, prep.bin_img, prep.edt)
        spread = np.max([np.abs(np.asarray(graph.nodes[n]["center"]) - center)
                         for n in members], axis=0)
        semi_axes = np.max([junction_axes[n] for n in members], axis=0) + spread
        shell = _RegionShell(prep.bin_img, center, semi_axes)
        exits, internal = _region_edges(graph, members, shell)

        spurs, not_spurs = [], []
        for exit_ in exits:
            is_spur = (exit_.exit_point is None and graph.degree(exit_.far_node) == 1
                       and exit_.extent <= spur_extent)
            (spurs if is_spur else not_spurs).append(exit_)
        stats["spurs"] += len(spurs)
        kept, tunnel_paths = _merge_tunnels(not_spurs, prep.edt, prep.edt_xy, elongation,
                                            tunnel_width)
        stats["tunnel_paths"] += len(tunnel_paths)
        # internal edges that leave the region and come back are kept as self-loops
        loops, dropped_internal = [], []
        for edge in internal:
            path = edge[3]
            leaves_region = len(path) and (shell.distance(path) >= 1.0).any()
            (loops if leaves_region else dropped_internal).append(edge)
        stats["internal_dropped"] += len(dropped_internal)
        for exit_ in spurs + tunnel_paths:
            _record_removed(graph, exit_.member, exit_.far_node, exit_.key, prep,
                            removed_paths)
        for u, v, key, _ in dropped_internal:
            _record_removed(graph, u, v, key, prep, removed_paths)

        n_components = len({e.component for e in kept if e.component is not None})
        maybe_crossing = n_components == 4 and len(kept) == 4
        changed = len(members) > 1 or len(kept) < len(exits) or len(loops) < len(internal)
        if not changed and not maybe_crossing:
            continue

        # rebuild the region as one node
        node = members[0]
        for member in members:
            for u, v, key in list(graph.edges(member, keys=True)):
                if graph.has_edge(u, v, key):
                    graph.remove_edge(u, v, key)
        for member in members[1:]:
            graph.remove_node(member)
        if len(members) > 1:
            stats["regions_merged"] += 1
        graph.nodes[node]["center"] = center.astype(np.int32)
        for exit_ in kept:
            _rejoin(graph, node, exit_.far_node, exit_.path)
        for _, _, _, path in loops:
            _rejoin(graph, node, node, path)

        if n_components == 4 and graph.degree(node) == 4:
            directions = np.array([(e.exit_point - center) / semi_axes for e in kept
                                   if e.exit_point is not None], float)
            if len(directions) == 4:
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                angle, pairs = best_pairing(directions)
                if angle >= crossing_angle:
                    far_nodes = [e.far_node for e in kept]
                    graph.nodes[node]["crossing"] = True
                    graph.nodes[node]["crossing_pairs"] = [(far_nodes[i], far_nodes[j])
                                                           for i, j in pairs]
                    stats["crossings"] += 1
    graph.remove_nodes_from([n for n in list(graph.nodes) if graph.degree(n) == 0])
    remove_degree_two_nodes(graph, keep_rings=prep.keep_rings)
    return stats


# ----- driver -----------------------------------------------------------------------------

def scaled_graph_from(prep: ScaledInput, spur_reach: float | str = DEFAULTS["spur_reach"],
                      junction_reach: float = DEFAULTS["junction_reach"],
                      tunnel_width: float = DEFAULTS["tunnel_width"],
                      crossing_angle: float = DEFAULTS["crossing_angle"],
                      removed_paths: list | None = None) -> tuple[nx.MultiGraph, dict]:
    """Simplifies a copy of ``prep.graph`` with the scaled-method decisions.

    Alternates ``prune_spurs`` and ``merge_junction_regions`` until nothing changes (at most
    ``MAX_ITER`` times), with the ellipsoid constants ``Z_MODE`` and ``MARGIN``.

    Args:
        prep: ``prepare_scaled`` output.
        spur_reach: A value, "auto" (``default_spur_reach`` of the label elongation) or
            "z_excess" (``choose_spur_reach``).
        junction_reach, tunnel_width, crossing_angle: See the module docstring.
        removed_paths: If a list, the paths of all removed edges are appended to it (for
            ``map_foreground_to_graph``).

    Returns:
        The graph and the summed decision counts (with the ``spur_reach`` used).
    """
    if spur_reach == "auto":
        reach = default_spur_reach(prep.elongation)
    elif spur_reach == "z_excess":
        reach, _ = choose_spur_reach(
            prep, junction_reach=junction_reach, tunnel_width=tunnel_width,
            crossing_angle=crossing_angle)
    else:
        reach = float(spur_reach)
    graph = copy.deepcopy(prep.graph)
    totals = {"spur_reach": float(reach), "spurs_pruned": 0}
    for _ in range(MAX_ITER):
        n_pruned = prune_spurs(graph, prep, reach, MARGIN, Z_MODE, prep.keep_rings,
                               removed_paths)
        counts = merge_junction_regions(graph, prep, junction_reach, reach, tunnel_width,
                                        crossing_angle, MARGIN, Z_MODE, removed_paths)
        totals["spurs_pruned"] += n_pruned
        for name, count in counts.items():
            totals[name] = totals.get(name, 0) + count
        graph_changed = (n_pruned or counts["regions_merged"] or counts["tunnel_paths"]
                         or counts["spurs"] or counts["internal_dropped"])
        if not graph_changed:
            break
    return graph, totals


def create_scaled_graph(bin_img: np.ndarray, **params) -> nx.MultiGraph:
    """Creates the graph of a binary image with the scaled-method decisions.

    Args:
        bin_img: 3D binary image (z, y, x), isotropic voxels.
        **params: ``scaled_graph_from`` parameters (defaults in ``DEFAULTS``).

    Returns:
        The graph (node ``center``, possibly ``crossing``; edge ``path`` and ``length``),
        in the format of ``create_graph``.
    """
    return scaled_graph_from(prepare_scaled(bin_img), **params)[0]


# ----- diagnostics without ground truth ----------------------------------------------------

def _along_z(vectors: np.ndarray, angle: float = 45.0) -> np.ndarray:
    """Whether each vector is within ``angle`` degrees of the z axis."""
    vectors = np.asarray(vectors, float).reshape(-1, 3)
    norms = np.linalg.norm(vectors, axis=1)
    return np.abs(vectors[:, 0]) >= np.cos(np.radians(angle)) * np.maximum(norms, 1e-9)


def graph_diagnostics(graph: nx.MultiGraph, prep: ScaledInput, border: float = 16.0,
                      spacing=(1.0, 1.0, 1.0), step: int = 4) -> dict:
    """Statistics of a graph that need no ground truth, measured in the interior.

    The interior excludes ``border`` voxels at the XY edges; leaves within their label
    half-height + 2 voxels of the z faces are left out as cut vessels.

    - ``leaf_z``: fraction of the leaves (tip - junction) within 45 deg of z.
    - ``edge_z``: length-weighted fraction of the path steps (``step`` points apart) of the
      junction-junction edges within 45 deg of z. Spurs grown inside axially elongated labels
      point along z, so ``z_excess = leaf_z - edge_z`` measures them.
    - densities per unit volume (of ``spacing``; per mm^3 when ``spacing`` is in um): branch
      points (crossings excluded), leaves, multi-edge pairs, self-loops, cycle rank, and
      length (mm / mm^3 when ``spacing`` is in um).
    - ``short_bridges``: fraction of junction-junction edges shorter than 2 x the sum of the
      in-plane radii of their nodes.

    Args:
        graph: Graph made from ``prep.bin_img``.
        prep: ``prepare_scaled`` output.
        border: XY border excluded (voxels).
        spacing: Voxel size, for the densities.
        step: Point spacing of the path steps for ``edge_z``.
    """
    shape = np.array(prep.bin_img.shape)
    border = float(border)
    spacing = np.asarray(spacing, float)
    interior_shape = np.r_[shape[0], np.maximum(shape[1:] - 2 * border, 1)]
    volume = float(np.prod(interior_shape * spacing)) * 1e-9

    def inside(point):
        return (border <= point[1] < shape[1] - border
                and border <= point[2] < shape[2] - border)

    centers = {n: np.asarray(d["center"], float) for n, d in graph.nodes(data=True)}
    degree = dict(graph.degree())
    branch_points = [n for n in graph if degree[n] >= 3 and inside(centers[n])
                     and not graph.nodes[n].get("crossing")]

    # leaves attached to a junction, away from the z faces
    n_leaves = n_leaves_along_z = 0
    for tip in graph:
        if degree[tip] != 1 or not inside(centers[tip]):
            continue
        junction = next(iter(graph.neighbors(tip)))
        if degree[junction] < 3:
            continue
        face_distance = half_height(prep.bin_img, centers[tip].astype(int)) + 2
        if centers[tip][0] < face_distance or centers[tip][0] > shape[0] - 1 - face_distance:
            continue
        n_leaves += 1
        n_leaves_along_z += bool(_along_z(centers[tip] - centers[junction])[0])

    length_along_z = length_steps = total_length = 0.0
    n_multi_edges = n_self_loops = 0
    multi_edge_pairs = set()
    n_short = n_junction_edges = 0
    for u, v, _, data in graph.edges(keys=True, data=True):
        path = np.asarray(data["path"], float).reshape(-1, 3)
        first, last = sorted((u, v))
        points = np.vstack([centers[first][None], path, centers[last][None]])
        points_inside = np.array([inside(p) for p in points])
        segments = np.diff(points, axis=0)
        segments_inside = points_inside[1:] & points_inside[:-1]
        total_length += float(np.linalg.norm(segments * spacing, axis=1)[segments_inside].sum())
        if u == v:
            n_self_loops += inside(centers[u])
            continue
        if ((first, last) not in multi_edge_pairs and graph.number_of_edges(u, v) > 1
                and inside(centers[u])):
            multi_edge_pairs.add((first, last))
            n_multi_edges += 1
        if degree[u] >= 3 and degree[v] >= 3:
            coarse = points[::step]
            if len(coarse) >= 2:
                steps = np.diff(coarse, axis=0)
                step_lengths = np.linalg.norm(steps, axis=1)
                steps_inside = points_inside[::step][1:]
                length_along_z += float(step_lengths[steps_inside & _along_z(steps)].sum())
                length_steps += float(step_lengths[steps_inside].sum())
            if inside(centers[u]):
                n_junction_edges += 1
                radius_u = min(float(prep.edt_xy[tuple(centers[u].astype(int))]), MAX_RADIUS)
                radius_v = min(float(prep.edt_xy[tuple(centers[v].astype(int))]), MAX_RADIUS)
                end_to_end = float(np.linalg.norm(points[-1] - points[0]))
                if end_to_end < 2 * (radius_u + radius_v):
                    n_short += 1
    cycle_rank = (graph.number_of_edges() - graph.number_of_nodes()
                  + nx.number_connected_components(nx.Graph(graph)))
    leaf_z = n_leaves_along_z / n_leaves if n_leaves else float("nan")
    edge_z = length_along_z / length_steps if length_steps else float("nan")
    return {"leaf_z": leaf_z, "edge_z": edge_z, "z_excess": leaf_z - edge_z,
            "n_leaves": n_leaves, "bp_density": len(branch_points) / volume,
            "leaf_density": n_leaves / volume,
            "length_density": total_length * 1e-3 / volume,
            "multi_density": n_multi_edges / volume, "loop_density": n_self_loops / volume,
            "cycle_density": cycle_rank / volume,
            "short_bridges": n_short / n_junction_edges if n_junction_edges else float("nan"),
            "elongation": prep.elongation}


def choose_spur_reach(prep: ScaledInput, candidates=(1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
                      max_z_excess: float = 0.1, border: float = 16.0,
                      **params) -> tuple[float, nx.MultiGraph]:
    """Smallest ``spur_reach`` whose graph has no excess of leaves along z.

    Real vessel ends point along z about as often as the vessels run along z; spurs grown
    inside axially elongated labels nearly always point along z. The first candidate whose
    ``graph_diagnostics`` ``z_excess`` is at most ``max_z_excess`` is chosen (the last one if
    none is). Labels without axial elongation have no such spurs and get the first candidate.

    Args:
        prep: ``prepare_scaled`` output.
        candidates: Increasing ``spur_reach`` values to try.
        max_z_excess: Accepted excess of leaves along z.
        border: See ``graph_diagnostics``.
        **params: Other ``scaled_graph_from`` parameters.

    Returns:
        The chosen value and its graph.
    """
    for spur_reach in candidates:
        graph, _ = scaled_graph_from(prep, spur_reach=spur_reach, **params)
        z_excess = graph_diagnostics(graph, prep, border=border)["z_excess"]
        if not np.isfinite(z_excess) or z_excess <= max_z_excess:
            return spur_reach, graph
    return candidates[-1], graph  # type: ignore[possibly-undefined]
