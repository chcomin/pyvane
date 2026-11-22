"""Runs pyvane's refine method on a label and summarizes the graph."""

from dataclasses import dataclass

import networkx as nx
import numpy as np

from pyvane.graph.creation import create_graph, graph_from_skeleton, remove_clusters
from pyvane.graph.refinement import JUNCTION_REACH, SPUR_REACH, TUNNEL_WIDTH, refine_graph
from pyvane.graph.validate import check_graph
from pyvane.skeletonization import lee

DEFAULT_KNOBS = {"spur_reach": SPUR_REACH, "junction_reach": JUNCTION_REACH,
                 "tunnel_width": TUNNEL_WIDTH}


@dataclass
class GraphResult:
    """The refined graph of a label and what is needed to draw and check it.

    Attributes:
        label: The binary label.
        skeleton: Its skeleton (``lee.skeletonize(method="fixed")``).
        graph: The refined graph.
        removed_paths: The paths removed by the refinement, outside the junction spheres.
        counts: ``graph_counts`` of the graph.
        problems: ``check_graph`` of the graph and label (empty if the graph is valid).
        settings: The knob values and ``keep_rings``.
    """
    label: np.ndarray
    skeleton: np.ndarray
    graph: nx.MultiGraph
    removed_paths: list
    counts: dict
    problems: list
    settings: dict


def skeletonize(label: np.ndarray) -> np.ndarray:
    """Skeleton of a label with the corrected thinning (no lost even-width ends)."""
    return lee.skeletonize(label.astype(np.uint8), method="fixed").astype(bool)


def graph_counts(graph: nx.MultiGraph) -> dict:
    """Counts of the structures each heuristic changes."""
    degrees = dict(graph.degree())
    multi_pairs = {(min(u, v), max(u, v)) for u, v in graph.edges()
                   if u != v and graph.number_of_edges(u, v) > 1}
    return {
        "branch_points": sum(degree >= 3 for degree in degrees.values()),
        "leaves": sum(degree == 1 for degree in degrees.values()),
        "border_leaves": sum(bool(border) for _, border in graph.nodes(data="border")),
        "multi_pairs": len(multi_pairs),
        "self_loops": nx.number_of_selfloops(graph),
        "edges": graph.number_of_edges(),
        "length": round(sum(length for *_, length in graph.edges(data="length")), 1),
    }


def build_graph(label: np.ndarray, skeleton: np.ndarray | None = None, keep_rings: bool = True,
                **knobs) -> GraphResult:
    """Refined graph of a label (``create_graph`` with the refine method).

    Args:
        label: Binary 2D or 3D label.
        skeleton: Its skeleton, if already computed.
        keep_rings: Passed to ``create_graph``.
        **knobs: ``spur_reach``, ``junction_reach``, ``tunnel_width``; the others keep their
            defaults.
    """
    if skeleton is None:
        skeleton = skeletonize(label)
    graph, _, removed_paths = create_graph(skeleton, label, keep_rings=keep_rings,
                                           full_output=True, **knobs)
    settings = {**DEFAULT_KNOBS, **knobs, "keep_rings": keep_rings}
    return GraphResult(label, skeleton, graph, list(removed_paths), graph_counts(graph),
                       check_graph(graph, label), settings)


def knob_sweep(label: np.ndarray, knob: str, values, **fixed) -> dict[float, GraphResult]:
    """Refined graphs of one label for several values of one knob.

    Args:
        label: Binary 2D or 3D label.
        knob: "spur_reach", "junction_reach" or "tunnel_width".
        values: Values of the knob; 0 turns its step off.
        **fixed: Other settings, the same for every value (the remaining knobs keep their
            defaults).

    Returns:
        A dict {value: GraphResult}.
    """
    skeleton = skeletonize(label)
    return {value: build_graph(label, skeleton, **fixed, **{knob: value}) for value in values}


@dataclass
class LeafNote:
    """What the leaf rule sees of one leaf branch (``leaf_notes``).

    Attributes:
        center: Tip center.
        reach: In-plane reach out of the parent, in parent radii (``guided._xy_reach``).
        guided: Matched to a tip of the projection guide (3D).
        border: Exempt as a branch reaching an XY face and pointing to it.
    """
    center: np.ndarray
    reach: float
    guided: bool
    border: bool

    def text(self) -> str:
        """Short text for a figure: reach in radii, G if guided, B if border."""
        flags = ("G" if self.guided else "") + ("B" if self.border else "")
        return f"{self.reach:.1f}" + (f" {flags}" if flags else "")


def leaf_notes(result: GraphResult) -> tuple[list[LeafNote], np.ndarray]:
    """The quantities the leaf rule (``guided.prune_leaves``) uses, for the leaves attached to
    a junction of a graph (normally made with ``spur_reach=0``, so that every leaf is there).

    The guide matching is the first pass of ``prune_leaves`` (with its default distances), on
    the final graph, whose border tips were already moved to the face: the values can differ
    slightly from those seen during the refinement.

    Returns:
        The notes and the (n, 2) YX positions of the projection guide tips (empty in 2D).
    """
    # private helpers of pyvane, used here to show what the rule measures
    from pyvane.graph.border import is_border_leaf
    from pyvane.graph.guided import (
        _junction_leaves,
        _match_tips,
        _parent_radius,
        _xy_reach,
        projection_guide,
    )
    from pyvane.graph.label_geometry import vessel_radius_map

    graph, label = result.graph, result.label
    radius_map = vessel_radius_map(label)
    guided_tips: set = set()
    guide_tips = np.zeros((0, 2))
    if label.ndim == 3:
        guide = projection_guide(label)
        guide_tips = guide.tip_positions
        _match_tips(graph, radius_map, guide, guided_tips, set(), match_distance=3.0,
                    direction_length=3.0)
    notes = []
    for tip, parent, data in _junction_leaves(graph):
        reach = (_xy_reach(graph, tip, parent, data, radius_map)
                 / _parent_radius(graph, parent, radius_map))
        notes.append(LeafNote(np.asarray(graph.nodes[tip]["center"]), reach,
                              tip in guided_tips,
                              is_border_leaf(graph, tip, parent, data, radius_map)))
    return notes, guide_tips


def trace_steps(label: np.ndarray, keep_rings: bool = True, **knobs) -> list[dict]:
    """Counts of the graph after every step of every iteration of ``refine_graph``.

    The steps are "degree_two", "false_anastomoses", "leaves", "collapse" and "tunnels"
    (``refine_graph`` repeats them until nothing changes). The final border marking is not a
    step. Rows are kept only for the steps that changed the graph, plus the initial graph.

    Returns:
        A list of rows {"iteration", "step", "removed_paths", **graph_counts}.
    """
    condensed = remove_clusters(graph_from_skeleton(skeletonize(label), keep_rings=keep_rings),
                                keep_rings=keep_rings)
    rows = [{"iteration": 0, "step": "condensed skeleton", "removed_paths": 0,
             **graph_counts(condensed)}]
    iteration = 0

    def record(step, graph, changed, trash_paths):
        nonlocal iteration
        if step == "degree_two":
            iteration += 1
        if changed:
            rows.append({"iteration": iteration, "step": step,
                         "removed_paths": len(trash_paths), **graph_counts(graph)})

    refine_graph(condensed, label, keep_rings=keep_rings, step_callback=record,
                 **{**DEFAULT_KNOBS, **knobs})
    return rows
