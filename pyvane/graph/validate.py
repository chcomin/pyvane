"""Consistency checks of a graph made by pyvane, for testing and debugging the refinement."""

import networkx as nx
import numpy as np

from pyvane.graph.util import find_inconsistent_edges


def check_graph(
        graph: nx.MultiGraph,
        bin_img: np.ndarray | None = None,
        min_inside: float = 0.9,
        allow_degree_two: bool = False,
        ) -> list[tuple[object, str]]:
    """Lists the violations of the invariants a refined graph should satisfy.

    Checked:
    - the path and length conventions of every edge (``find_inconsistent_edges``);
    - edge lengths are finite and positive;
    - no node of degree 0, and no node of degree 2 except the anchor of a pure ring (a node
      whose only edge is one self-loop), unless ``allow_degree_two``;
    - if ``bin_img`` is given, at least ``min_inside`` of the voxels of each edge path, and
      every node center, are foreground.

    Args:
        graph: Graph with node ``center`` and edge ``path`` and ``length`` attributes.
        bin_img: Optional binary image the graph was made from.
        min_inside: Minimum fraction of the path voxels of an edge that must be foreground.
        allow_degree_two: Do not report degree-2 nodes (e.g. on an intermediate graph).

    Returns:
        A list of (node or edge, message). Empty if the graph is valid.
    """
    problems: list[tuple[object, str]] = list(find_inconsistent_edges(graph))

    for u, v, key, data in graph.edges(keys=True, data=True):
        length = data.get("length")
        if length is None or not np.isfinite(length) or length <= 0:
            problems.append(((u, v, key), f"invalid length {length}"))

    for n in graph.nodes:
        deg = graph.degree(n)
        if deg == 0:
            problems.append((n, "isolated node"))
        elif deg == 2 and not allow_degree_two:
            is_ring = graph.number_of_edges(n, n) == 1
            if not is_ring:
                problems.append((n, "degree-2 node"))

    if bin_img is not None:
        fg = np.asarray(bin_img, bool)
        for n, c in graph.nodes(data="center"):
            if not fg[tuple(np.asarray(c, int))]:
                problems.append((n, "node center outside the mask"))
        for u, v, key, data in graph.edges(keys=True, data=True):
            path = np.asarray(data["path"], int).reshape(-1, fg.ndim)
            if len(path) == 0:
                continue
            inside = fg[tuple(path.T)].mean()
            if inside < min_inside:
                problems.append(((u, v, key), f"only {inside:.0%} of the path is foreground"))

    return problems
