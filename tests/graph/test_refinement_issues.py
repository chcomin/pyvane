"""Minimal reproductions of the problems in pyvane/graph listed in ../pyvane_issues.md.

Each test fails on pyvane commit 37b5db0 and should pass once the problem is fixed. Run with
    pytest test_refinement_issues.py
or, without pytest,
    python test_refinement_issues.py
(run from the repository root; pyproject.toml puts it on the import path.)

Conventions used by pyvane (graph/creation.py, graph/util.py): node `center` and edge `path`
are integer coordinates (Z, Y, X); the path of an edge excludes the node centres and is stored
from the node of smaller index to the node of larger index; `length` is the arc length from
the centre of one node, along the path, to the centre of the other.
"""

import networkx as nx
import numpy as np
from pyvane.graph.refinement import collapse_internal_bridges, remove_degree_two_nodes
from pyvane.graph.util import calculate_edge_lengths


def line(x0, x1):
    """Path along X from x0 to x1 (inclusive), at Z = Y = 0."""
    step = 1 if x1 >= x0 else -1
    return np.array([[0, 0, x] for x in range(x0, x1 + step, step)], np.int32)


def true_length(graph, u, v, path):
    """Arc length centre(min) -> path -> centre(max), the pyvane convention."""
    a, b = min(u, v), max(u, v)
    pts = np.vstack(
        [graph.nodes[a]["center"], np.asarray(path).reshape(-1, 3), graph.nodes[b]["center"]]
    )
    return float(np.linalg.norm(np.diff(pts.astype(float), axis=0), axis=1).sum())


def test_remove_degree_two_nodes_length():
    """Issue 1: merging a degree-2 node whose neighbours u, w come with u > w.

    Nodes on a line: 2 at x=0, 1 at x=4, 0 at x=8. Removing node 1 gives an edge 0-2 of
    length 8. pyvane calls compute_arc_length(new_path, u=2, w=0) with the path stored from 0
    to 2, so the gaps are measured to the wrong ends: 7 + 6 + 7 = 20.
    """
    g = nx.MultiGraph()
    g.add_node(2, center=np.array([0, 0, 0]))
    g.add_node(1, center=np.array([0, 0, 4]))
    g.add_node(0, center=np.array([0, 0, 8]))
    g.add_edge(1, 2, path=line(3, 1), length=4.0)  # from 1 (x=4) to 2 (x=0)
    g.add_edge(0, 1, path=line(7, 5), length=4.0)  # from 0 (x=8) to 1 (x=4)
    remove_degree_two_nodes(g)
    ((u, v, data),) = g.edges(data=True)
    assert abs(data["length"] - true_length(g, u, v, data["path"])) < 1e-6, data["length"]
    assert abs(data["length"] - 8.0) < 1e-6, data["length"]


def bridge_graph(with_self_loop=False):
    """Two branch points joined by a short bridge: node 5 at x=10 and node 6 at x=12 (y=15),
    each with two leaves along Y. When 6 is merged into 5, its edges go to 5: one to node 3
    (neighbour < 5) and one to node 8 (neighbour > 5).
    """
    g = nx.MultiGraph()
    g.graph["is_2d"] = False
    centers = {
        5: (0, 15, 10),
        6: (0, 15, 12),
        1: (0, 25, 10),
        9: (0, 5, 10),
        3: (0, 25, 12),
        8: (0, 5, 12),
    }
    for n, c in centers.items():
        g.add_node(n, center=np.array(c))
    for leaf, parent in ((1, 5), (9, 5), (3, 6), (8, 6)):
        y_leaf, x = centers[leaf][1], centers[leaf][2]
        step = 1 if y_leaf > 15 else -1
        path = np.array(
            [[0, y, x] for y in range(15 + step, y_leaf, step)], np.int32
        )  # parent -> leaf
        g.add_edge(leaf, parent, path=path if parent < leaf else path[::-1])
    g.add_edge(5, 6, path=np.array([[0, 15, 11]], np.int32))
    if with_self_loop:
        loop = np.array(
            [[0, 16, 9], [0, 17, 8], [0, 18, 8], [0, 19, 9], [0, 18, 10], [0, 17, 10], [0, 16, 10]],
            np.int32,
        )
        g.add_edge(5, 5, path=loop)
    calculate_edge_lengths(g)
    return g


EDT = np.full((1, 32, 32), 5.0)  # radius 5 everywhere: the bridge (length 2) is collapsed


def test_collapse_internal_bridges_transferred_edge_length():
    """Issue 1 (second call site): the edges moved from v to u when a bridge is collapsed get
    their length from compute_arc_length(new_path, u, neighbor) after the path was reversed to
    go from min(u, neighbor) to max(u, neighbor). With neighbor < u the ends are swapped.
    """
    g = bridge_graph()
    collapse_internal_bridges(g, EDT, collapse_length_ratio_threshold=1.0)
    assert 6 not in g, "the bridge was not collapsed"
    wrong = [
        (u, v, round(d["length"], 3), round(true_length(g, u, v, d["path"]), 3))
        for u, v, d in g.edges(data=True)
        if abs(d["length"] - true_length(g, u, v, d["path"])) > 1e-6
    ]
    assert not wrong, f"(u, v, stored length, true length): {wrong}"


def test_collapse_internal_bridges_self_loop_length():
    """Issue 2: when the bridge is collapsed, node u moves to the midpoint of u and v, but a
    self-loop of u keeps its old path and length, so its stored length no longer matches its
    path and the new centre.
    """
    g = bridge_graph(with_self_loop=True)
    collapse_internal_bridges(g, EDT, collapse_length_ratio_threshold=1.0)
    assert 6 not in g, "the bridge was not collapsed"
    loops = [(u, d) for u, v, d in g.edges(data=True) if u == v]
    assert loops, "the self-loop was removed"
    for u, d in loops:
        assert abs(d["length"] - true_length(g, u, u, d["path"])) < 1e-6, (
            d["length"],
            true_length(g, u, u, d["path"]),
        )


def test_collapse_internal_bridges_does_not_raise():
    """Issue 3: ValueError "Inconsistent state for edge" in collapse_internal_bridges.

    The candidate bridges are listed once per pass and not re-checked after each merge. Here
    (all edges are short, radii 3): merging 1 into 0 moves the two parallel edges 1-2 and the
    edge 1-3 to node 0, so 0-2 becomes a triple edge. Then the candidate (0, 2), listed when it
    was a single edge, is collapsed: all three 0-2 edges are dropped as "internal", node 0 is
    left with degree 2, and the next candidate (0, 3) raises. Found by random testing (1 of
    20000 random graphs); it happened on one real stack (Het(2), smoothing sigma 0.75).
    Besides the crash, dropping all the edges between u and v discards the parallel edges
    (real loops) that were transferred from the merged node.
    """
    g = nx.MultiGraph()
    g.graph["is_2d"] = False
    centers = {
        0: (10, 11, 10),
        1: (12, 12, 11),
        2: (12, 10, 10),
        3: (9, 12, 9),
        4: (3, 18, 15),
        5: (3, 18, 15),
    }
    for n, c in centers.items():
        g.add_node(n, center=np.array(c))
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 2), (1, 3), (3, 4), (3, 5)]:
        g.add_edge(u, v, path=np.zeros((0, 3), np.int32))
    calculate_edge_lengths(g)
    collapse_internal_bridges(g, np.full((20, 20, 20), 3.0), collapse_length_ratio_threshold=1.0)


def test_calculate_edge_lengths_node_order():
    """Issue 4 (latent): calculate_edge_lengths calls compute_arc_length(path, u, v) in the order
    networkx yields the edge, which follows the insertion order of the nodes, not their index.
    Graphs made by graph_from_skeleton have nodes in increasing order, so this does not show up
    there, but any graph whose nodes were inserted in another order gets wrong lengths.
    """
    g = nx.MultiGraph()
    g.add_node(1, center=np.array([0, 0, 8]))  # inserted first
    g.add_node(0, center=np.array([0, 0, 0]))
    g.add_edge(1, 0, path=line(1, 7))  # stored from 0 (x=0) to 1 (x=8)
    calculate_edge_lengths(g)
    ((_, _, d),) = g.edges(data=True)
    assert abs(d["length"] - 8.0) < 1e-6, d["length"]


if __name__ == "__main__":
    failed = 0
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            try:
                test()
                print(f"PASS {name}")
            except Exception as error:
                failed += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
    raise SystemExit(1 if failed else 0)
