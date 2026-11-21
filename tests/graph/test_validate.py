"""The step callback of refine_graph and the invariants of check_graph."""

from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image

from pyvane.graph.creation import graph_from_skeleton, remove_clusters
from pyvane.graph.refinement import refine_graph
from pyvane.graph.validate import check_graph
from pyvane.skeletonization import lee

STEPS = ["degree_two", "multi_edges", "self_loops", "false_anastomoses", "prune", "collapse"]


def _condensed():
    bin_img = np.array(Image.open(Path(__file__).parent / "12749_segmented.png")) > 0
    skel = lee.skeletonize(bin_img)
    return bin_img, remove_clusters(graph_from_skeleton(skel))


def test_step_callback_reports_every_step_and_the_graph_stays_valid():
    bin_img, cond = _condensed()
    calls = []

    def cb(step, graph, changed, trash):
        calls.append((step, changed, len(trash)))
        # between steps degree-2 nodes may exist (they are merged at the start of each pass)
        assert check_graph(graph, bin_img, allow_degree_two=True, min_inside=0.0) == []

    graph, _, trash = refine_graph(cond, bin_img, step_callback=cb)
    names = [c[0] for c in calls]
    assert names[:6] == STEPS and names[-1] == "components"
    n_pass = (len(names) - 1) // len(STEPS)
    assert names[:-1] == STEPS * n_pass and n_pass >= 2
    # the last pass changes nothing; every trash path was reported to the callback
    assert not any(c[1] for c in calls[-1 - len(STEPS):-1])
    assert sum(c[2] for c in calls) == len(trash)
    assert check_graph(graph, bin_img, min_inside=0.0) == []


def test_callback_does_not_change_the_result():
    bin_img, cond = _condensed()
    a, _, _ = refine_graph(cond, bin_img)
    b, _, _ = refine_graph(cond, bin_img, step_callback=lambda *args: None)

    def summary(g):
        nodes = {n: tuple(c) for n, c in g.nodes(data="center")}
        edges = sorted((min(u, v), max(u, v), round(d["length"], 6), tuple(map(tuple, d["path"])))
                       for u, v, d in g.edges(data=True))
        return nodes, edges

    assert summary(a) == summary(b)


def test_check_graph_finds_violations():
    g = nx.MultiGraph()
    g.add_node(0, center=np.array([1, 1, 1]))
    g.add_node(1, center=np.array([1, 1, 5]))
    g.add_node(2, center=np.array([1, 5, 5]))
    g.add_node(3, center=np.array([9, 9, 9]))
    path = np.array([[1, 1, 2], [1, 1, 3], [1, 1, 4]])
    g.add_edge(0, 1, path=path, length=4.0)
    g.add_edge(1, 2, path=np.array([[1, 2, 5], [1, 3, 5], [1, 4, 5]]), length=-1.0)
    img = np.zeros((10, 10, 10), bool)
    img[1, 1, 1:6] = True
    msgs = [m for _, m in check_graph(g, img)]
    assert "isolated node" in msgs
    assert "degree-2 node" in msgs
    assert any(m.startswith("invalid length") for m in msgs)
    assert any("stored length" in m for m in msgs)
    assert "node center outside the mask" in msgs
    assert any("of the path is foreground" in m for m in msgs)
    # a pure ring anchor is allowed
    r = nx.MultiGraph()
    r.add_node(0, center=np.array([0, 0, 0]))
    ring = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]])
    from pyvane.graph.util import compute_arc_length
    r.add_edge(0, 0, path=ring, length=0.0)
    r[0][0][0]["length"] = compute_arc_length(ring, 0, 0, r)
    assert check_graph(r) == []
