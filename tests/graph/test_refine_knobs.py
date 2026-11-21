"""Tests of the refine_graph knobs (leaves, split junctions, tunnels) and of to_simple_graph."""

import networkx as nx
import numpy as np
import pytest
from conftest import draw_tubes as draw

from pyvane.graph.creation import graph_from_skeleton, remove_clusters
from pyvane.graph.refinement import refine_graph, to_simple_graph
from pyvane.graph.validate import check_graph
from pyvane.skeletonization import lee


def _refine(img, **knobs):
    skeleton = lee.skeletonize(img, method="fixed")
    condensed = remove_clusters(graph_from_skeleton(skeleton))
    graph, _, _ = refine_graph(condensed, img, **knobs)
    return graph


def _counts(graph):
    degrees = dict(graph.degree())
    multi_pairs = {(min(u, v), max(u, v)) for u, v in graph.edges()
                   if u != v and graph.number_of_edges(u, v) > 1}
    return {"branch_points": sum(d >= 3 for d in degrees.values()),
            "leaves": sum(d == 1 for d in degrees.values()),
            "multi_pairs": len(multi_pairs)}


def _tube_with_hole(hole_width):
    img = draw((17, 40, 80), [((8, 20, 5), (8, 20, 75))], 4)
    start = 20 - hole_width // 2
    img[:, start:start + hole_width, 36:46] = False     # a hole through the tube, along z
    return img


@pytest.mark.parametrize("hole_width, two_d", [(3, False), (5, False), (3, True)])
def test_tunnel_around_a_hole_is_merged(hole_width, two_d):
    img = _tube_with_hole(hole_width)
    if two_d:
        img = img[8]
    assert _counts(_refine(img, tunnel_width=0.0))["multi_pairs"] == 1
    merged = _refine(img)
    assert _counts(merged) == {"branch_points": 0, "leaves": 2, "multi_pairs": 0}
    assert check_graph(merged, img) == []


def test_real_loop_is_kept_and_merged_only_with_a_wide_tunnel_width():
    # two vessels 20 voxels apart (4 label widths) join the same two junctions
    segments = [((8, 25, 5), (8, 25, 25)), ((8, 25, 65), (8, 25, 85)),
                ((8, 25, 25), (8, 15, 45)), ((8, 15, 45), (8, 25, 65)),
                ((8, 25, 25), (8, 35, 45)), ((8, 35, 45), (8, 25, 65))]
    img = draw((17, 50, 90), segments, 2)
    assert _counts(_refine(img))["multi_pairs"] == 1
    assert _counts(_refine(img, tunnel_width=6.0))["multi_pairs"] == 0


def _tall_t_with_bumps(scale=1):
    """A T junction 4x taller than wide (axially blurred), with bumps on its top and bottom
    surfaces that make the skeleton grow spurs along z. All sizes are multiplied by
    ``scale``."""
    base = draw((17, 60, 80), [((8, 20, 5), (8, 20, 75)), ((8, 20, 40), (8, 55, 40))], 3)
    img = np.repeat(base, 4, axis=0)            # the label spans z 20-47
    zz, yy, xx = np.indices(img.shape)
    for x in (15, 28, 55, 65):
        for z in (19, 48):
            img |= (zz - z) ** 2 + (yy - 20) ** 2 + (xx - x) ** 2 <= 9
    for axis in range(3):
        img = np.repeat(img, scale, axis=axis)
    return img


@pytest.mark.parametrize("scale", [1, 2])
def test_z_spurs_are_removed_and_the_real_branch_is_kept(scale):
    img = _tall_t_with_bumps(scale)
    assert _counts(_refine(img, spur_reach=0.0))["leaves"] > 3
    pruned = _refine(img)
    assert _counts(pruned)["branch_points"] == 1 and _counts(pruned)["leaves"] == 3
    assert check_graph(pruned, img) == []


def test_callback_reports_the_steps_of_the_knobs():
    img = _tube_with_hole(3)
    steps = []
    _refine(img, step_callback=lambda step, *rest: steps.append(step))
    assert steps[:5] == ["degree_two", "false_anastomoses", "leaves", "collapse", "tunnels"]
    steps.clear()
    _refine(img, spur_reach=0.0, step_callback=lambda step, *rest: steps.append(step))
    assert "leaves" not in steps


def test_to_simple_graph_keeps_the_longest_of_parallel_edges_and_no_loops():
    img = _tube_with_hole(3)
    graph = _refine(img, tunnel_width=0.0)
    lengths = sorted(d["length"] for u, v, d in graph.edges(data=True) if u != v
                     and graph.number_of_edges(u, v) > 1)
    to_simple_graph(graph)
    assert not any(graph.number_of_edges(u, v) > 1 for u, v in graph.edges())
    assert nx.number_of_selfloops(graph) == 0
    # the longest parallel edge survives, merged with its neighbors into one edge
    assert max(d["length"] for *_, d in graph.edges(data=True)) > lengths[-1]
