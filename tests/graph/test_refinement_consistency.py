"""Randomized checks that the refinement functions keep the edge conventions of pyvane.

Random multigraphs are built with consistent paths (Bresenham lines between node centers,
stored from the smaller to the larger node id). Each refinement function must run without
raising and leave a graph that `find_inconsistent_edges` accepts.
"""

import networkx as nx
import numpy as np
import pytest

from pyvane.graph.guided import prune_leaves
from pyvane.graph.refinement import (
    collapse_internal_bridges,
    merge_tunnels,
    refine_graph,
    remove_degree_two_nodes,
    remove_false_anastomoses,
    to_simple_graph,
)
from pyvane.graph.util import bresenham_3d, calculate_edge_lengths, find_inconsistent_edges


def _line(c0, c1):
    return bresenham_3d(np.asarray(c0, np.int32), np.asarray(c1, np.int32))[1:-1]


def _detour_path(c0, c1, rng, size):
    """Path from c0 to c1 (excluding both) through a random intermediate voxel."""
    mid = (np.asarray(c0) + np.asarray(c1)) // 2 + rng.integers(-3, 4, 3)
    mid = np.clip(mid, 0, size - 1)
    if np.array_equal(mid, c0) or np.array_equal(mid, c1):
        return _line(c0, c1)
    return np.vstack([_line(c0, mid), mid[None], _line(mid, c1)]).astype(np.int32)


def _loop_path(c, rng, size):
    """Self-loop path that leaves the center and comes back to it."""
    a = np.clip(np.asarray(c) + rng.integers(-4, 5, 3), 0, size - 1)
    b = np.clip(np.asarray(c) + rng.integers(-4, 5, 3), 0, size - 1)
    if np.abs(a - c).max() < 2 or np.abs(b - c).max() < 2 or np.abs(a - b).max() < 2:
        return None
    return np.vstack(
        [_line(c, a), a[None], _line(a, b), b[None], _line(b, c)]).astype(np.int32)


def random_graph(seed, n_nodes=12, n_edges=18, size=24):
    rng = np.random.default_rng(seed)
    g = nx.MultiGraph()
    g.graph["is_2d"] = False
    centers = {}
    # Insert nodes in random order to exercise node-order independence.
    for n in rng.permutation(n_nodes):
        c = rng.integers(0, size, 3).astype(np.int32)
        centers[int(n)] = c
        g.add_node(int(n), center=c)
    for _ in range(n_edges):
        u, v = (int(x) for x in rng.integers(0, n_nodes, 2))
        if u == v:
            path = _loop_path(centers[u], rng, size)
            if path is None:
                continue
        else:
            a, b = min(u, v), max(u, v)
            if rng.random() < 0.5:
                path = _line(centers[a], centers[b])
            else:
                path = _detour_path(centers[a], centers[b], rng, size)
        g.add_edge(u, v, path=path.reshape(-1, 3).astype(np.int32))
    calculate_edge_lengths(g)
    return g, np.full((size, size, size), float(rng.uniform(1, 6)))


SEEDS = range(300)


@pytest.mark.parametrize("seed", SEEDS)
def test_random_graph_is_consistent(seed):
    g, _ = random_graph(seed)
    assert not find_inconsistent_edges(g)


@pytest.mark.parametrize("seed", SEEDS)
def test_collapse_internal_bridges_random(seed):
    g, edt = random_graph(seed)
    remove_degree_two_nodes(g)
    collapse_internal_bridges(g, edt, collapse_length_ratio_threshold=1.0)
    assert not find_inconsistent_edges(g)


@pytest.mark.parametrize("seed", SEEDS)
def test_other_heuristics_random(seed):
    g, edt = random_graph(seed)
    remove_degree_two_nodes(g)
    assert not find_inconsistent_edges(g)
    # the distance transform also serves as the radius map here
    remove_false_anastomoses(g, edt, edt)
    prune_leaves(g, edt, edt)
    merge_tunnels(g, edt, edt, elongation=1.0)
    assert not find_inconsistent_edges(g)
    to_simple_graph(g)
    assert not find_inconsistent_edges(g)


@pytest.mark.parametrize("seed", range(100))
def test_refine_graph_random(seed):
    g, edt = random_graph(seed)
    mask = edt > 0  # only used for the distance transform
    refined, _, _ = refine_graph(g, mask)
    assert not find_inconsistent_edges(refined)


def test_collapse_transfers_parallel_edges_of_v():
    """Parallel edges between v and a third node survive the collapse as parallel edges of u."""
    g = nx.MultiGraph()
    g.graph["is_2d"] = False
    centers = {0: (5, 10, 10), 1: (5, 10, 12), 2: (5, 20, 10), 3: (5, 0, 10), 4: (5, 10, 25)}
    for n, c in centers.items():
        g.add_node(n, center=np.array(c, np.int32))
    for u, v in [(0, 1), (0, 2), (0, 3)]:
        g.add_edge(u, v, path=_line(centers[u], centers[v]).reshape(-1, 3))
    rng = np.random.default_rng(0)
    g.add_edge(1, 4, path=_line(centers[1], centers[4]).reshape(-1, 3))
    g.add_edge(1, 4, path=_detour_path(centers[1], centers[4], rng, 30))
    calculate_edge_lengths(g)
    collapse_internal_bridges(g, np.full((10, 30, 30), 5.0), collapse_length_ratio_threshold=1.0)
    assert 1 not in g
    assert g.number_of_edges(0, 4) == 2
    assert not find_inconsistent_edges(g)
