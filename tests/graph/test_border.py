"""Vessels cut by the XY faces of the image (pyvane.graph.border)."""

import numpy as np
import pytest
from conftest import draw_tubes as draw

from pyvane.graph.creation import create_graph
from pyvane.skeletonization import lee


def _graph(img, method="refine"):
    return create_graph(lee.skeletonize(img, method="fixed"), img, method=method)


def _leaves(graph):
    return [n for n in graph.nodes if graph.degree(n) == 1]


@pytest.mark.parametrize("two_d", [False, True])
def test_vessel_crossing_the_border_ends_on_the_face_and_is_marked(two_d):
    img = draw((17, 40, 80), [((8, 20, -10), (8, 20, 60))], 3)     # cut by x = 0
    if two_d:
        img = img[8]
    graph = _graph(img)
    (border_tip,) = [n for n in _leaves(graph) if graph.nodes[n]["border"]]
    assert graph.nodes[border_tip]["center"][-1] == 0             # moved to the face
    (inner_tip,) = [n for n in _leaves(graph) if not graph.nodes[n]["border"]]
    assert graph.nodes[inner_tip]["center"][-1] > 50
    (edge,) = list(graph.edges(data=True))
    # the vessel spans x 0-63; the path now reaches the face
    assert edge[2]["length"] == pytest.approx(60, abs=4)
    assert 0 < edge[2]["border_fraction"] < 0.2


def _t_with_short_branch(branch_end_y):
    """A vessel along x at y = 12 with a short side branch towards y = 0."""
    segments = [((8, 12, 5), (8, 12, 75)), ((8, 12, 40), (8, branch_end_y, 40))]
    return draw((17, 40, 80), segments, 3)


@pytest.mark.parametrize("two_d", [False, True])
def test_short_branch_cut_by_the_border_is_kept(two_d):
    cut = _t_with_short_branch(-5)          # crosses y = 0
    if two_d:
        cut = cut[8]
    graph = _graph(cut)
    border_tips = [n for n in _leaves(graph) if graph.nodes[n]["border"]]
    assert len(border_tips) == 1 and graph.nodes[border_tips[0]]["center"][-2] == 0
    assert sum(d >= 3 for _, d in graph.degree()) == 1


def test_short_branch_ending_inside_is_pruned_in_2d():
    # in 3D the same branch is kept by the projection guide (it is a branch of the projected
    # label), so the contrast with the border branch is checked in 2D
    graph = _graph(_t_with_short_branch(6)[8])
    assert sum(d >= 3 for _, d in graph.degree()) == 0


def test_z_spurs_on_a_vessel_touching_the_border_are_still_pruned():
    # an axially blurred vessel (4x taller than wide) along x whose label touches y = 0, with
    # bumps on its top and bottom surfaces that make the skeleton grow spurs along z
    base = draw((17, 40, 80), [((8, 3, 5), (8, 3, 75))], 3)
    img = np.repeat(base, 4, axis=0)
    zz, yy, xx = np.indices(img.shape)
    for x in (20, 40, 60):
        for z in (19, 48):
            img |= (zz - z) ** 2 + (yy - 3) ** 2 + (xx - x) ** 2 <= 9
    graph = _graph(img)
    assert len(_leaves(graph)) == 2 and graph.number_of_edges() == 1


def test_border_fraction_of_a_vessel_along_the_border_and_of_an_inner_vessel():
    img = draw((17, 50, 80), [((8, 2, 10), (8, 2, 70)), ((8, 35, 10), (8, 35, 70))], 3)
    graph = _graph(img)
    fractions = sorted(d["border_fraction"] for *_, d in graph.edges(data=True))
    assert fractions[0] == 0.0 and fractions[1] > 0.9


def test_the_scaled_method_marks_border_leaves():
    img = np.repeat(draw((17, 40, 80), [((8, 20, -10), (8, 20, 60))], 3), 3, axis=0)
    graph = _graph(img, method="scaled")
    border_tips = [n for n in _leaves(graph) if graph.nodes[n]["border"]]
    assert len(border_tips) == 1 and graph.nodes[border_tips[0]]["center"][-1] == 0
    assert all("border_fraction" in d for *_, d in graph.edges(data=True))
