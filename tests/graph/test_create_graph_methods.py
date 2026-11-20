"""Tests of the method switch of create_graph ("refine" and "scaled")."""

import numpy as np
import pytest
from conftest import draw_tubes as draw

from pyvane.graph.creation import create_graph, create_graph_with_mapping
from pyvane.graph.scaled import create_scaled_graph
from pyvane.skeletonization import lee

T_SEGMENTS = [((8, 20, 5), (8, 20, 75)), ((8, 20, 40), (8, 55, 40))]


@pytest.fixture
def t_junction():
    img = np.repeat(draw((17, 60, 80), T_SEGMENTS, 3), 3, axis=0)    # 3x taller label
    return lee.skeletonize(img, method="fixed"), img


def test_refine_is_the_default(t_junction):
    skel, img = t_junction
    graph = create_graph(skel, img)
    assert graph.graph["method"] == "refine"
    explicit = create_graph(skel, img, method="refine")
    assert sorted(graph.edges()) == sorted(explicit.edges())


def test_scaled_gives_the_scaled_graph_with_the_create_graph_attributes(t_junction):
    skel, img = t_junction
    graph = create_graph(skel, img, method="scaled")
    reference = create_scaled_graph(img)
    assert sorted(graph.edges()) == sorted(reference.edges())
    assert graph.graph["method"] == "scaled" and graph.graph["elongation"] > 2
    assert all("radius" in data for _, data in graph.nodes(data=True))
    assert all("trim_amount" in data and "outer_length" in data
               for *_, data in graph.edges(data=True))


def test_scaled_params_are_passed(t_junction):
    skel, img = t_junction
    tight = create_graph(skel, img, method="scaled", scaled_params={"spur_reach": 0.1})
    loose = create_graph(skel, img, method="scaled", scaled_params={"spur_reach": 10.0})
    assert loose.number_of_edges() <= tight.number_of_edges()


def test_full_output_and_mapping_work_with_both_methods(t_junction):
    skel, img = t_junction
    for method in ("refine", "scaled"):
        graph, edt_bg, trash_paths = create_graph(skel, img, method=method, full_output=True)
        assert edt_bg.shape == img.shape and isinstance(trash_paths, list)
        graph, labeled, id_cl_map = create_graph_with_mapping(skel, img, method=method)
        assert labeled.shape == img.shape and id_cl_map
        # most of the foreground is assigned to the graph
        assert (labeled[img] > 0).mean() > 0.8, method


def test_invalid_combinations_raise(t_junction):
    skel, img = t_junction
    with pytest.raises(ValueError):
        create_graph(skel, img, method="other")
    with pytest.raises(ValueError):
        create_graph(skel, img, scaled_params={"spur_reach": 2.0})
    with pytest.raises(ValueError):
        create_graph(skel[8], img[8], method="scaled")
