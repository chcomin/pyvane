"""The refine_graph thresholds are relative to the vessel radius: a phantom drawn at twice the
size gives the same graph with the same parameters."""

import networkx as nx
import numpy as np
import pytest
from conftest import draw_tubes

from pyvane.graph.creation import graph_from_skeleton, remove_clusters
from pyvane.graph.label_geometry import vessel_radius, vessel_radius_map
from pyvane.graph.refinement import refine_graph
from pyvane.skeletonization import lee


def _scaled(points, scale):
    return tuple(scale * c for c in points)


def _tubes(shape, segments, radius, scale):
    return draw_tubes(_scaled(shape, scale),
                      [(_scaled(a, scale), _scaled(b, scale)) for a, b in segments],
                      radius * scale)


def _t_with_stub(scale):
    """A T junction with a long branch and a short stub."""
    segments = [((8, 20, 5), (8, 20, 75)), ((8, 20, 40), (8, 55, 40)), ((8, 20, 20), (8, 27, 20))]
    return _tubes((17, 60, 80), segments, 3, scale)


def _tube_with_hole(scale):
    img = _tubes((17, 40, 80), [((8, 20, 5), (8, 20, 75))], 4, scale)
    img[:, 19 * scale:22 * scale, 36 * scale:46 * scale] = False
    return img


def _tube_and_debris(scale):
    img = _tubes((17, 40, 80), [((8, 10, 5), (8, 10, 75))], 3, scale)
    return img | _tubes((17, 40, 80), [((8, 30, 40), (8, 30, 44))], 2, scale)


def _h_bridge(scale):
    """Two parallel tubes joined by a thin bridge."""
    img = _tubes((17, 40, 80), [((8, 12, 5), (8, 12, 75)), ((8, 28, 5), (8, 28, 75))], 4, scale)
    return img | _tubes((17, 40, 80), [((8, 12, 40), (8, 28, 40))], 1.5, scale)


def _counts(graph):
    degrees = dict(graph.degree())
    multi_pairs = {(min(u, v), max(u, v)) for u, v in graph.edges()
                   if u != v and graph.number_of_edges(u, v) > 1}
    return {"branch_points": sum(d >= 3 for d in degrees.values()),
            "leaves": sum(d == 1 for d in degrees.values()),
            "multi_pairs": len(multi_pairs),
            "components": nx.number_connected_components(graph)}


def _refine(img, **params):
    skeleton = lee.skeletonize(img, method="fixed")
    graph, _, _ = refine_graph(remove_clusters(graph_from_skeleton(skeleton)), img, **params)
    return _counts(graph)


# (phantom, parameters, expected counts): each rule on and off
CASES = [
    (_t_with_stub, {}, {"branch_points": 1, "leaves": 3, "multi_pairs": 0, "components": 1}),
    (_t_with_stub, {"bulge_ratio_threshold": 0.01},
     {"branch_points": 2, "leaves": 4, "multi_pairs": 0, "components": 1}),
    (_tube_with_hole, {"multi_edge_threshold": 20.0},
     {"branch_points": 0, "leaves": 2, "multi_pairs": 0, "components": 1}),
    (_tube_with_hole, {"multi_edge_threshold": 0.0},
     {"branch_points": 2, "leaves": 2, "multi_pairs": 1, "components": 1}),
    (_tube_and_debris, {"comp_length_threshold": 5.0},
     {"branch_points": 0, "leaves": 2, "multi_pairs": 0, "components": 1}),
    (_tube_and_debris, {"comp_length_threshold": 0.0},
     {"branch_points": 0, "leaves": 4, "multi_pairs": 0, "components": 2}),
    (_h_bridge, {"bridge_length_ratio_threshold": 3.0},
     {"branch_points": 0, "leaves": 4, "multi_pairs": 0, "components": 2}),
    (_h_bridge, {"bridge_length_ratio_threshold": 0.0},
     {"branch_points": 2, "leaves": 4, "multi_pairs": 0, "components": 1}),
]


@pytest.mark.parametrize("phantom, params, expected", CASES)
@pytest.mark.parametrize("scale", [1, 2])
def test_same_graph_at_twice_the_size(phantom, params, expected, scale):
    assert _refine(phantom(scale), **params) == expected


def test_vessel_radius_map_is_in_plane_in_3d_and_the_edt_in_2d():
    tall = np.zeros((21, 21, 21), bool)
    tall[2:19, 8:13, 8:13] = True               # 17 voxels tall, 5 wide
    radius_map = vessel_radius_map(tall)
    assert vessel_radius(radius_map, (10, 10, 10)) == pytest.approx(3.0)
    flat = vessel_radius_map(tall[10])
    assert flat.shape == (21, 21)
    assert vessel_radius(flat, (10, 10)) == pytest.approx(3.0)
    assert vessel_radius(radius_map, (0, 0, 0)) == 1.0      # at least one voxel
