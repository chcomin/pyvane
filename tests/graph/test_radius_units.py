"""The refine_graph knobs are relative to the vessel radius: a phantom drawn at twice the size
gives the same graph with the same values."""

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


def _t_with_stub_2d(scale):
    """A T junction with a long branch and a short stub, in 2D (no projection guide)."""
    segments = [((8, 20, 5), (8, 20, 75)), ((8, 20, 40), (8, 55, 40)), ((8, 20, 20), (8, 27, 20))]
    return _tubes((17, 60, 80), segments, 3, scale)[8 * scale]


def _tube_with_hole(scale):
    img = _tubes((17, 40, 80), [((8, 20, 5), (8, 20, 75))], 4, scale)
    img[:, 19 * scale:22 * scale, 36 * scale:46 * scale] = False
    return img


def _two_close_bifurcations(scale):
    """Branches leave a vessel on opposite sides, 4 voxels apart (a Y-Y pair)."""
    segments = [((8, 35, 5), (8, 35, 85)), ((8, 35, 42), (8, 5, 20)), ((8, 35, 46), (8, 65, 20))]
    return _tubes((17, 70, 90), segments, 3, scale)


def _h_bridge(scale, gap):
    """Two parallel tubes (radius 4) whose surfaces are ``gap`` voxels apart, joined by a thin
    bridge."""
    y0, y1 = 12, 12 + 8 + gap
    img = _tubes((17, 40, 80), [((8, y0, 5), (8, y0, 75)), ((8, y1, 5), (8, y1, 75))], 4, scale)
    return img | _tubes((17, 40, 80), [((8, y0, 40), (8, y1, 40))], 1.5, scale)


def _counts(graph):
    degrees = dict(graph.degree())
    multi_pairs = {(min(u, v), max(u, v)) for u, v in graph.edges()
                   if u != v and graph.number_of_edges(u, v) > 1}
    return {"branch_points": sum(d >= 3 for d in degrees.values()),
            "leaves": sum(d == 1 for d in degrees.values()),
            "multi_pairs": len(multi_pairs),
            "components": nx.number_connected_components(graph)}


def _refine(img, **knobs):
    skeleton = lee.skeletonize(img, method="fixed")
    graph, _, _ = refine_graph(remove_clusters(graph_from_skeleton(skeleton)), img, **knobs)
    return _counts(graph)


def _expected(branch_points, leaves, multi_pairs, components):
    return {"branch_points": branch_points, "leaves": leaves, "multi_pairs": multi_pairs,
            "components": components}


# (phantom, knobs, expected counts): each knob at its default and off, and the contact rule
CASES = [
    (_t_with_stub_2d, {}, _expected(1, 3, 0, 1)),
    (_t_with_stub_2d, {"spur_reach": 0.0}, _expected(2, 4, 0, 1)),
    (_tube_with_hole, {}, _expected(0, 2, 0, 1)),
    (_tube_with_hole, {"tunnel_width": 0.0}, _expected(2, 2, 1, 1)),
    (_two_close_bifurcations, {}, _expected(1, 4, 0, 1)),
    (_two_close_bifurcations, {"junction_reach": 0.0}, _expected(2, 4, 0, 1)),
    (lambda scale: _h_bridge(scale, gap=2), {}, _expected(0, 4, 0, 2)),     # removed
    (lambda scale: _h_bridge(scale, gap=12), {}, _expected(2, 4, 0, 1)),    # too long: kept
]


@pytest.mark.parametrize("phantom, knobs, expected", CASES)
@pytest.mark.parametrize("scale", [1, 2])
def test_same_graph_at_twice_the_size(phantom, knobs, expected, scale):
    assert _refine(phantom(scale), **knobs) == expected


def test_vessel_radius_map_is_in_plane_in_3d_and_the_edt_in_2d():
    tall = np.zeros((21, 21, 21), bool)
    tall[2:19, 8:13, 8:13] = True               # 17 voxels tall, 5 wide
    radius_map = vessel_radius_map(tall)
    assert vessel_radius(radius_map, (10, 10, 10)) == pytest.approx(3.0)
    flat = vessel_radius_map(tall[10])
    assert flat.shape == (21, 21)
    assert vessel_radius(flat, (10, 10)) == pytest.approx(3.0)
    assert vessel_radius(radius_map, (0, 0, 0)) == 1.0      # at least one voxel
