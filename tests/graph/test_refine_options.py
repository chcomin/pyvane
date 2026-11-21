"""Tests of the optional refine_graph steps: z-spur pruning, tunnel merging, crossing flags and
the X-crossing collapse."""

import numpy as np
import pytest
from conftest import draw_tubes as draw

from pyvane.graph.creation import create_graph, graph_from_skeleton, remove_clusters
from pyvane.graph.refinement import refine_graph
from pyvane.graph.validate import check_graph
from pyvane.skeletonization import lee

def _refine(img, **options):
    skeleton = lee.skeletonize(img, method="fixed")
    condensed = remove_clusters(graph_from_skeleton(skeleton))
    graph, edt, _ = refine_graph(condensed, img, **options)
    return graph, edt


def _counts(graph):
    degrees = dict(graph.degree())
    multi_pairs = {(min(u, v), max(u, v)) for u, v in graph.edges()
                   if u != v and graph.number_of_edges(u, v) > 1}
    return {"branch_points": sum(d >= 3 for d in degrees.values()),
            "leaves": sum(d == 1 for d in degrees.values()),
            "multi_pairs": len(multi_pairs),
            "crossings": sum(1 for n in graph if graph.nodes[n].get("crossing"))}


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
    kept, _ = _refine(img, multi_edge_threshold=0.0)
    assert _counts(kept)["multi_pairs"] == 1
    merged, _ = _refine(img, multi_edge_threshold=0.0, tunnel_width=2.0)
    assert _counts(merged) == {"branch_points": 0, "leaves": 2, "multi_pairs": 0,
                               "crossings": 0}
    assert check_graph(merged, img) == []


def test_real_loop_is_kept():
    # two vessels 14 voxels apart join the same two junctions
    segments = [((8, 25, 5), (8, 25, 25)), ((8, 25, 65), (8, 25, 85)),
                ((8, 25, 25), (8, 18, 45)), ((8, 18, 45), (8, 25, 65)),
                ((8, 25, 25), (8, 32, 45)), ((8, 32, 45), (8, 25, 65))]
    img = draw((17, 50, 90), segments, 2)
    graph, _ = _refine(img, multi_edge_threshold=0.0, tunnel_width=2.0)
    assert _counts(graph)["multi_pairs"] == 1


def test_x_crossing_is_flagged_and_t_junction_is_not():
    x_img = draw((17, 70, 70), [((8, 35, 5), (8, 35, 65)), ((8, 5, 35), (8, 65, 35))], 3)
    graph, _ = _refine(x_img, crossing_angle=150.0)
    (node,) = [n for n in graph if graph.degree(n) == 4]
    assert graph.nodes[node]["crossing"]
    pairs = graph.nodes[node]["crossing_pairs"]
    assert len(pairs) == 2 and len({n for pair in pairs for n in pair}) == 4
    # each pair continues one of the two straight vessels
    for a, b in pairs:
        ca, cb = (np.asarray(graph.nodes[n]["center"]) for n in (a, b))
        assert ca[1] == cb[1] or ca[2] == cb[2]
    t_img = draw((17, 60, 80), [((8, 20, 5), (8, 20, 75)), ((8, 20, 40), (8, 55, 40))], 3)
    graph, _ = _refine(t_img, crossing_angle=150.0)
    assert _counts(graph)["crossings"] == 0


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
def test_z_spur_pruning_removes_the_spurs_and_keeps_the_branch(scale):
    # the distances are relative to the vessel radius: the same parameters work at any scale
    img = _tall_t_with_bumps(scale)
    default, _ = _refine(img)
    assert _counts(default)["leaves"] > 3
    pruned, _ = _refine(img, z_spur_params={})
    assert _counts(pruned)["branch_points"] == 1 and _counts(pruned)["leaves"] == 3
    assert check_graph(pruned, img) == []


def test_new_steps_are_reported_to_the_callback_and_off_by_default():
    img = _tube_with_hole(3)
    steps = []
    _refine(img, tunnel_width=2.0, z_spur_params={},
            step_callback=lambda step, *rest: steps.append(step))
    assert "z_spurs" in steps and "tunnels" in steps
    assert steps.index("z_spurs") < steps.index("collapse") < steps.index("tunnels")
    steps.clear()
    _refine(img, step_callback=lambda step, *rest: steps.append(step))
    assert "z_spurs" not in steps and "tunnels" not in steps


def test_create_graph_passes_refine_params():
    img = _tube_with_hole(3)
    skeleton = lee.skeletonize(img, method="fixed")
    default = create_graph(skeleton, img, refine_params={"multi_edge_threshold": 0.0})
    merged = create_graph(skeleton, img, refine_params={"multi_edge_threshold": 0.0,
                                                        "tunnel_width": 2.0})
    assert _counts(default)["multi_pairs"] == 1 and _counts(merged)["multi_pairs"] == 0
    with pytest.raises(ValueError):
        create_graph(skeleton, img, method="scaled", refine_params={"tunnel_width": 2.0})


def test_z_spur_pruning_is_skipped_in_2d():
    img = np.zeros((40, 40), bool)
    img[18:22, 5:35] = True
    skeleton = lee.skeletonize(img)
    condensed = remove_clusters(graph_from_skeleton(skeleton))
    steps = []
    refine_graph(condensed, img, z_spur_params={},
                 step_callback=lambda step, *rest: steps.append(step))
    assert "z_spurs" not in steps


def test_crossing_at_different_depths_is_flagged():
    # two vessels 4 voxels apart in z whose labels merge into one degree-4 node
    img = draw((34, 80, 80), [((12, 40, 5), (12, 40, 75)), ((16, 5, 40), (16, 75, 40))], 3)
    graph, _ = _refine(img, crossing_angle=150.0)
    assert _counts(graph)["crossings"] == 1


def test_split_crossing_is_collapsed_with_the_crossing_ratio():
    # two straight vessels 10 voxels apart in z, joined by a vertical bridge: >-< seen from
    # the side, an X from above
    segments = [((10, 35, 5), (10, 35, 65)), ((20, 5, 35), (20, 65, 35)),
                ((10, 35, 35), (20, 35, 35))]
    img = draw((40, 70, 70), segments, 3)
    split, _ = _refine(img, crossing_angle=150.0)
    assert _counts(split)["branch_points"] == 2 and _counts(split)["crossings"] == 0
    merged, _ = _refine(img, crossing_angle=150.0, crossing_collapse_ratio=3.0)
    assert _counts(merged)["branch_points"] == 1 and _counts(merged)["crossings"] == 1
    # a T at the same distance is not an X and stays split
    t_segments = [((10, 35, 5), (10, 35, 65)), ((20, 35, 35), (20, 65, 35)),
                  ((10, 35, 35), (20, 35, 35))]
    t_img = draw((40, 70, 70), t_segments, 3)
    t_graph, _ = _refine(t_img, crossing_angle=150.0, crossing_collapse_ratio=3.0)
    assert _counts(t_graph)["crossings"] == 0
