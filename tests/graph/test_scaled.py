"""Tests of the scaled graph creation on drawn tubes."""

import numpy as np
import pytest
from conftest import draw_tubes as draw

from pyvane.graph import scaled
from pyvane.graph.validate import check_graph
from pyvane.skeletonization import lee


def counts(g):
    bp = sum(1 for n in g if g.degree(n) >= 3 and not g.nodes[n].get("crossing"))
    flagged = sum(1 for n in g if g.nodes[n].get("crossing"))
    cyc = g.number_of_edges() - g.number_of_nodes() + 1
    return bp, flagged, cyc


def tall(img, factor):
    """The same label ``factor`` times taller (the axial blur of widefield stacks)."""
    return np.repeat(img, factor, axis=0)


T_SEGMENTS = [((8, 20, 5), (8, 20, 75)), ((8, 20, 40), (8, 55, 40))]


def test_inplane_edt_is_the_per_slice_edt():
    img = draw((10, 30, 30), [((5, 15, 2), (5, 15, 28))], 4)
    out = scaled.inplane_edt(img)
    from scipy.ndimage import distance_transform_edt
    ref = distance_transform_edt(img[5])
    np.testing.assert_allclose(out[5], ref)


def test_elongation_of_round_and_tall_labels():
    img = draw((17, 40, 80), T_SEGMENTS, 3)
    assert scaled.prepare_scaled(img).elongation < 1.5
    assert scaled.prepare_scaled(tall(img, 3)).elongation > 2.0


def test_t_junction_in_a_tall_label_has_one_branch_point():
    img = tall(draw((17, 40, 80), T_SEGMENTS, 3), 3)
    g = scaled.create_scaled_graph(img)
    assert counts(g)[:2] == (1, 0)
    assert check_graph(g, img) == []


def test_connected_crossing_is_one_flagged_node():
    img = draw((17, 70, 70), [((8, 35, 3), (8, 35, 67)), ((8, 3, 35), (8, 67, 35))], 3)
    g = scaled.create_scaled_graph(img)
    assert counts(g)[:2] == (0, 1)
    n = next(n for n in g if g.nodes[n].get("crossing"))
    assert len(g.nodes[n]["crossing_pairs"]) == 2


def labelled_hole():
    """One vessel whose label has a short hole through it: two parallel paths (a cycle) in
    the skeleton."""
    img = draw((17, 30, 75), [((8, 15, 5), (8, 15, 70))], 4)
    img[:, 14:17, 30:34] = False
    return img


def test_real_loop_survives_and_parallel_paths_inside_one_label_merge():
    # two vessels 20 voxels apart joined at both ends: a real loop
    segs = [((8, 10, 5), (8, 10, 70)), ((8, 10, 20), (8, 30, 25)), ((8, 30, 25), (8, 30, 50)),
            ((8, 30, 50), (8, 10, 55))]
    img = draw((17, 40, 75), segs, 2.5)
    assert counts(scaled.create_scaled_graph(img, tunnel_width=3.0))[2] == 1
    # the same vessel with a hole in its label: one path
    img = labelled_hole()
    assert counts(scaled.prepare_scaled(img).graph)[2] == 1
    g = scaled.create_scaled_graph(img)
    assert counts(g)[2] == 0


def test_prepared_graph_is_not_modified():
    img = draw((17, 40, 80), T_SEGMENTS, 3)
    prep = scaled.prepare_scaled(img)
    n = (prep.graph.number_of_nodes(), prep.graph.number_of_edges())
    for s in (0.5, 3.0):
        scaled.scaled_graph_from(prep, spur_reach=s)
    assert (prep.graph.number_of_nodes(), prep.graph.number_of_edges()) == n


def test_choose_spur_reach_and_diagnostics():
    img = tall(draw((17, 40, 80), T_SEGMENTS, 3), 3)
    prep = scaled.prepare_scaled(img)
    s, g = scaled.choose_spur_reach(prep)
    assert s in (1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
    _, st = scaled.scaled_graph_from(prep, spur_reach="z_excess")
    assert st["spur_reach"] == s
    _, st = scaled.scaled_graph_from(prep)
    assert st["spur_reach"] == pytest.approx(scaled.default_spur_reach(prep.elongation))
    assert scaled.default_spur_reach(0.7) == 1.25 and scaled.default_spur_reach(5.0) == 2.05
    d = scaled.graph_diagnostics(g, prep, border=0)
    assert d["bp_density"] > 0 and d["multi_density"] == 0


def test_removed_paths_are_recorded():
    img = labelled_hole()
    prep = scaled.prepare_scaled(img)
    removed = []
    g, stats = scaled.scaled_graph_from(prep, removed_paths=removed)
    assert stats["spurs_pruned"] + stats["spurs"] + stats["tunnel_paths"] > 0
    assert removed and all(len(path) and img[tuple(path.T)].all() for path in removed)
    # the same graph as without recording
    g2, _ = scaled.scaled_graph_from(prep)
    assert (g.number_of_nodes(), g.number_of_edges()) == (g2.number_of_nodes(),
                                                          g2.number_of_edges())


def test_given_skeleton_is_used_and_2d_is_refused():
    img = draw((17, 40, 80), T_SEGMENTS, 3)
    skel = lee.skeletonize(img, method="fixed")
    given = scaled.prepare_scaled(img, skel_img=skel).graph
    computed = scaled.prepare_scaled(img).graph
    assert given.number_of_edges() == computed.number_of_edges()
    with pytest.raises(ValueError):
        scaled.prepare_scaled(img[8])
