"""Tests of the "voxel" and "smoothed" edge measurements and of the geometry switch of the
graph metrics."""

import numpy as np
import pytest
from conftest import draw_tubes as draw

from pyvane.graph.creation import create_graph
from pyvane.metrics import metrics, smoothed
from pyvane.skeletonization import lee

SHARED_ATTRIBUTES = {"mean_radius", "min_radius", "max_radius", "radii", "points",
                     "measured_length", "tortuosity"}
T_SEGMENTS = [((8, 20, 5), (8, 20, 75)), ((8, 20, 40), (8, 55, 40))]


def graph_of(img, method="refine"):
    return create_graph(lee.skeletonize(img, method="fixed"), img, method=method)


def test_smooth_path_keeps_ends_and_straight_lines():
    x = np.arange(41, dtype=float)
    stair = np.c_[np.zeros(41), np.floor(x / 2) + (x % 2), x]
    out = smoothed.smooth_path(stair, 2.0)
    np.testing.assert_allclose(out[[0, -1]], stair[[0, -1]])
    new = np.linalg.norm(np.diff(out, axis=0), axis=1).sum()
    assert new == pytest.approx(np.linalg.norm(stair[-1] - stair[0]), rel=0.03)
    line = np.c_[np.zeros(10), np.zeros(10), np.arange(10.0)]
    np.testing.assert_allclose(smoothed.smooth_path(line, 2.0), line, atol=1e-9)


@pytest.mark.parametrize("graph_method", ["refine", "scaled"])
@pytest.mark.parametrize("method", ["voxel", "smoothed"])
def test_both_methods_write_the_same_attributes(graph_method, method):
    img = draw((17, 60, 80), T_SEGMENTS, 3)
    graph = graph_of(img, graph_method)
    metrics.measure_edges(graph, img, method=method, pix_size=(2.0, 2.0, 2.0))
    assert graph.graph["geometry"] == method
    for *_, data in graph.edges(data=True):
        assert SHARED_ATTRIBUTES <= set(data)
        assert len(data["radii"]) == len(data["path"])
        assert np.asarray(data["points"]).shape[1] == 3


@pytest.mark.parametrize("seg", [((8, 20, 5), (8, 20, 75)),       # along x
                                 ((2, 20, 10), (14, 20, 70)),      # tilted in z
                                 ((8, 5, 5), (8, 35, 75))])        # diagonal in plane
def test_smoothed_tube_length_radius_and_tortuosity(seg):
    img = draw((17, 40, 80), [seg], 3)
    graph = graph_of(img, "scaled")
    metrics.measure_edges(graph, img, method="smoothed")
    (_, _, data), = list(graph.edges(data=True))
    true_length = np.linalg.norm(np.subtract(seg[1], seg[0]))
    # the label of a tube with round caps is 2 radii longer than its axis
    assert data["measured_length"] == pytest.approx(true_length + 6, abs=2.0)
    # the distance to the nearest background voxel center is within a voxel of r
    assert 3.0 - 0.5 <= data["mean_radius"] <= 3.0 + 1.0
    assert data["min_radius"] <= data["mean_radius"] <= data["max_radius"]
    # raw voxel paths give up to 1.23; the thinned skeleton of an oblique tube also wobbles
    # at longer wavelengths than the smoothing removes
    assert data["tortuosity"] == pytest.approx(1.0, abs=0.07)


def test_pix_size_scales_lengths_and_radii():
    img = draw((17, 40, 80), [((8, 20, 5), (8, 20, 75))], 3)
    for method in ("voxel", "smoothed"):
        values = []
        for size in (1.0, 2.0):
            graph = graph_of(img)
            metrics.measure_edges(graph, img, method=method, pix_size=(size,) * 3)
            (_, _, data), = list(graph.edges(data=True))
            values.append((data["measured_length"], data["mean_radius"]))
        np.testing.assert_allclose(values[1], 2 * np.asarray(values[0]), rtol=1e-6)


def test_density_and_tortuosity_follow_the_recorded_geometry():
    img = draw((17, 60, 80), T_SEGMENTS, 3)
    graph = graph_of(img)
    voxel_density = metrics.vessel_density(graph, img.shape)
    with pytest.raises(ValueError):     # no smoothed points yet
        metrics.vessel_density(graph, img.shape, geometry="smoothed")
    metrics.measure_edges(graph, img, method="smoothed")
    smoothed_density = metrics.vessel_density(graph, img.shape)
    assert smoothed_density != voxel_density
    assert metrics.vessel_density(graph, img.shape, geometry="voxel") == voxel_density
    roi = np.zeros(img.shape, np.uint8)
    roi[:, :, :40] = 1
    assert 0 < metrics.vessel_density(graph, img.shape, img_roi=roi) * roi.sum() < (
        smoothed_density * roi.size)


def test_smoothing_lowers_the_scale_tortuosity_of_a_straight_diagonal_tube():
    # the voxel path of a diagonal tube is a staircase
    img = draw((17, 60, 80), [((8, 5, 5), (8, 35, 75))], 3)
    graph = graph_of(img)
    voxel_value = metrics.tortuosity(graph, scale=5)
    metrics.measure_edges(graph, img, method="smoothed")
    assert metrics.tortuosity(graph, scale=5, geometry="voxel") == voxel_value
    assert 0 <= metrics.tortuosity(graph, scale=5) < 0.5 * voxel_value


def test_flagged_crossings_are_not_branch_points():
    img = draw((17, 70, 70), [((8, 35, 3), (8, 35, 67)), ((8, 3, 35), (8, 67, 35))], 3)
    graph = graph_of(img, "scaled")
    assert any(data.get("crossing") for _, data in graph.nodes(data=True))
    assert metrics.num_branch_points(graph) == 0
