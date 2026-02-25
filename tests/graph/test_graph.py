import networkx as nx
import numpy as np
from PIL import Image

from pyvane.graph.creation import (
    assign_node_radii_and_outer_props,
    graph_from_skeleton,
    map_foreground_to_graph,
    remove_clusters,
)
from pyvane.graph.refinement import refine_graph
from pyvane.metrics import metrics
from pyvane.skeletonization import lee


def check_graph_artifacts(graph, return_ids=True):
    stats = {
        "is_multigraph_class": graph.is_multigraph(),
        "self_loop_count": nx.number_of_selfloops(graph),
        "multi_edge_count": graph.number_of_edges() - len(set(graph.edges())),
        "zero_degree_count": sum(1 for node in graph.nodes() if graph.degree(node) == 0)
    }

    if not return_ids:
        return stats
    
    self_loop_ids = list(nx.selfloop_edges(graph, keys=True))
    
    # 1. Identify pairs (u, v) that have more than one edge
    parallel_pairs = [(u, v) for u, v in graph.edges(keys=False) if graph.number_of_edges(u, v) > 1]

    # 2. Get the unique set of these pairs (to avoid duplicates from the edge list)
    unique_parallel_pairs = {tuple(sorted(pair)) for pair in parallel_pairs}

    # 3. Get the full (u, v, key) IDs for every parallel edge
    parallel_edge_ids = []
    for u, v in unique_parallel_pairs:
        for key in graph[u][v]:
            parallel_edge_ids.append((u, v, key))

    stats = {**stats, "parallel_edge_ids": parallel_edge_ids, "self_loop_ids": self_loop_ids}

    return stats

def test_graph():

    length_threshold = 5.0

    bin_img = np.array(Image.open("12749_segmented.png"))
    skel_img = lee.skeletonize(bin_img)

    init_graph = graph_from_skeleton(skel_img, keep_rings = False)
    cond_graph = remove_clusters(init_graph)
    simple_graph, edt_bg, trash_paths = refine_graph(
        cond_graph, 
        bin_img,
        bulge_len_threshold = length_threshold,
        bulge_size_threshold = 0.0,
        bulge_ratio_threshold = 0.0,
        elongation_threshold = 0.0,
        multi_edge_threshold = None, 
        self_loop_threshold = None, 
        bridge_radius_ratio_threshold = 0.5,
        bridge_length_threshold = 3.0,
        collapse_length_ratio_threshold = 1.0,
        comp_size_threshold = 0, 
        comp_length_threshold = 1.0, 
        )

    assign_node_radii_and_outer_props(simple_graph, edt_bg)

    labeled_volume, id_cl_map = map_foreground_to_graph(
        graph = simple_graph, 
        bin_img = bin_img, 
        edt_bg = edt_bg, 
        edges_only = True, 
        compactness = 0.1,
        trash_paths = trash_paths
        )

    metrics.assign_centerline_radii(simple_graph, labeled_volume > 0)

    stats = check_graph_artifacts(simple_graph, return_ids=False)
    
    has_degree_2 = any(d==2 for d in dict(simple_graph.degree()).values())
    assert not has_degree_2, "Graph should not have degree-2 nodes"

    assert stats["self_loop_count"] == 0, "Graph should not have self-loops"
    assert stats["multi_edge_count"] == 0, "Graph should not have multi-edges"
    assert stats["zero_degree_count"] == 0, "Graph should not have zero-degree nodes"

    for node in simple_graph.nodes():
        assert simple_graph.nodes[node]["radius"] > 0.0, f"Node {node} has non-positive radius"

    for u, v, key, data in simple_graph.edges(keys=True, data=True):
        assert len(data["path"]) > 0, f"Edge ({u}, {v}, {key}) has empty path"
        assert data["length"] > 0.0, f"Edge ({u}, {v}, {key}) has non-positive length"
        assert min(data["radii"]) > 0.0, f"Edge ({u}, {v}, {key}) has non-positive radius"

if __name__ == "__main__":
    test_graph()
    print("All tests passed!")