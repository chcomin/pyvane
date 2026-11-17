# Graph refinement issues

Problems found in `pyvane/graph` at commit `37b5db0`. Each one has a minimal reproduction in
`graph/test_refinement_issues.py`; the tests fail on `37b5db0` and pass once the issue is fixed.

Conventions: the `path` of an edge excludes the node centers and runs from the node of smaller
index to the node of larger index. `length` is the arc length from one center, along the path,
to the other center.

## 1. Edge lengths too long after merging nodes

`compute_arc_length(path, u, v, graph)` adds the gap from `center[u]` to `path[0]` and from
`path[-1]` to `center[v]`, so it is only correct when `u < v`. `remove_degree_two_nodes` and
`collapse_internal_bridges` oriented the new path from min to max node but then called it with
the nodes in their original order. When `u > v`, each gap was measured to the wrong end, adding
about two chords to the length. Every length-based step of `refine_graph` that runs after a
merge used these wrong lengths, so the topology was affected too.

Tests: `test_remove_degree_two_nodes_length`,
`test_collapse_internal_bridges_transferred_edge_length`.

## 2. `collapse_internal_bridges` did not update self-loops of the merged node

When a bridge `u`-`v` is collapsed, `u` moves to the midpoint of `u` and `v`. Self-loops of
`u` kept their old path and length, so the path no longer touched the new center and the
stored length no longer matched it.

Test: `test_collapse_internal_bridges_self_loop_length`.

## 3. `collapse_internal_bridges` raised "Inconsistent state for edge"

Candidate bridges were listed once per pass and not re-checked after each merge. An earlier
merge could turn a single-edge candidate into a multi-edge; collapsing it then dropped all the
parallel edges, left a node with degree 2, and the next candidate raised `ValueError`.

Test: `test_collapse_internal_bridges_does_not_raise`.

## 4. `calculate_edge_lengths` depended on node insertion order (latent)

`calculate_edge_lengths` passed `(u, v)` in the order networkx yields the edge, which follows
node insertion order, not node index. Graphs from `graph_from_skeleton` insert nodes in
increasing order, so this did not show up there, but other graphs got wrong lengths.

Test: `test_calculate_edge_lengths_node_order`.

## Fix

`compute_arc_length` now orders `u` and `v` itself (issues 1 and 4).
`collapse_internal_bridges` re-checks each candidate on the current graph right before
collapsing it (issue 3) and re-joins the self-loops of `u` to the new center (issue 2).
