# %% [markdown]
# # tunnel_width: paths around holes in the label
#
# Rule (`refinement.merge_tunnels`): a hole through a vessel's label makes its skeleton split
# into two paths around the hole: two edges between the same nodes (a multi-edge pair), or a
# self-loop. For each pair of nodes joined by several edges, the most central edge is kept and
# every other edge that stays within `tunnel_width` local label widths of it is removed. A
# self-loop whose two halves stay within `tunnel_width` label widths of each other is removed.
# In 3D, the label width accounts for the label elongation along z.
#
# Sweep: 0 (off), 1.5, 3 (default), 6. The other knobs keep their defaults.
# Figures: see `README.md` for the legend. In 3D, the bottom row is the XZ projection.

# %%
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path[:0] = [str(HERE), str(HERE.parent)]

from labels import Sample, Scene  # noqa: E402
from report import Report  # noqa: E402

report = Report("03_tunnel_width")
TUNNEL_VALUES = [0, 1.5, 3, 6]
MID = 8

# %% [markdown]
# ## Holes along z
# A tube (radius 4) along x with three holes through it along z, 1, 2 and 3 voxels wide
# (in y) and 10 voxels long (in x), centered on the axis.
#
# **What to look for**
# - At 0, each hole wide enough to split the skeleton gives a multi-edge pair (two edges
#   between the same two junctions).
# - At the default (and 1.5), the pairs are merged: one edge from end to end, no branch
#   points, the kept path through the middle of the tube.

# %%
holed = Scene((17, 64, 152)).tube((MID, 32, 8), (MID, 32, 144), 4)
for x, width in [(30, 1), (70, 2), (110, 3)]:
    y_start = 32 - width // 2
    holed.holes.append((slice(None), slice(y_start, y_start + width), slice(x, x + 10)))
holed_sample = Sample(
    "holes_along_z", holed, "holes 1-3 voxels wide through the tube along z",
    expected={"all": {"branch_points": 0, "leaves": 2, "multi_pairs": 0, "self_loops": 0}})
holed_results = report.sweep(holed_sample, "tunnel_width", TUNNEL_VALUES)

# %%
report.viewer(holed_sample, "3D round")

# %% [markdown]
# ## Holes along y (3D only)
# The same tube with three slits through it along y, 1, 2 and 3 slices thick (in z) and 10
# voxels long, at the middle of the tube: the skeleton splits into a top and a bottom path,
# visible only in the XZ row. In the elongated label the tube is 32 slices tall, so the slits
# are thin compared with the label.
#
# **What to look for**
# - Same as above: one edge from end to end at the default.

# %%
slits = Scene((17, 64, 152)).tube((MID, 32, 8), (MID, 32, 144), 4)
for x, thickness in [(30, 1), (70, 2), (110, 3)]:
    z_start = MID - thickness // 2
    slits.holes.append((slice(z_start, z_start + thickness), slice(None), slice(x, x + 10)))
slits_sample = Sample(
    "holes_along_y", slits, "slits 1-3 slices thick through the tube along y",
    expected={"all": {"branch_points": 0, "leaves": 2, "multi_pairs": 0, "self_loops": 0}},
    only=("3D round", "3D elongated"))
slits_results = report.sweep(slits_sample, "tunnel_width", TUNNEL_VALUES)

# %% [markdown]
# ## Real loops
# A vessel (radius 2, label width 4) that splits into two parallel vessels and joins again,
# three times. The two vessels of each loop run 6, 18 and 24 voxels apart (1.5, 4.5 and 6
# label widths).
#
# **What to look for**
# - The default (3) merges the 1.5-width loop (two vessels touching or nearly) and keeps the
#   4.5- and 6-width loops (multi_pairs 2, branch points 4).
# - 6 merges every loop: the side effect of a high value.
# - Elongated: the label width used by the rule accounts for the elongation; check that the
#   same loops are kept.

# %%
loops = Scene((17, 96, 200))
Y0, RADIUS, LOOP_LENGTH = 48, 2, 44
starts = [16, 76, 136]
loops.tube((MID, Y0, 6), (MID, Y0, starts[0]), RADIUS)
for start, next_start, separation in zip(starts, [*starts[1:], 194], [6, 18, 24], strict=True):
    end = start + LOOP_LENGTH
    for sign in (1, -1):
        y_side = Y0 + sign * separation / 2
        loops.polyline([(MID, Y0, start), (MID, y_side, start + 8), (MID, y_side, end - 8),
                        (MID, Y0, end)], RADIUS)
    loops.tube((MID, Y0, end), (MID, Y0, next_start), RADIUS)
loops_sample = Sample(
    "real_loops", loops, "loops 1.5, 4.5 and 6 label widths wide",
    expected={"all": {"branch_points": 4, "leaves": 2, "multi_pairs": 2}})
loops_results = report.sweep(loops_sample, "tunnel_width", TUNNEL_VALUES)

# %%
report.viewer(loops_sample, "3D elongated")

# %% [markdown]
# ## Rings on stems
# A vessel (radius 2) along x with two stems (radius 2, 16 voxels long) going up, each ending
# on a ring (radius 2): a small one (ring radius 4, so its hole is 2 label widths across) and
# a large one (ring radius 12, 6 label widths). Each ring closes on itself at the stem top: a
# self-loop on that junction.
#
# **What to look for**
# - The default removes the small ring's self-loop (a hole in a blob of label); its stem
#   becomes a leaf branch, long enough to be kept. The large ring is kept (self_loops 1).
# - 6 removes the large ring too.

# %%
rings = Scene((17, 80, 136)).tube((MID, 66, 8), (MID, 66, 128), 2)
for x, ring_radius in [(40, 4), (92, 12)]:
    stem_top = 66 - 2 - 16
    rings.tube((MID, 66, x), (MID, stem_top, x), 2)
    rings.ring((MID, stem_top - ring_radius, x), ring_radius, 2)
rings_sample = Sample(
    "rings_on_stems", rings, "small and large ring on stems",
    expected={"all": {"branch_points": 3, "leaves": 3, "self_loops": 1}})
rings_results = report.sweep(rings_sample, "tunnel_width", TUNNEL_VALUES)

# %% [markdown]
# ## Step trace
# Counts after each refine step that changed the graph (default knobs), for the elongated
# real loops: which step changes the loops.

# %%
report.trace(loops_sample, "3D elongated")

# %%
report.finish()
