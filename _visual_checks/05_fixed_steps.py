# %% [markdown]
# # Steps without a knob
#
# - False anastomoses (`refinement.remove_false_anastomoses`, always on): an edge between two
#   junctions is removed if its smallest radius is below 0.7 x the smaller junction radius
#   and its length outside the two junctions is below 1 x that radius: a thin, short bridge
#   between two vessels that touch.
# - Rings (`keep_rings` of `create_graph`, default True): isolated loops are kept as a node
#   with a self-loop.
#
# The figures compare named settings; the step trace shows which step removed what.

# %%
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path[:0] = [str(HERE), str(HERE.parent)]

from labels import Sample, Scene  # noqa: E402
from report import Report  # noqa: E402

report = Report("05_fixed_steps")
MID = 8

# %% [markdown]
# ## Bridges between parallel vessels
# Two parallel vessels (radius 4) 12 voxels apart (axis to axis, so 4 voxels of background
# between them) joined by four bridges of radius 1, 2, 3 and 4. Below them, two vessels 24
# voxels apart joined by one thin, long bridge (radius 1).
#
# **What to look for**
# - The thin, short bridges (radius 1 and 2) are removed: the two vessels are separate.
# - The radius-4 bridge is as thick as the vessels: kept. Radius 3 is near the threshold.
# - The long thin bridge is kept (a capillary, not a contact).
# - With the default `junction_reach`, the two junctions of a kept short bridge are close
#   (about 12 voxels, sum of radii about 9-10): they may be merged into one degree-4 node.
#   The `junction_reach = 0` column shows the bridges without that merge.

# %%
bridges = Scene((17, 120, 176))
bridges.tube((MID, 26, 8), (MID, 26, 168), 4).tube((MID, 38, 8), (MID, 38, 168), 4)
for x, radius in [(32, 1), (68, 2), (104, 3), (140, 4)]:
    bridges.tube((MID, 26, x), (MID, 38, x), radius)
bridges.tube((MID, 72, 8), (MID, 72, 168), 4).tube((MID, 96, 8), (MID, 96, 168), 4)
bridges.tube((MID, 72, 88), (MID, 96, 88), 1)
bridges_sample = Sample("bridges", bridges, "bridges of radius 1-4 between vessels")
bridges_results = report.compare(
    bridges_sample, {"default": {}, "junction_reach = 0": {"junction_reach": 0}},
    expected_column="default")

# %%
report.viewer(bridges_sample, "3D round")

# %%
report.trace(bridges_sample, "3D round")

# %% [markdown]
# ## Isolated ring
# A ring (ring radius 12, radius 2) next to a straight vessel.
#
# **What to look for**
# - `keep_rings=True`: the ring is one node with a self-loop (a degree-2 node, diamond).
# - `keep_rings=False`: the ring is gone, the vessel is unchanged.

# %%
ring = Scene((17, 64, 112)).tube((MID, 52, 8), (MID, 52, 104), 2).ring((MID, 24, 56), 12, 2)
ring_sample = Sample("isolated_ring", ring, "a ring next to a vessel",
                     expected={"all": {"leaves": 2, "self_loops": 1}})
ring_results = report.compare(
    ring_sample, {"keep_rings=True (default)": {}, "keep_rings=False": {"keep_rings": False}},
    expected_column="keep_rings=True (default)")

# %%
report.finish()
