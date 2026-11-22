# %% [markdown]
# # junction_reach: split junctions
#
# Rule (`refinement.collapse_internal_bridges`): two junctions (degree >= 3) joined by an
# edge shorter than `junction_reach` times the sum of their radii are merged into one node,
# placed at their midpoint. The thinning often splits a crossing or a junction of several
# branches into two junctions joined by a short edge.
#
# Sweep: 0 (off), 0.75, 1.5 (default), 3. The other knobs keep their defaults.
# Figures: see `README.md` for the legend. In 3D, the bottom row is the XZ projection.

# %%
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np  # noqa: E402
from labels import Sample, Scene  # noqa: E402
from report import Report  # noqa: E402

report = Report("02_junction_reach")
JUNCTION_VALUES = [0, 0.75, 1.5, 3]
MID = 8

# %% [markdown]
# ## Branch pairs at increasing offsets
# A vessel (radius 3) along x with five pairs of long branches (radius 2), one up and one
# down, offset along the vessel by 0, 4, 8, 14 and 20 voxels. The radius at a junction is
# about 3.5, so the sum of the two junction radii is about 7 and the default (1.5) merges
# junctions closer than about 10 voxels.
#
# **What to look for**
# - Offset 0 is a crossing: one node of degree 4 at every value > 0.
# - Offsets 4 and 8: merged at the default (one degree-4 node), kept apart at 0.75 for
#   offset 8.
# - Offsets 14 and 20: two junctions at the default; only 3 merges them.
# - Side effects: the merged node sits at the midpoint, so the edges are re-joined to it;
#   their paths must stay inside the label (`check_graph`).

# %%
offsets_scene = Scene((17, 96, 256)).tube((MID, 48, 6), (MID, 48, 250), 3)
for x, offset in zip([24, 64, 106, 150, 196], [0, 4, 8, 14, 20], strict=True):
    offsets_scene.tube((MID, 48, x), (MID, 20, x), 2)
    offsets_scene.tube((MID, 48, x + offset), (MID, 76, x + offset), 2)
offsets_sample = Sample(
    "branch_pairs", offsets_scene, "pairs of branches at offsets 0-20",
    expected={"all": {"branch_points": 7, "leaves": 12}})
offsets_results = report.sweep(offsets_sample, "junction_reach", JUNCTION_VALUES)

# %%
report.viewer(offsets_sample, "3D elongated")

# %% [markdown]
# ## Crossings at 90° and 30°
# Two crossings of vessels of radius 3 at the same depth. The thinning splits the 30°
# crossing into two junctions far apart (the vessels overlap over a long stretch).
#
# **What to look for**
# - 90°: one degree-4 node at every value > 0.
# - 30°: two junctions joined by a long edge; whether the default merges them (and where it
#   places the merged node).

# %%
crossings = Scene((17, 96, 192))
crossings.tube((MID, 48, 8), (MID, 48, 88), 3).tube((MID, 8, 48), (MID, 88, 48), 3)
# two vessels at +-15 degrees from x cross at (48, 140)
half_length, half_angle = 44, np.radians(15)
for sign in (1, -1):
    dy, dx = sign * half_length * np.sin(half_angle), half_length * np.cos(half_angle)
    crossings.tube((MID, 48 - dy, 140 - dx), (MID, 48 + dy, 140 + dx), 3)
crossings_sample = Sample("crossings", crossings, "crossings at 90 and 30 degrees")
crossings_results = report.sweep(crossings_sample, "junction_reach", JUNCTION_VALUES)

# %% [markdown]
# ## Thick vessel with thin branches
# A thick vessel (radius 6) with thin branches (radius 2) on opposite sides, offset by 0,
# 8, 16 and 24 voxels. The junction radius is the thick one (about 6), so the default merges
# junctions closer than about 18 voxels: real separate branch points 1.3-2.7 thick radii
# apart.
#
# **What to look for**
# - Which offsets are merged at the default; a merged pair is a lost branch point and moves
#   the junction off the branches.

# %%
thick = Scene((25, 96, 216)).tube((12, 48, 14), (12, 48, 202), 6)
for x, offset in zip([24, 68, 112, 160], [0, 8, 16, 24], strict=True):
    thick.tube((12, 48, x), (12, 14, x), 2)
    thick.tube((12, 48, x + offset), (12, 82, x + offset), 2)
thick_sample = Sample("thick_thin", thick, "thin branches on a thick vessel")
thick_results = report.sweep(thick_sample, "junction_reach", JUNCTION_VALUES)

# %% [markdown]
# ## Crossing at different depths (3D only)
# Two vessels (radius 3) crossing in XY, 8 voxels apart in z. In the round label they do
# not touch (8 > 2 x 3); in the elongated label (cross-sections 24 voxels tall) they overlap
# and form one component, as two vessels at different depths do in axially blurred stacks.
#
# **What to look for**
# - Round: two separate vessels, no junction.
# - Elongated: the merged label gives junctions; how many, and whether `junction_reach`
#   turns them into one crossing node. The label cannot tell this from a real anastomosis.

# %%
depths = Scene((25, 96, 96))
depths.tube((8, 48, 8), (8, 48, 88), 3).tube((16, 8, 48), (16, 88, 48), 3)
depths_sample = Sample("depth_crossing", depths, "vessels crossing 8 voxels apart in z",
                       expected={"3D round": {"branch_points": 0, "leaves": 4}},
                       only=("3D round", "3D elongated"))
depths_results = report.sweep(depths_sample, "junction_reach", JUNCTION_VALUES)

# %%
report.viewer(depths_sample, "3D elongated")

# %% [markdown]
# ## Step trace
# Counts after each refine step that changed the graph (default knobs), for the elongated
# branch pairs: whether collapsing junctions creates multi-edges or self-loops that a later
# step removes.

# %%
report.trace(offsets_sample, "3D elongated")

# %%
report.finish()
