# %% [markdown]
# # spur_reach: leaf branches
#
# Rule (`guided.prune_leaves`): a leaf branch is removed if it reaches less than
# `spur_reach` parent radii out of its parent vessel in the image plane (the bulge length:
# largest in-plane distance to the parent centerline, minus the parent radius, plus the tip
# radius; the parent radius is the mean radius along the parent centerline). Exceptions:
#
# - 3D: a leaf whose tip matches a tip of the graph of the label projected along z (the
#   *projection guide*, itself pruned with the same `spur_reach`) is kept for good;
# - 3D: a leaf whose tip is on the top or bottom face is kept;
# - a leaf whose label reaches an XY face and whose end points to it is kept (`border.py`).
#
# Sweep: 0 (off), 1.5, 2.5 (default), 4. The other knobs keep their defaults.
#
# Units: pyvane measures radii with the distance transform, about r + 1 for a tube drawn
# with radius r (the distance from the axis to the nearest background voxel). The scenes below
# give protrusions in drawn radii; a protrusion of p drawn radii r measures about
# (p r + 1) / (r + 1) radii for pyvane.
# Figures: see `README.md` for the legend. In 3D, the bottom row is the XZ projection. In the
# `spur_reach = 0` column, each leaf attached to a junction is annotated with what the rule
# measures: its reach in parent radii (removed below `spur_reach`), G if it matches a tip of
# the projection guide (white crosses) and B if the border exemption applies.

# %%
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path[:0] = [str(HERE), str(HERE.parent)]

from labels import Sample, Scene, side_branch  # noqa: E402
from report import Report  # noqa: E402

report = Report("01_spur_reach")
SPUR_VALUES = [0, 1.5, 2.5, 4]

# %% [markdown]
# ## Comb
# A vessel (radius 3) along x with five side branches (radius 2) protruding 0.5, 2, 2.5,
# 4.5 and 6 drawn parent radii out of it, alternately up and down; two of them leave at 45°.
#
# **What to look for**
# - At the default (2.5) the branches of 0.5-2.5 drawn radii (about 0.6-2.1 measured) are
#   removed and those of 4.5 and 6 (about 3.6 and 4.8) kept, in 2D and 3D alike: the
#   projection guide is pruned with the same `spur_reach`. 0 keeps every branch, 4 keeps the
#   longest only (the 4.5 one is borderline).
# - Elongated: the bumps on the top and bottom surfaces grow spurs along z (visible in the XZ
#   row at `spur_reach = 0`); they must be removed at the default without removing the side
#   branches.
# - Side effects: the main vessel must stay one edge from end to end between the kept
#   branches; no `check_graph` problems.

# %%
PARENT_RADIUS, BRANCH_RADIUS, MID = 3, 2, 8
comb = Scene((17, 80, 136)).tube((MID, 40, 8), (MID, 40, 128), PARENT_RADIUS)
for x, direction, protrusion in [(24, (0, -1, 0), 0.5), (46, (0, 1, 0), 2),
                                 (68, (0, -1, -1), 2.5), (90, (0, 1, 0), 4.5),
                                 (112, (0, -1, 1), 6)]:
    comb.tubes.append(side_branch((MID, 40, x), direction, protrusion * PARENT_RADIUS,
                                  PARENT_RADIUS, BRANCH_RADIUS))
comb_sample = Sample(
    "comb", comb, "branches of 0.5-6 drawn radii",
    expected={"all": {"branch_points": 2, "leaves": 4}})
comb_results = report.sweep(comb_sample, "spur_reach", SPUR_VALUES,
                            annotate_leaves=True)

# %%
report.viewer(comb_sample, "3D elongated")

# %% [markdown]
# ## Short branches near a face
# A vessel (radius 3) along x at y = 9, so that its surface is 2 radii from the y = 0 face,
# with three short branches (radius 2) towards the face:
# - x = 24: perpendicular, cut by the face;
# - x = 48: perpendicular, ending 0.5 radius out of the parent (its label does not reach
#   the face);
# - x = 72: at 45°, cut by the face.
#
# **What to look for**
# - The two cut branches are kept at every `spur_reach` (border exemption), their tips are
#   moved to the face and flagged `border=True` (magenta outline).
# - The branch that does not reach the face is removed for `spur_reach` > 0.
# - Elongated: the z-spurs grown by the bumps near the face are still removed (they point
#   along z, not to the face).

# %%
near_face = Scene((17, 64, 96)).tube((MID, 9, 10), (MID, 9, 86), PARENT_RADIUS)
near_face.tube((MID, 9, 24), (MID, -4, 24), BRANCH_RADIUS)
near_face.tubes.append(side_branch((MID, 9, 48), (0, -1, 0), 0.5 * PARENT_RADIUS,
                                   PARENT_RADIUS, BRANCH_RADIUS))
near_face.tube((MID, 9, 66), (MID, -4, 79), BRANCH_RADIUS)
near_face_sample = Sample(
    "near_face", near_face, "short branches cut by the y = 0 face",
    expected={"all": {"branch_points": 2, "leaves": 4, "border_leaves": 2}})
near_face_results = report.sweep(near_face_sample, "spur_reach", SPUR_VALUES,
                                 annotate_leaves=True)

# %%
report.viewer(near_face_sample, "3D elongated")

# %% [markdown]
# ## Branches along z (3D only)
# A vessel (radius 3) along x in the middle of a deeper stack, with three branches
# (radius 2) leaving it upwards (the elongated variant is not padded in z, so that the first
# branch stays cut by the top face):
# - x = 24: vertical, cut by the top face;
# - x = 48: vertical, ending 2 radii above the parent surface (inside the stack);
# - x = 72: going up and sideways, protruding about 4.3 drawn radii in the image plane.
#
# **What to look for**
# - The branch cut by the top face should be kept by the z-face exemption (tip on the top
#   face). The thinning stops the skeleton before the face, so the tip is not on the face and
#   the branch is removed (known; a column-based exemption kept too many z-spurs on the
#   benchmark, `outputs/refine_eval/leaf_reach`).
# - The vertical branch ending inside cannot be told from a z-spur: it is removed for
#   `spur_reach` > 0 (in the projection it is a blob on the vessel, with no tip).
# - The oblique branch has a tip in the projection and is kept by the projection guide.

# %%
DEEP_MID = 16
z_branches = Scene((33, 64, 96), pad_z=False)
z_branches.tube((DEEP_MID, 32, 10), (DEEP_MID, 32, 86), PARENT_RADIUS)
z_branches.tube((DEEP_MID, 32, 24), (40, 32, 24), BRANCH_RADIUS)
z_branches.tubes.append(side_branch((DEEP_MID, 32, 48), (1, 0, 0), 2 * PARENT_RADIUS,
                                    PARENT_RADIUS, BRANCH_RADIUS))
z_branches.tube((DEEP_MID, 32, 72), (DEEP_MID + 10, 32 - 14, 72), BRANCH_RADIUS)
z_branches_sample = Sample(
    "z_branches", z_branches, "branches leaving the vessel along z",
    expected={"all": {"branch_points": 2, "leaves": 4}}, only=("3D round", "3D elongated"))
z_branches_results = report.sweep(z_branches_sample, "spur_reach", SPUR_VALUES,
                                  annotate_leaves=True)

# %%
report.viewer(z_branches_sample, "3D round")

# %% [markdown]
# ## Step trace
# Counts after each refine step that changed the graph (default knobs), for the elongated
# comb: which step removes the z-spurs, and whether a later step changes the result.

# %%
report.trace(comb_sample, "3D elongated")

# %%
report.finish()
