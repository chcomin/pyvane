# %% [markdown]
# # Vessels cut by the XY faces of the image
#
# `pyvane.graph.border`: the thinning treats the outside of the image as background, so the
# skeleton of a vessel cut by an image face stops up to about one radius before the face.
#
# - Leaf branches whose mask reaches an XY face (tip within 2 radii + 1 voxel of the face and
#   the straight segment to the face inside the mask) and whose end points to the face
#   (last 3 tip radii within 60° of the face normal) are exempt from leaf pruning.
# - At the end, every leaf tip whose mask reaches an XY face gets `border=True` (magenta
#   outline) and is moved to the face, its path extended. Every edge gets `border_fraction`,
#   the fraction of its path where the mask reaches an XY face (written on the edges).
# - The z faces are not treated this way.
#
# There is no border knob: the figures compare `spur_reach = 0` (no leaf pruning) with the
# default, and the `spur_reach = 0` column shows what the leaf rule measures (reach in
# parent radii, G: projection guide, B: border exemption).

# %%
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np  # noqa: E402
from labels import Sample, Scene, side_branch  # noqa: E402
from report import Report  # noqa: E402

report = Report("04_border")
SPUR_VALUES = [0, 2.5]
MID = 8


def crossing_tube(inside, outward, angle_to_normal, length, radius):
    """Tube from an inside point leaving the image through the face whose outward normal is
    ``outward`` (y, x), at ``angle_to_normal`` degrees from it."""
    normal = np.asarray(outward, float)
    tangent = np.array([-normal[1], normal[0]])
    angle = np.radians(angle_to_normal)
    direction = np.cos(angle) * normal + np.sin(angle) * tangent
    end = np.asarray(inside, float) + length * direction
    return (MID, *inside), (MID, *end), radius


# %% [markdown]
# ## Vessels leaving the image at different angles
# Four vessels (radius 3), each starting inside and leaving through a different XY face at
# 0°, 30°, 60° and 80° from the face normal (the last one grazes the face).
#
# **What to look for**
# - Every face tip is moved to the face and flagged `border=True`, whatever the angle (the
#   flag only needs the mask to reach the face); the inner tips are not flagged.
# - `border_fraction` is the fraction of the path within the tolerance of a face whose
#   straight segment to it stays in the mask: here only the stretch next to the face.
# - The edge lengths include the segment added to reach the face.

# %%
faces = Scene((17, 128, 128))
faces.tubes += [crossing_tube((40, 50), (0, -1), 0, 70, 3),       # x = 0 face
                crossing_tube((50, 80), (-1, 0), 30, 70, 3),      # y = 0 face
                crossing_tube((100, 96), (0, 1), 60, 70, 3),      # x = 127 face
                crossing_tube((108, 16), (1, 0), 80, 100, 3)]     # y = 127 face, grazing
faces_sample = Sample("face_angles", faces, "vessels leaving at 0-80 degrees",
                      expected={"all": {"branch_points": 0, "leaves": 8, "border_leaves": 4}})
faces_results = report.sweep(faces_sample, "spur_reach", SPUR_VALUES, annotate_border=True)

# %%
report.viewer(faces_sample, "3D elongated")

# %% [markdown]
# ## Vessel running along a face
# A vessel (radius 3) running along the y = 0 face, its label touching the face, with a long
# branch going into the image. Its two ends are inside the image along x: it is not cut by
# the face, it only touches it.
#
# **What to look for**
# - The two end tips touch the face, so they are flagged `border=True` and moved to the face
#   (y = 0), although the vessel is not cut there: the ambiguity of the border flag. Whether
#   a tip is flagged depends on where the thinning ends it under the rounded end of the label
#   (the segment from the tip to the face must stay in the mask).
# - `border_fraction` of the edges along the face is close to 1.
# - Elongated: the z-spurs grown by the bumps on a vessel touching the face point along z, so
#   they are not exempt and are still removed at the default.

# %%
along = Scene((17, 80, 128)).tube((MID, 2, 16), (MID, 2, 112), 3)
along.tubes.append(side_branch((MID, 2, 64), (0, 1, 0), 50, 3, 2))
along_sample = Sample(
    "along_face", along, "vessel touching the y = 0 face",
    expected={"all": {"branch_points": 1, "leaves": 3, "border_leaves": 2}})
along_results = report.sweep(along_sample, "spur_reach", SPUR_VALUES, annotate_border=True,
                             annotate_leaves=True)

# %%
report.viewer(along_sample, "3D elongated")

# %%
report.finish()
