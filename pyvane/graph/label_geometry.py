"""Geometry of binary vessel labels used by several graph methods.

In-plane distance maps, the elongation of the label along z (axially blurred labels are taller
than wide), and comparisons of paths and branch directions in units of the local label size.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Cap for distance values (voxels), so that a label touching no background gives finite radii
MAX_RADIUS = 1e3


def inplane_edt(bin_img: np.ndarray) -> np.ndarray:
    """Distance (voxels) of every foreground voxel to the background of its own z slice.

    Args:
        bin_img: 3D binary image (z, y, x).

    Returns:
        float32 array; 1e6 in slices without background.
    """
    distances = np.zeros(bin_img.shape, np.float32)
    for z in range(bin_img.shape[0]):
        plane = bin_img[z]
        if not plane.any():
            continue
        if plane.all():
            distances[z] = np.float32(1e6)
        else:
            distances[z] = ndi.distance_transform_edt(plane)
    return distances


def vessel_radius_map(bin_img: np.ndarray) -> np.ndarray:
    """Map of the vessel radius (voxels), the unit of the distances relative to the vessel size.

    In 3D, the in-plane distance to the background (``inplane_edt``): axially blurred labels
    are taller than wide, so the in-plane distance is the vessel radius. In 2D, the distance
    transform of the background. Values are capped at ``MAX_RADIUS``.

    Args:
        bin_img: 2D or 3D binary image (z, y, x).
    """
    bin_img = np.asarray(bin_img, bool)
    if bin_img.ndim == 3:
        radii = inplane_edt(bin_img)
    else:
        radii = ndi.distance_transform_edt(bin_img).astype(np.float32)
    return np.minimum(radii, np.float32(MAX_RADIUS))


def vessel_radius(radius_map: np.ndarray, voxel) -> float:
    """Vessel radius (voxels) at ``voxel`` from a ``vessel_radius_map``, at least 1."""
    return min(max(float(radius_map[tuple(np.asarray(voxel, int))]), 1.0), MAX_RADIUS)


def half_height(bin_img: np.ndarray, voxel) -> float:
    """Half the length (voxels) of the foreground run along z through ``voxel``."""
    column = bin_img[:, int(voxel[1]), int(voxel[2])]
    z = int(voxel[0])
    if not column[z]:
        return 0.5
    bottom = z
    while bottom > 0 and column[bottom - 1]:
        bottom -= 1
    top = z
    while top < len(column) - 1 and column[top + 1]:
        top += 1
    return (top - bottom + 1) / 2.0


def label_elongation(bin_img: np.ndarray, edt_xy: np.ndarray, points: np.ndarray,
                     n: int = 2000, seed: int = 0) -> float:
    """Median ratio of the half-height to the in-plane radius of the label at skeleton points.

    About 1 for round cross-sections (0.7-0.75 after the voxel discretization), 2.5-5 for the
    axially blurred labels of widefield or confocal stacks.

    Args:
        bin_img: 3D binary image.
        edt_xy: ``inplane_edt(bin_img)``.
        points: (n, 3) skeleton voxels.
        n: Number of points sampled.
        seed: Seed of the sampling.
    """
    if len(points) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(n, len(points)), replace=False)]
    ratios = [half_height(bin_img, voxel) / max(float(edt_xy[tuple(voxel)]), 0.5)
              for voxel in sample]
    return float(np.median(ratios))


def label_elongation_no_skeleton(bin_img: np.ndarray, edt_xy: np.ndarray | None = None,
                                 min_half_width: float = 2.5) -> float:
    """Elongation of the label along z, measured without a skeleton.

    Uses the runs of foreground along z (one run per vessel crossed by a column) whose center
    is on a vessel axis: an in-plane ridge of ``edt_xy`` (a value at least as large as its 8
    in-plane neighbours). The elongation is the median, over these runs, of the half-height
    (half the run length) divided by the in-plane half-width (``edt_xy`` at the run center
    minus half a voxel). Steep vessels give runs much taller than their width; the median
    keeps their weight small in labels where most vessels run in-plane.

    About 1 for round cross-sections, 4-5 for confocal labels and 2.5-3.5 for 10x widefield
    labels. Use it for ``thinning.skeletonize(elongation=...)``.

    Args:
        bin_img: 3D binary image (z, y, x).
        edt_xy: ``inplane_edt(bin_img)`` (computed if not given).
        min_half_width: Runs thinner than this (voxels) are not used: they are label
            fragments or the edges of vessels.

    Returns:
        The elongation; 1 if no run qualifies.
    """
    bin_img = np.asarray(bin_img, bool)
    if edt_xy is None:
        edt_xy = inplane_edt(bin_img)
    edt_xy = np.minimum(edt_xy, np.float32(MAX_RADIUS))
    # runs along z, column by column: (y, x, first z, one past the last z)
    columns_last = np.moveaxis(bin_img, 0, -1).astype(np.int8)
    steps = np.diff(np.pad(columns_last, ((0, 0), (0, 0), (1, 1))), axis=-1)
    y, x, z_first = np.nonzero(steps == 1)
    _, _, z_end = np.nonzero(steps == -1)
    z_center = (z_first + z_end - 1) // 2
    center_width = edt_xy[z_center, y, x]
    on_ridge = ndi.maximum_filter(edt_xy, size=(1, 3, 3))[z_center, y, x] <= center_width
    half_width = center_width - 0.5
    # runs cut by the first or last slice have an unknown height
    inside = (z_first > 0) & (z_end < bin_img.shape[0])
    keep = on_ridge & inside & (half_width >= min_half_width)
    if not keep.any():
        return 1.0
    half_height = (z_end - z_first) / 2
    return float(np.median(half_height[keep] / half_width[keep]))


def paths_close(path_a: np.ndarray, path_b: np.ndarray, edt_xy: np.ndarray,
                elongation: float, width: float = 1.0) -> bool:
    """Whether two paths stay within ``width`` local label widths of each other.

    The local label width is 2 x the in-plane distance to the background + 1 voxel, taken at
    each point of ``edt_xy`` (in 2D, the distance transform); in 3D, z distances are divided
    by ``elongation``.
    """
    if len(path_a) == 0 or len(path_b) == 0:
        return True
    scale = np.ones(edt_xy.ndim)
    if edt_xy.ndim == 3:
        scale[0] = 1.0 / elongation
    for path, other_path in ((path_a, path_b), (path_b, path_a)):
        distances, _ = cKDTree(other_path * scale).query(path * scale)
        inplane_radii = np.minimum(edt_xy[tuple(np.asarray(path, int).T)], MAX_RADIUS)
        label_widths = 2.0 * inplane_radii + 1.0
        if np.any(distances > width * label_widths):
            return False
    return True


def best_pairing(directions: np.ndarray):
    """Pairing of 4 unit vectors into 2 pairs that maximizes the smaller angle (degrees)
    between the vectors of a pair. Returns (smaller angle, pairs)."""
    pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    best_angle, best_pairs = -1.0, pairings[0]
    for pairs in pairings:
        angles = [np.degrees(np.arccos(np.clip(directions[i] @ directions[j], -1, 1)))
                  for i, j in pairs]
        if min(angles) > best_angle:
            best_angle, best_pairs = min(angles), pairs
    return best_angle, best_pairs
