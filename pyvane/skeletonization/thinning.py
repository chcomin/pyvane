"""Directional 3D thinning (Lee et al. 1994 scheme) that preserves the end points.

scikit-image's ``skeletonize`` for 3D images (Lee's method) checks the end-point condition only
when the candidate points of a sub-iteration are collected, not when they are deleted one after
the other. Objects of even width therefore erode from their free ends: an isolated 2 x 2 or
4 x 4 bar disappears completely, and leaf branches of even width are shortened back to their
junction or removed. Which branches survive depends on the parity of their width in voxels.

This implementation follows the same scheme (six directional sub-iterations; candidates are
border points that are not end points and are simple; candidates are then deleted
sequentially after re-checking) but re-checks the end-point condition, together with
simplicity, right before each deletion.

A point is simple (26-connectivity for the object, 6-connectivity for the background) when its
26-neighborhood contains exactly one 26-connected component of object points, and its
18-neighborhood contains exactly one 6-connected component of background points 6-adjacent to
it (topological numbers T26 = T6 = 1, Bertrand & Malandain 1994).
"""

import numpy as np
from numba import njit, prange

# ----- neighborhood tables (index = 9 * (dz + 1) + 3 * (dy + 1) + (dx + 1), center = 13) -----
_OFFSETS = np.array([(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
                    np.int64)


def _adjacency(kind: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR adjacency between the 26 neighbors (center excluded). kind 26 or 6."""
    ptr = [0]
    idx = []
    for i in range(27):
        if i != 13:
            for j in range(27):
                if j == 13 or j == i:
                    continue
                d = np.abs(_OFFSETS[i] - _OFFSETS[j])
                if (kind == 26 and d.max() <= 1) or (kind == 6 and d.sum() == 1):
                    idx.append(j)
        ptr.append(len(idx))
    return np.array(ptr, np.int64), np.array(idx, np.int64)


_ADJ26_PTR, _ADJ26 = _adjacency(26)
_ADJ6_PTR, _ADJ6 = _adjacency(6)
_IN_N18 = np.array([i != 13 and np.abs(_OFFSETS[i]).sum() <= 2 for i in range(27)])
_IS_N6 = np.array([np.abs(_OFFSETS[i]).sum() == 1 for i in range(27)])
# 6-neighbor index for each of the six thinning directions: -z, +z, -y, +y, -x, +x
_DIRECTIONS = np.array([4, 22, 10, 16, 12, 14], np.int64)


@njit(cache=True, inline="always")
def _neighborhood(img, z, y, x, nb):
    i = 0
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nb[i] = img[z + dz, y + dy, x + dx]
                i += 1


@njit(cache=True)
def _count_components(nb, value, allowed, ptr, adj, seeds_only, is_seed):
    """Number of components of neighbors with nb == value (restricted to ``allowed``), using the
    given adjacency. If seeds_only, only components containing a seed point are counted."""
    label = np.zeros(27, np.int64)
    stack = np.empty(27, np.int64)
    n_comp = 0
    for start in range(27):
        if start == 13 or not allowed[start] or nb[start] != value or label[start]:
            continue
        n_comp += 1
        has_seed = is_seed[start]
        label[start] = n_comp
        top = 0
        stack[top] = start
        top += 1
        while top:
            top -= 1
            i = stack[top]
            for k in range(ptr[i], ptr[i + 1]):
                j = adj[k]
                if allowed[j] and nb[j] == value and not label[j]:
                    label[j] = n_comp
                    has_seed = has_seed or is_seed[j]
                    stack[top] = j
                    top += 1
        if seeds_only and not has_seed:
            n_comp -= 1
    return n_comp


@njit(cache=True)
def _is_simple(nb, all_true, in_n18, is_n6, p26, a26, p6, a6):
    if _count_components(nb, 1, all_true, p26, a26, False, all_true) != 1:
        return False
    return _count_components(nb, 0, in_n18, p6, a6, True, is_n6) == 1


@njit(cache=True, inline="always")
def _is_endpoint(nb):
    s = 0
    for i in range(27):
        if i != 13 and nb[i]:
            s += 1
    return s == 1


@njit(parallel=True, cache=True)
def _candidates(img, pts, direction_idx, all_true, in_n18, is_n6, p26, a26, p6, a6, offsets):
    n = pts.shape[0]
    flags = np.zeros(n, np.bool_)
    dz, dy, dx = offsets[direction_idx, 0], offsets[direction_idx, 1], offsets[direction_idx, 2]
    for t in prange(n):
        z, y, x = pts[t, 0], pts[t, 1], pts[t, 2]
        if img[z, y, x] == 0 or img[z + dz, y + dy, x + dx] != 0:
            continue
        nb = np.empty(27, np.uint8)
        _neighborhood(img, z, y, x, nb)
        if _is_endpoint(nb):
            continue
        if _is_simple(nb, all_true, in_n18, is_n6, p26, a26, p6, a6):
            flags[t] = True
    return flags


@njit(cache=True)
def _delete_sequentially(img, pts, flags, all_true, in_n18, is_n6, p26, a26, p6, a6):
    nb = np.empty(27, np.uint8)
    changed = 0
    for t in range(pts.shape[0]):
        if not flags[t]:
            continue
        z, y, x = pts[t, 0], pts[t, 1], pts[t, 2]
        _neighborhood(img, z, y, x, nb)
        if _is_endpoint(nb):
            continue
        if _is_simple(nb, all_true, in_n18, is_n6, p26, a26, p6, a6):
            img[z, y, x] = 0
            changed += 1
    return changed


def skeletonize(image: np.ndarray) -> np.ndarray:
    """Thins a 2D or 3D binary image to a one-voxel-wide skeleton, preserving the topology
    (26-connected object, 6-connected background) and the free ends of all branches.

    Args:
        image: Binary image (any dtype; nonzero is object).

    Returns:
        Boolean skeleton with the same shape.
    """
    img = np.asarray(image) != 0
    is_2d = img.ndim == 2
    if is_2d:
        img = img[None]
    if img.ndim != 3:
        raise ValueError("image must be 2D or 3D")
    work = np.pad(img, 1).astype(np.uint8)
    pts = np.argwhere(work).astype(np.int64)
    all_true = np.ones(27, np.bool_)
    unchanged = 0
    while unchanged < 6:
        unchanged = 0
        for d in _DIRECTIONS:
            flags = _candidates(work, pts, d, all_true, _IN_N18, _IS_N6, _ADJ26_PTR, _ADJ26,
                                _ADJ6_PTR, _ADJ6, _OFFSETS)
            changed = 0
            if flags.any():
                changed = _delete_sequentially(work, pts, flags, all_true, _IN_N18, _IS_N6,
                                               _ADJ26_PTR, _ADJ26, _ADJ6_PTR, _ADJ6)
            if changed == 0:
                unchanged += 1
            else:
                pts = pts[work[pts[:, 0], pts[:, 1], pts[:, 2]] != 0]
    out = work[1:-1, 1:-1, 1:-1].astype(bool)
    return out[0] if is_2d else out
