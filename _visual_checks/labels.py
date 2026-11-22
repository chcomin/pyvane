"""Synthetic labels for the visual checks of the refine heuristics.

Every sample is a scene of tubes (``Scene``), with its vessels around the middle slice, made in
three variants (``variants``):

- ``2D``: the middle slice of the round label;
- ``3D round``: tubes with round cross-sections;
- ``3D elongated``: the same tubes at the same positions, with cross-sections ``ELONGATION``
  times taller than wide (the Z/XY elongation of the confocal labels, which comes from the
  axial blur), plus small bumps on the top and bottom surfaces of the vessels
  (``add_bumps``), which make the thinning grow spurs along z as in the real labels. The stack
  is padded in z so that the taller vessels fit; vessels at different depths can touch.

The labels are not realistic vessels: each one isolates a situation a heuristic must handle.
"""

import math
from dataclasses import dataclass, field

import numpy as np

# Z/XY elongation of the confocal labels
ELONGATION = 4.0
# Radius (voxels) of the bumps added to the elongated variant
BUMP_RADIUS = 2

VARIANTS = ("2D", "3D round", "3D elongated")


@dataclass
class Scene:
    """Tubes in (z, y, x) voxel coordinates of the round label.

    Attributes:
        shape: Shape of the round label (the elongated one is padded in z).
        tubes: (start, end, radius) of each tube.
        holes: (z slice, y slice, x slice) boxes set to background after drawing, in the
            coordinates of the round label (``slice(None)`` for every slice).
        bumps: (y, x) columns where the elongated variant gets bumps on the top and bottom
            surfaces of the label. None places one bump every 12 voxels along every tube that
            is not vertical.
        pad_z: Whether the elongated variant is padded in z (False keeps the round shape, so
            that vessels cut by the z faces stay cut).
    """
    shape: tuple[int, int, int]
    tubes: list = field(default_factory=list)
    holes: list = field(default_factory=list)
    bumps: list | None = None
    pad_z: bool = True

    def tube(self, start, end, radius):
        """Adds a tube and returns the scene, so that calls can be chained."""
        self.tubes.append((start, end, radius))
        return self

    def polyline(self, points, radius):
        """Adds tubes joining consecutive points and returns the scene."""
        for start, end in zip(points[:-1], points[1:]):
            self.tube(start, end, radius)
        return self

    def ring(self, center, ring_radius: float, radius: float, n_segments: int = 32):
        """Adds a ring of tubes in the XY plane around ``center`` and returns the scene."""
        angles = np.linspace(0, 2 * np.pi, n_segments + 1)
        z, y, x = center
        points = [(z, y + ring_radius * np.sin(a), x + ring_radius * np.cos(a)) for a in angles]
        return self.polyline(points, radius)


def draw(scene: Scene, elongation: float = 1.0) -> np.ndarray:
    """Binary 3D label of a scene with cross-sections ``elongation`` times taller than wide.

    A voxel is foreground if its distance to a tube axis is at most the tube radius, with z
    distances divided by ``elongation``. With ``elongation`` > 1 and ``scene.pad_z``, the stack
    is padded in z by (``elongation`` - 1) x the largest radius on each side.
    """
    max_radius = max(radius for *_, radius in scene.tubes)
    pad = math.ceil((elongation - 1) * max_radius) if scene.pad_z else 0
    shape = (scene.shape[0] + 2 * pad, *scene.shape[1:])
    zz, yy, xx = np.indices(shape).astype(float)
    # z scaled by the elongation: the anisotropic distance becomes Euclidean
    voxels = np.stack([(zz - pad) / elongation, yy, xx], -1)
    scale = np.array([1 / elongation, 1, 1])
    label = np.zeros(shape, bool)
    for start, end, radius in scene.tubes:
        start, end = np.asarray(start, float) * scale, np.asarray(end, float) * scale
        axis = end - start
        t = np.clip(((voxels - start) @ axis) / (axis @ axis), 0, 1)
        label |= np.linalg.norm(voxels - (start + t[..., None] * axis), axis=-1) <= radius
    for z_range, y_range, x_range in scene.holes:
        if z_range.start is not None or z_range.stop is not None:
            z_range = slice((z_range.start or 0) + pad,
                            (scene.shape[0] if z_range.stop is None else z_range.stop) + pad)
        label[z_range, y_range, x_range] = False
    return label


def side_branch(origin, direction, protrusion: float, parent_radius: float,
                branch_radius: float) -> tuple[tuple, tuple, float]:
    """A tube leaving a parent vessel whose label protrudes ``protrusion`` voxels out of the
    parent surface (the bulge length of the refine leaf rule).

    Args:
        origin: Point (z, y, x) on the parent axis.
        direction: Direction (dz, dy, dx) of the branch (normalized here).
        protrusion: How far the branch label reaches beyond the parent surface.
        parent_radius: Radius of the parent vessel.
        branch_radius: Radius of the branch.

    Returns:
        (start, end, radius) of the tube, for ``Scene.tubes``.
    """
    origin = np.asarray(origin, float)
    direction = np.asarray(direction, float) / np.linalg.norm(direction)
    axis_length = protrusion + parent_radius - branch_radius
    end = origin + axis_length * direction
    return tuple(origin), tuple(end), branch_radius


def _default_bump_columns(scene: Scene, spacing: float = 12.0) -> list[tuple[int, int]]:
    """One (y, x) column every ``spacing`` voxels along every tube of the scene."""
    columns = []
    for start, end, _ in scene.tubes:
        start, end = np.asarray(start, float), np.asarray(end, float)
        n_bumps = int(np.linalg.norm((end - start)[1:]) // spacing)
        for t in (np.arange(n_bumps) + 0.5) / max(n_bumps, 1):
            y, x = (start + t * (end - start))[1:]
            columns.append((round(y), round(x)))
    return columns


def add_bumps(label: np.ndarray, columns, radius: int = BUMP_RADIUS) -> np.ndarray:
    """Adds a ball of ``radius`` on the top and on the bottom surface of the label in each
    (y, x) column (columns without foreground or outside the image are skipped)."""
    label = label.copy()
    zz, yy, xx = np.indices(label.shape)
    for y, x in columns:
        if not (0 <= y < label.shape[1] and 0 <= x < label.shape[2]):
            continue
        z_inside = np.nonzero(label[:, y, x])[0]
        if len(z_inside) == 0:
            continue
        for z in (z_inside[0], z_inside[-1]):
            label |= (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    return label


def variants(scene: Scene) -> dict[str, np.ndarray]:
    """The 2D, 3D round and 3D elongated labels of a scene (see the module docstring)."""
    round_label = draw(scene)
    elongated = draw(scene, ELONGATION)
    columns = _default_bump_columns(scene) if scene.bumps is None else scene.bumps
    elongated = add_bumps(elongated, columns)
    return {"2D": round_label[scene.shape[0] // 2], "3D round": round_label,
            "3D elongated": elongated}


@dataclass
class Sample:
    """A scene to check, with what to look for in the figures.

    Attributes:
        name: Short name, used in file names.
        scene: The scene drawn in the three variants.
        what_to_look_for: The intended behavior and the likely side effects.
        expected: Counts expected at the default knobs, per variant ("all" for every variant),
            e.g. ``{"all": {"leaves": 4}}``. Only counts that follow from the design of the
            scene are given.
        only: Variants to run (default: all).
    """
    name: str
    scene: Scene
    what_to_look_for: str
    expected: dict = field(default_factory=dict)
    only: tuple = VARIANTS

    def labels(self) -> dict[str, np.ndarray]:
        """The labels of the variants to run."""
        return {name: label for name, label in variants(self.scene).items()
                if name in self.only}

    def expected_for(self, variant: str) -> dict:
        """Expected counts for one variant."""
        return {**self.expected.get("all", {}), **self.expected.get(variant, {})}

