"""Figures of the visual checks: a static figure of a knob sweep and an interactive slice viewer.

Static figure (``sweep_figure``): one column for the label and its skeleton, then one column
per graph (e.g. per knob value). In 3D, each column shows the maximum projection along z (XY,
top) and along y (XZ, bottom, z upward), so that branches growing along z are visible.

- label: dark gray;
- skeleton and graph paths: colored by z (``DEPTH_CMAP``); in a projection the voxel of
  largest z is drawn. One color in 2D;
- removed paths (outside the junction spheres): light gray;
- nodes: shape by degree (circle 1, diamond 2 (rings), triangle 3, square 4+), filled with
  their z color, outlined in magenta if ``border=True``.

Interactive viewer (``slice_viewer``): one slice of the label with toggles for the layers.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D

from pipeline import GraphResult

DEPTH_CMAP = plt.get_cmap("turbo")
COLOR_2D = "#33c3ff"            # skeleton and paths in 2D
LABEL_GRAY = 0.28
REMOVED_COLOR = (0.8, 0.8, 0.8, 0.55)
BORDER_COLOR = "magenta"
PANEL_INCHES = 3.4


def degree_marker(degree: int) -> str:
    """Marker of a node of a given degree."""
    return {1: "o", 2: "D", 3: "^"}.get(degree, "s")


def _as_points(paths, n_dims: int) -> np.ndarray:
    """All voxels of a list of paths, as an (N, n_dims) int array."""
    arrays = [np.asarray(path, int).reshape(-1, n_dims) for path in paths if len(path)]
    return np.vstack(arrays) if arrays else np.zeros((0, n_dims), int)


def graph_points(result: GraphResult) -> np.ndarray:
    """Voxels of the edge paths of a graph."""
    return _as_points([path for *_, path in result.graph.edges(data="path")],
                      result.label.ndim)


def graph_nodes(result: GraphResult) -> list[tuple[np.ndarray, int, bool]]:
    """(center, degree, border) of each node."""
    return [(np.asarray(data["center"]), result.graph.degree(node), bool(data.get("border")))
            for node, data in result.graph.nodes(data=True)]


# ----------------------------------------------------------------------------- static figure

def _projection_axes(view: str) -> tuple[int, int, int]:
    """(projected axis, vertical axis, horizontal axis) of a 3D view."""
    return (0, 1, 2) if view == "XY" else (1, 0, 2)


def _depth_rgba(points: np.ndarray, shape, view: str, depth: int) -> np.ndarray:
    """RGBA image of 3D points projected along the view, colored by z (largest z on top)."""
    _, vertical, horizontal = _projection_axes(view)
    image = np.zeros((shape[vertical], shape[horizontal], 4))
    if len(points) == 0:
        return image
    points = points[np.argsort(points[:, 0])]          # larger z drawn last
    image[points[:, vertical], points[:, horizontal]] = DEPTH_CMAP(
        points[:, 0] / max(depth - 1, 1))
    return image


def _flat_rgba(points: np.ndarray, shape, color) -> np.ndarray:
    """RGBA image of 2D points in one color."""
    image = np.zeros((*shape, 4))
    if len(points):
        image[points[:, 0], points[:, 1]] = colors.to_rgba(color)
    return image


def _draw_points(ax, points: np.ndarray, label: np.ndarray, view: str | None):
    """Draws points (skeleton or paths) projected along a view (None in 2D)."""
    if label.ndim == 2:
        ax.imshow(_flat_rgba(points, label.shape, COLOR_2D), interpolation="nearest")
    else:
        ax.imshow(_depth_rgba(points, label.shape, view, label.shape[0]),
                  interpolation="nearest", origin="lower" if view == "XZ" else "upper")


def _draw_removed(ax, result: GraphResult, view: str | None):
    """Draws the removed paths in light gray."""
    label = result.label
    points = _as_points(result.removed_paths, label.ndim)
    if len(points) == 0:
        return
    if label.ndim == 3:
        _, vertical, horizontal = _projection_axes(view)
        points = points[:, [vertical, horizontal]]
        shape = (label.shape[vertical], label.shape[horizontal])
    else:
        shape = label.shape
    image = np.zeros((*shape, 4))
    image[points[:, 0], points[:, 1]] = REMOVED_COLOR
    ax.imshow(image, interpolation="nearest",
              origin="lower" if view == "XZ" else "upper")


def _draw_nodes(ax, result: GraphResult, view: str | None, size: float = 55):
    """Draws the nodes: shape by degree, z color, magenta outline for border nodes."""
    label = result.label
    for center, degree, border in graph_nodes(result):
        if label.ndim == 3:
            _, vertical, horizontal = _projection_axes(view)
            y, x = center[vertical], center[horizontal]
            face = DEPTH_CMAP(center[0] / max(label.shape[0] - 1, 1))
        else:
            (y, x), face = center, "gold"
        ax.scatter([x], [y], s=size * (0.5 if degree == 2 else 1), marker=degree_marker(degree),
                   c=[face], edgecolors=BORDER_COLOR if border else "black",
                   linewidths=2.4 if border else 0.7, zorder=5)


def _annotate_border_fraction(ax, result: GraphResult, view: str | None):
    """Writes the ``border_fraction`` of each edge with a positive one at its path middle."""
    label = result.label
    for _, _, data in result.graph.edges(data=True):
        fraction = data.get("border_fraction", 0.0)
        path = np.asarray(data["path"], int).reshape(-1, label.ndim)
        if fraction <= 0 or len(path) == 0:
            continue
        middle = path[len(path) // 2]
        if label.ndim == 3:
            _, vertical, horizontal = _projection_axes(view)
            y, x = middle[vertical], middle[horizontal]
        else:
            y, x = middle
        ax.text(x, y, f"{fraction:.2f}", color="white", fontsize=7, ha="center",
                va="bottom", zorder=6,
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 1, "linewidth": 0})


def _annotate_leaves(ax, leaf_notes, guide_tips: np.ndarray, view: str | None):
    """Writes the leaf notes next to the tips and draws the guide tips (XY row only)."""
    for note in leaf_notes:
        if view == "XZ":
            continue
        y, x = note.center[-2:]
        ax.text(x + 2, y - 2, note.text(), color="white", fontsize=7, zorder=6,
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 1, "linewidth": 0})
    if view == "XY" and len(guide_tips):
        ax.scatter(guide_tips[:, 1], guide_tips[:, 0], marker="x", s=40, c="white",
                   linewidths=1.2, zorder=6)


def _draw_label(ax, label: np.ndarray, view: str | None):
    """Draws the label (maximum projection along the view in 3D) in dark gray."""
    if label.ndim == 3:
        projected, _, _ = _projection_axes(view)
        image = label.max(axis=projected)
    else:
        image = label
    ax.imshow(image * LABEL_GRAY, cmap="gray", vmin=0, vmax=1, interpolation="nearest",
              origin="lower" if view == "XZ" else "upper")
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])


def counts_title(result: GraphResult) -> str:
    """Short summary of the counts of a graph, for a panel title."""
    counts = result.counts
    text = (f"bp {counts['branch_points']}  leaves {counts['leaves']}"
            f" (border {counts['border_leaves']})\nmulti {counts['multi_pairs']}"
            f"  loops {counts['self_loops']}  len {counts['length']:.0f}")
    if result.problems:
        text += f"\n{len(result.problems)} check_graph problems"
    return text


def sweep_figure(results: dict[str, GraphResult], title: str,
                 annotate_border: bool = False, notes: dict | None = None) -> plt.Figure:
    """Static figure of the skeleton and of several graphs of the same label.

    Args:
        results: {column title: GraphResult}, all made from the same label and skeleton.
        title: Figure title.
        annotate_border: Write the ``border_fraction`` of the edges that have one.
        notes: {column title: ``pipeline.leaf_notes`` of that column}: the leaves are
            annotated with their reach in parent radii (G: matched to the projection guide,
            B: border exemption) and the guide tips are drawn as white crosses (XY row).
    """
    notes = notes or {}
    first = next(iter(results.values()))
    label = first.label
    views = ["XY", "XZ"] if label.ndim == 3 else [None]
    n_columns = len(results) + 1
    heights = [label.shape[1], label.shape[0]] if label.ndim == 3 else [label.shape[0]]
    width = label.shape[-1]
    fig_height = PANEL_INCHES * sum(heights) / width + 1.1
    fig, axes = plt.subplots(len(views), n_columns, squeeze=False,
                             figsize=(PANEL_INCHES * n_columns, fig_height),
                             gridspec_kw={"height_ratios": heights})
    for row, view in enumerate(views):
        ax = axes[row, 0]
        _draw_label(ax, label, view)
        _draw_points(ax, np.argwhere(first.skeleton), label, view)
        if row == 0:
            ax.set_title("label + skeleton", fontsize=9)
        for column, (name, result) in enumerate(results.items(), start=1):
            ax = axes[row, column]
            _draw_label(ax, label, view)
            _draw_removed(ax, result, view)
            _draw_points(ax, graph_points(result), label, view)
            _draw_nodes(ax, result, view)
            if annotate_border:
                _annotate_border_fraction(ax, result, view)
            if name in notes:
                _annotate_leaves(ax, *notes[name], view)
            if row == 0:
                ax.set_title(f"{name}\n{counts_title(result)}", fontsize=8,
                             color="red" if result.problems else "black")
        if view is not None:
            axes[row, 0].set_ylabel(view, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06 if label.ndim == 2 else 0.04, 1, 1))
    _add_legend(fig, label)
    return fig


def _add_legend(fig, label: np.ndarray):
    """Legend of the node markers and, in 3D, the z colorbar."""
    handles = [Line2D([], [], marker=degree_marker(degree), linestyle="", markersize=7,
                      markerfacecolor="white", markeredgecolor="black", label=name)
               for degree, name in [(1, "degree 1"), (2, "degree 2 (ring)"), (3, "degree 3"),
                                    (4, "degree 4+")]]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=7,
                          markerfacecolor="white", markeredgecolor=BORDER_COLOR,
                          markeredgewidth=2.4, label="border=True"))
    handles.append(Line2D([], [], color=REMOVED_COLOR[:3], linewidth=4, label="removed path"))
    fig.legend(handles=handles, loc="lower left", ncol=len(handles), fontsize=8,
               frameon=False)
    if label.ndim == 3:
        mappable = plt.cm.ScalarMappable(colors.Normalize(0, label.shape[0] - 1), DEPTH_CMAP)
        colorbar_ax = fig.add_axes((0.78, 0.015, 0.2, 0.015))
        fig.colorbar(mappable, cax=colorbar_ax, orientation="horizontal")
        colorbar_ax.set_title("z (slice)", fontsize=8)
        colorbar_ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------- slice viewer

LAYER_COLORS = {"skeleton": "#ffb000", "paths": "#33c3ff", "removed": "#d0d0d0",
                "nodes": "#ff3355"}


class _ViewerLayers:
    """Voxels of every layer of one graph result, for slicing."""

    def __init__(self, result: GraphResult):
        self.skeleton = np.argwhere(result.skeleton)
        self.paths = graph_points(result)
        self.removed = _as_points(result.removed_paths, result.label.ndim)
        self.nodes = graph_nodes(result)


def _slice_image(points: np.ndarray, axis: int, index: int, shape, color) -> np.ndarray:
    """RGBA image of the points lying on slice ``index`` of ``axis``."""
    kept = points[points[:, axis] == index] if points.ndim == 2 and len(points) else points
    other_axes = [a for a in range(len(shape)) if a != axis]
    image = np.zeros((shape[other_axes[0]], shape[other_axes[1]], 4))
    if len(kept):
        image[kept[:, other_axes[0]], kept[:, other_axes[1]]] = colors.to_rgba(color)
    return image


def slice_viewer(results: dict[str, GraphResult], title: str = ""):
    """Interactive viewer of a label slice by slice, with the layers toggled on and off.

    Controls: the slice plane (XY or XZ, 3D only), the slice index, the graph shown (one of
    ``results``) and checkboxes for the skeleton, the graph paths, the nodes (those within one
    slice of the shown one, outlined in magenta if ``border=True``) and the removed paths.

    Args:
        results: {name: GraphResult}, all made from the same label.
        title: Shown above the controls.

    Returns:
        The ipywidgets container (displayed when it is the last value of a cell).
    """
    import ipywidgets as widgets

    first = next(iter(results.values()))
    label = first.label
    layers = {name: _ViewerLayers(result) for name, result in results.items()}
    is_3d = label.ndim == 3

    plane = widgets.ToggleButtons(options=["XY", "XZ"], value="XY", description="plane",
                                  layout={"display": "flex" if is_3d else "none"})
    index = widgets.IntSlider(value=label.shape[0] // 2 if is_3d else 0, min=0,
                              max=label.shape[0] - 1 if is_3d else 0, description="slice",
                              continuous_update=False,
                              layout={"width": "500px",
                                      "display": "flex" if is_3d else "none"})
    graph_name = widgets.Dropdown(options=list(results), description="graph")
    toggles = {name: widgets.Checkbox(value=name != "removed", description=name, indent=False,
                                      layout={"width": "110px"})
               for name in ["skeleton", "paths", "nodes", "removed"]}
    output = widgets.Output()

    def on_plane_change(change):
        axis = 0 if change["new"] == "XY" else 1
        index.max = label.shape[axis] - 1
        index.value = label.shape[axis] // 2

    plane.observe(on_plane_change, names="value")

    def redraw(*_):
        from IPython.display import display

        with output:
            output.clear_output(wait=True)
            fig = _draw_slice(label, layers[graph_name.value], plane.value if is_3d else None,
                              index.value, {name: box.value for name, box in toggles.items()})
            display(fig)
            plt.close(fig)

    for control in [plane, index, graph_name, *toggles.values()]:
        control.observe(redraw, names="value")
    redraw()
    header = widgets.HTML(f"<b>{title}</b>") if title else widgets.HTML("")
    return widgets.VBox([header, widgets.HBox([plane, graph_name]), index,
                         widgets.HBox(list(toggles.values())), output])


def _draw_slice(label: np.ndarray, layers: _ViewerLayers, view: str | None, index: int,
                shown: dict[str, bool]) -> plt.Figure:
    """Figure of one slice of the viewer."""
    if view is None:
        axis, image, shape = None, label, label.shape
    else:
        axis = 0 if view == "XY" else 1
        image = label[index] if axis == 0 else label[:, index, :]
        shape = label.shape
    origin = "lower" if view == "XZ" else "upper"
    fig, ax = plt.subplots(figsize=(7, 7 * image.shape[0] / image.shape[1] + 0.4))
    ax.imshow(image * LABEL_GRAY * 1.5, cmap="gray", vmin=0, vmax=1, interpolation="nearest",
              origin=origin)
    for name in ["removed", "skeleton", "paths"]:
        if not shown[name]:
            continue
        points = getattr(layers, name)
        if axis is None:
            layer_image = _flat_rgba(points, shape, LAYER_COLORS[name])
        else:
            layer_image = _slice_image(points, axis, index, shape, LAYER_COLORS[name])
        if name == "skeleton" and shown["paths"]:
            layer_image[..., 3] *= 0.6
        ax.imshow(layer_image, interpolation="nearest", origin=origin)
    if shown["nodes"]:
        for center, degree, border in layers.nodes:
            if axis is not None and abs(center[axis] - index) > 1:
                continue
            y, x = (center if axis is None
                    else [center[a] for a in range(3) if a != axis])
            ax.scatter([x], [y], s=70, marker=degree_marker(degree), c=LAYER_COLORS["nodes"],
                       edgecolors=BORDER_COLOR if border else "black",
                       linewidths=2.4 if border else 0.7, zorder=5)
    ax.set_title(f"{view} slice {index}" if view else "2D", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
