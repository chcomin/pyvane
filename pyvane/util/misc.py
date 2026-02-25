"""Utility functions for image and graph manipulation as well as some useful classes."""

import heapq
import itertools
from collections.abc import Hashable
from typing import Any

import networkx as nx
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter1d


class PriorityQueue:
    """Priority queue that allows changes to, or removal of, elements of a pending task."""

    def __init__(
            self,
            priorities: list | None = None,
            keys: list | None = None,
            data: list | None = None,
            ) -> None:
        """Args:
        priorities: The priorities of the tasks. Lower values are executed first. The value
            can be any comparable type. If None, an empty queue is created.
        keys: List of unique keys for each task. Must have the same length as `priorities`.
        data: Data to associate with each task. Elements can be any object. Must have the
            same length as `priorities`. If None, all tasks will have data set to None.
        """
        

        if priorities is None:
            if keys is None:
                priorities = []
                keys = []
                data = []
            else:
                raise ValueError("`priorities` and `keys` need to be both None or both list.")
        else:
            if keys is None:
                raise ValueError("`priorities` and `keys` need to be both None or both list.")

        num_entries = len(priorities)

        if data is None:
            data = [None]*num_entries

        entries = list(map(list, zip(priorities, range(num_entries), keys, data)))
        heapq.heapify(entries)
        _, _, keys, _ = zip(*entries) # Obtain keys again since the order of the entries has changed
        entries_map = dict(zip(keys, entries))

        self.queue = entries
        self.entries_map = entries_map
        self.counter = itertools.count(start=num_entries)
        self.removed_tag = "<removed>"

    def add_task(self, priority: float, key: Hashable, data: Any = None) -> None:
        """Adds a new task or updates the priority of an existing task.

        Args:
            priority: The priority of the task. Lower values are executed first. The value
                can be any comparable type.
            key: Unique key for the task.
            data: Data to associate with the task.
        """

        entries_map = self.entries_map
        if key in entries_map:
            self.remove_task(key)

        count = next(self.counter)
        entry = [priority, count, key, data]
        entries_map[key] = entry
        heapq.heappush(self.queue, entry)

    def remove_task(self, key: Hashable) -> None:
        """Marks an existing task as removed.

        Args:
            key: The key of the task.

        Raises:
            KeyError: If the key is not found in the queue.
        """

        entry = self.entries_map.pop(key)
        entry[-1] = self.removed_tag

    def pop_task(self) -> tuple[Any, Hashable, Any]:
        """Removes and returns the lowest priority task.

        Returns:
            A tuple (priority, key, data) for the lowest-priority pending task.

        Raises:
            KeyError: If the queue is empty.
        """

        queue = self.queue
        while queue:
            priority, _, key, data = heapq.heappop(queue)
            if data!=self.removed_tag:
                del self.entries_map[key]
                return (priority, key, data)

        raise KeyError("pop from an empty priority queue")

    def __len__(self) -> int:

        return len(self.entries_map)

def scalar(val: Any) -> Any:
    """Converts a 0-dim numpy array to a Python scalar value.

    If `val` is not a 0-dim array, it is returned unchanged.

    Args:
        val: Value to convert. Can be a 0-dim numpy array or any other type.

    Returns:
        A Python scalar if `val` is a 0-dim array, otherwise `val` unchanged.
    """

    if hasattr(val, "item"):
        return val.item()
    else:
        return val

def remove_small_comp(
        img_bin: np.ndarray,
        tam_threshold: int = 20,
        img_label: np.ndarray | None = None,
        structure: np.ndarray | None = None,
        ) -> np.ndarray:
    """Removes connected components smaller than `tam_threshold` from a binary image.

    If `img_label` is not None, the provided labels are used as components.

    Args:
        img_bin: Binary image.
        tam_threshold: Minimum component size in pixels. Components smaller than this
            value are removed.
        img_label: Pre-computed label array. Must have the same format as the array
            returned by `scipy.ndimage.label`. Zero values are ignored.
        structure: Structuring element used for detecting connected components.

    Returns:
        Binary image with small components removed.
    """

    if img_label is None:
        img_lab, num_comp = ndi.label(img_bin, structure)
    else:
        num_comp = img_label.max()
    tam_comp = ndi.sum(img_bin, img_lab, range(num_comp+1))

    mask = tam_comp>tam_threshold
    mask[0] = False

    img_bin_final = mask[img_lab].astype(np.uint8)

    return img_bin_final

def ips_edges_to_img(
        ips, edges, img_shape, node_color=(255, 0, 0), edge_color=(0, 0, 255), out_img=None):
    """Draws interest points and edges into an image array.

    `ips` and `edges` are used in the network creation module.

    Args:
        ips: Bifurcations and terminations identified in an image.
        edges: Edge list containing blood vessel segments. Each element is a tuple
            (node1, node2, path), where path contains the pixels of the segment.
        img_shape: Shape of the output image.
        node_color: RGB color for the interest points.
        edge_color: RGB color for the edges.
        out_img: If provided, the graph is drawn onto this array. Otherwise a new
            array is created.

    Returns:
        Image array with the drawn interest points and edges.
    """

    if out_img is None:
        out_img = np.zeros((*img_shape, 3), dtype=np.uint8)

    for ip in ips:
        for pixel in ip.pixels:
            out_img[tuple(pixel)] = node_color

    for edge in edges:
        for pixel in edge[2]:
            out_img[tuple(pixel)] = edge_color

    return out_img

def graph_to_img(
        graph: nx.Graph,
        img_shape: tuple[int, ...] | None = None,
        node_color: tuple[int, int, int] = (255, 0, 0),
        node_pixels_color: tuple[int, int, int] = (255, 255, 255),
        edge_color: tuple[int, int, int] = (0, 0, 255),
        out_img: np.ndarray | None = None,
        ) -> np.ndarray:
    """Draws a NetworkX graph into an image array.

    Requires nodes to have 'pixels' attributes and edges to have 'path' attributes.
    Optionally uses 'center' node attributes for the node center color.

    Args:
        graph: Graph whose nodes contain 'pixels' and optionally 'center' attributes,
            and whose edges contain a 'path' attribute.
        img_shape: Shape of the output image. If None, uses `graph.graph['shape']`.
        node_color: RGB color for the center pixel of each node.
        node_pixels_color: RGB color for all non-center pixels of each node.
        edge_color: RGB color for the edge path pixels.
        out_img: If provided, the graph is drawn onto this array. Otherwise a new
            array is created.

    Returns:
        Image array with the graph drawn onto it.
    """

    if img_shape is None:
        img_shape = graph.graph["shape"]

    if out_img is None:
        out_img = np.zeros((*img_shape, 3), dtype=np.uint8)

    for _, pixels in graph.nodes(data="pixels"):
        for pixel in pixels:
            out_img[tuple(pixel)] = node_pixels_color

    has_center = all("center" in graph.nodes[n] for n in graph.nodes)
    if has_center:
        for _, center in graph.nodes(data="center"):
            out_img[tuple(center)] = node_color

    for _, _, path in graph.edges(data="path"):
        for pixel in path:
            out_img[tuple(pixel)] = edge_color

    return out_img

def gaussian_filter_with_anchors(
    y: np.ndarray, 
    anchor_indices: list, 
    sigma: float = 1.0, 
    correction_sigma: float | None = None
) -> np.ndarray:
    """Smooths a 1D signal while preserving exact values at specified anchor indices.

    Applies a Gaussian filter to reduce discretization artifacts, then uses a localized
    Gaussian blending window at each anchor index to restore the original value.

    Args:
        y: 1D array containing the signal to smooth.
        anchor_indices: Integer indices whose values must remain unchanged after smoothing.
        sigma: Standard deviation of the Gaussian smoothing kernel. Values between 1.0
            and 2.0 are typically suitable for discretization artifacts.
        correction_sigma: Width of the blending window for anchor correction. Defaults to
            the same value as `sigma` if None.

    Returns:
        Smoothed signal array of the same shape as `y`, with anchor values preserved.
    """
    if correction_sigma is None:
        correction_sigma = sigma

    # Apply uniform Gaussian smoothing to the entire array.
    # mode='nearest' prevents the endpoints from dipping toward zero.
    y_smooth = gaussian_filter1d(y, sigma=sigma, mode="nearest")

    indices = np.arange(len(y))
    
    for anchor_idx in anchor_indices:
        # Calculate the exact displacement at the anchor point.
        delta = y[anchor_idx] - y_smooth[anchor_idx]
        
        # Create a localized Gaussian correction window centered on the anchor.
        # This ensures the correction fades out smoothly into the surrounding data 
        # rather than creating a sharp, artificial spike.
        correction_window = np.exp(-0.5 * ((indices - anchor_idx) / correction_sigma)**2)
        
        # Blend the correction back into the smoothed signal.
        y_smooth += delta * correction_window
        
    for anchor_idx in anchor_indices:
        # Hard-enforce the anchor to eliminate any microscopic floating-point drift.
        y_smooth[anchor_idx] = y[anchor_idx]
    
    return y_smooth