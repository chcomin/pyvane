"""Utility functions for image and graph manipulation as well as some useful classes."""

import heapq
import itertools

import networkx as nx
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter1d


class PriorityQueue:
    """Priority queue that allows changes to, or removal of, elements of a pending task.

    Parameters
    ----------
    priorities : list of Union[int, float]
        The priorities of the task. Lower values are executed first. The values can actually be
        any type that can be compared, that is, it must be possible to define which value is
        lower. If None, an empty queue is created.
    keys : list of hashable
        List containing a unique key for each task. Must have the same size as `priorities`.
    data : list, optional
        Data to associate to each task. The elements can be any object. Must have the same size
        as `priorities`. If None, all tasks will have None as data.
    """

    def __init__(self, priorities=None, keys=None, data=None):

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
        _, _, keys, _ = zip(*entries)      # Obtain keys again since the order of the entries has changed
        entries_map = dict(zip(keys, entries))

        self.queue = entries
        self.entries_map = entries_map
        self.counter = itertools.count(start=num_entries)
        self.removed_tag = "<removed>"

    def add_task(self, priority, key, data=None):
        """Add a new task or update the priority of an existing task.

        Parameters
        ----------
        priority : Union[int, float]
            The priority of the task. Lower values are executed first. The values can actually be
            any type that can be compared, that is, it must be possible to define which value is
            lower.
        key : hashable
            Unique key for the task.
        data : object, optional
            Data to associate to the task.
        """

        entries_map = self.entries_map
        if key in entries_map:
            self.remove_task(key)

        count = next(self.counter)
        entry = [priority, count, key, data]
        entries_map[key] = entry
        heapq.heappush(self.queue, entry)

    def remove_task(self, key):
        """Mark an existing task as removed.  Raises KeyError if not found.

        Parameters
        ----------
        key : hashable
            The key of the task.
        """

        entry = self.entries_map.pop(key)
        entry[-1] = self.removed_tag

    def pop_task(self):
        """Remove and return the lowest priority task. Raises KeyError if empty.

        Returns:
        -------
        tuple
            A tuple associated with the task, with elements (priority, key, data).
        """

        queue = self.queue
        while queue:
            priority, count, key, data = heapq.heappop(queue)
            if data!=self.removed_tag:
                del self.entries_map[key]
                return (priority, key, data)

        raise KeyError("pop from an empty priority queue")

    def __len__(self):

        return len(self.entries_map)

def scalar(val):
    """Utility function for converting a 0-dim array to a scalar value. If `val` is not a 0-dim 
    array, it is returned unchanged.
    """

    if hasattr(val, "item"):
        return val.item()
    else:
        return val

def remove_small_comp(img_bin, tam_threshold=20, img_label=None, structure=None):
    """For a binary image, remove connected components smaller than `tam_threshold`. If `img_label`
    is not None, use the provided labels as components.

    Parameters
    ----------
    img_bin : ndarray
        Binary image.
    tam_threshold : int
        Size threshold for removing components.
    img_label : ndarray, optional
        Array containing image components. Must have the same format as the array returned
        by scipy.ndimage.label(...). Zero values are ignored.
    structure : ndarray, optional
        Structuring element used for detecting connected components.

    Returns:
    -------
    img_bin_final : ndarray
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

def ips_edges_to_img(ips, edges, img_shape, node_color=(255, 0, 0), edge_color=(0, 0, 255), out_img=None):
    """Draw interest points and edges in an image. `ips` and `edges` are used in the network creation
    module.

    Parameters
    ---------
    ips : list of InterestPoint
        Bifurcations and terminations identified in an image.
    edges : list of tuple
        Edge list containing blood vessel segments. Each element is a tuple (node1, node2, path), where
        path contains the pixels of the segment.
    img_shape : tuple of int
        Image size to draw the network.
    node_color : tuple of int, optional
        Color to use for the interest points.
    edge_color : tuple of int, optional
        Color to use for the edges.
    out_img : ndarray, optional
        If provided, the image is drawn on this array.

    Returns:
    -------
    out_img : ndarray
        The image drawn.
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

def graph_to_img(graph, img_shape=None, node_color=(255, 0, 0), node_pixels_color=(255, 255, 255),
                 edge_color=(0, 0, 255), out_img=None):
    """Draw networkx graph in an image.

    Parameters
    ---------
    graph : networkx.Graph
        Graph containing node and edge positions.
    img_shape : tuple of int, optional
        Image size to draw the network.
    node_color : tuple of int, optional
        Color to use for the center position of a node.
    node_pixels_color : tuple of int, optional
        Color to use for pixels associated with a node.
    edge_color : tuple of int, optional
        Color to use for the edges.
    out_img : ndarray, optional
        If provided, the image is drawn on this array.

    Returns:
    -------
    out_img : ndarray
        The image drawn.
    """

    if img_shape is None:
        img_shape = graph.graph["shape"]

    if out_img is None:
        out_img = np.zeros((*img_shape, 3), dtype=np.uint8)

    for node, pixels in graph.nodes(data="pixels"):
        for pixel in pixels:
            out_img[tuple(pixel)] = node_pixels_color

    has_center = all("center" in graph.nodes[n] for n in graph.nodes)
    if has_center:
        for node, center in graph.nodes(data="center"):
            out_img[tuple(center)] = node_color

    for _, _, path in graph.edges(data="path"):
        for pixel in path:
            out_img[tuple(pixel)] = edge_color

    return out_img

def gaussian_filter_with_anchors(
    y: np.ndarray, 
    anchor_indices: list, 
    sigma: float = 1.0, 
    correction_sigma: float = None
) -> np.ndarray:
    """
    Smooths a 1D signal to remove discretization artifacts using a Gaussian filter,
    while strictly preserving the exact value at anchor_indices using a blending window.
    
    Args:
        y: 1D numpy array containing the signal.
        anchor_indices: The list of integer indices that must remain unchanged.
        sigma: The standard deviation of the main Gaussian smoothing kernel.
               For discretization, 1.0 to 2.0 is usually ideal.
        correction_sigma: The width of the blending window for the anchor correction.
                          If None, it defaults to the same value as `sigma`.
                          
    Returns:
        A 1D numpy array of the smoothed, anchored signal.
    """
    if correction_sigma is None:
        correction_sigma = sigma

    # Apply uniform Gaussian smoothing to the entire array.
    # mode='nearest' prevents the endpoints from dipping toward zero.
    y_smooth = gaussian_filter1d(y, sigma=sigma, mode='nearest')

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