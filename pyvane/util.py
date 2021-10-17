"""Utility functions for image and graph manipulation as well as some useful classes."""

import heapq
import itertools
import scipy.ndimage as ndi
import numpy as np
import networkx as nx

try:
    import igraph
except ImportError:
    raise ImportError('igraph not found, will not be able to convert graphs to igraph format.')

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
                raise ValueError('`priorities` and `keys` need to be both None or both list.')
        else:
            if keys is None:
                raise ValueError('`priorities` and `keys` need to be both None or both list.')

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
        self.removed_tag = '<removed>'

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

        Returns
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

        raise KeyError('pop from an empty priority queue')

    def __len__(self):

        return len(self.entries_map)

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

    Returns
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

def get_print_interval(n, frac=100):
    """Utility function for getting printing interval. `n` is typically the number of iterations
    of a loop and `frac` the number of times the message will be printed.

    Parameters
    ----------
    n : int
        Number of iterations.
    frac : int
        Number of desired messages.

    Returns
    ------
    print_interv : int
        The number of iterations between two print calls.
    """

    print_interv = n//frac
    if print_interv==0:
        print_interv = 1

    return print_interv

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

    Returns
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

    Returns
    -------
    out_img : ndarray
        The image drawn.
    """

    if img_shape is None:
        img_shape = graph.graph['shape']

    if out_img is None:
        out_img = np.zeros((*img_shape, 3), dtype=np.uint8)

    for node, pixels in graph.nodes(data='pixels'):
        for pixel in pixels:
            out_img[tuple(pixel)] = node_pixels_color

    for node, center in graph.nodes(data='center'):
        out_img[tuple(center)] = node_color

    for _, _, path in graph.edges(data='path'):
        for pixel in path:
            out_img[tuple(pixel)] = edge_color

    return out_img

def erase_line(num_chars=80):

    print('\r'+' '*num_chars, end='\r')

# Functions related to igraph
def to_igraph(ips, edges):
    """Convert an interest point list and an edge list to igraph.

    Parameters
    ---------
    ips : list of InterestPoint
        Bifurcations and terminations identified in an image.
    edges : list of tuple
        Edge list containing blood vessel segments. Each element is a tuple (node1, node2, path), where
        path contains the pixels of the segment.

    Returns
    -------
    graph : igraph.Graph
        The created graph.

    """

    ip_dict = {'pixels':[], 'center':[], 'type':[], 'ndim':[], 'branches':[]}
    for ip in ips:
        ip_dict['pixels'].append(ip.pixels)
        ip_dict['center'].append(ip.center)
        ip_dict['type'].append(ip.type)
        ip_dict['ndim'].append(ip.ndim)
        ip_dict['branches'].append(ip.branches)

    edge_dict = {'path':[]}
    edge_list = []
    for edge in edges:
        edge_list.append((edge[0], edge[1]))
        edge_dict['path'].append(edge[2])

    graph = igraph.Graph(n=len(ips), edges=edge_list, vertex_attrs=ip_dict, edge_attrs=edge_dict)

    return graph

def nx_to_igraph(graph):
    """Convert a networkx graph to igraph. The graph can contain node and edge attributes as well as
    graph attributes.

    Parameters
    ----------
    graph : networkx.Graph
        Networkx graph to convert.

    Returns
    -------
    ig_graph : igraph.Graph
        The converted graph as an igraph.Graph object.
    """

    node_attrs_keys = set()
    for node, attrs in graph.nodes(data=True):
        [node_attrs_keys.add(k) for k in attrs]

    edge_attrs_keys = set()
    for node1, node2, attrs in graph.edges(data=True):
        [edge_attrs_keys.add(k) for k in attrs]

    node_attrs_list = dict(zip(node_attrs_keys, [[] for i in range(len(node_attrs_keys))]))
    for node, node_attrs in graph.nodes(data=True):
        for node_attrs_key in node_attrs_keys:
            if node_attrs_key not in node_attrs:
                print(f'Warning, node {node} has no attribute {node_attrs_key}.')
                att_value = None
            else:
                att_value = node_attrs[node_attrs_key]
            node_attrs_list[node_attrs_key].append(att_value)

    edges = []
    edge_attrs_list = dict(zip(edge_attrs_keys, [[] for i in range(len(edge_attrs_keys))]))
    for node1, node2, edge_attrs in graph.edges(data=True):
        edges.append((node1, node2))
        for edge_attrs_key in edge_attrs_keys:
            if edge_attrs_key not in edge_attrs:
                print(f'Warning, edge ({node1},{node2}) has no attribute {edge_attrs_key}.')
                att_value = None
            else:
                att_value = edge_attrs[edge_attrs_key]
            edge_attrs_list[edge_attrs_key].append(att_value)

    is_directed = graph.is_directed()
    ig_graph = igraph.Graph(n=graph.number_of_nodes(), edges=edges, directed=is_directed, graph_attrs=graph.graph,
                            vertex_attrs=node_attrs_list, edge_attrs=edge_attrs_list)

    return ig_graph

def igraph_to_nx(ig_graph):
    """Convert a igraph graph to networkx. The graph can contain node and edge attributes as well as
    graph attributes.

    Parameters
    ----------
    ig_graph : igraph.Graph
        igraph graph to convert.

    Returns
    -------
    graph : networkx.Graph
        The converted graph as an networkx.Graph object.
    """

    node_attrs_keys = ig_graph.vs.attribute_names()
    edge_attrs_keys = ig_graph.es.attribute_names()

    is_directed = ig_graph.is_directed()
    is_multiple = max(ig_graph.is_multiple())

    if is_directed:
        if is_multiple:
            constructor = nx.MultiDiGraph
        else:
            constructor = nx.DiGraph
    else:
        if is_multiple:
            constructor = nx.MultiGraph
        else:
            constructor = nx.Graph

    graph = constructor()
    for node_idx, node in enumerate(ig_graph.vs):
        node_attrs = {}
        for node_attr_key in node_attrs_keys:
            node_attrs[node_attr_key] = node[node_attr_key]
        graph.add_node(node_idx, **node_attrs)

    for edge_idx, edge in enumerate(ig_graph.es):
        edge_attrs = {}
        for edge_attr_key in edge_attrs_keys:
            edge_attrs[edge_attr_key] = edge[edge_attr_key]
        graph.add_edge(*edge.tuple, **edge_attrs)

    graph_attrs_keys = ig_graph.attributes()
    graph.graph = {graph_attr_key:ig_graph[graph_attr_key] for graph_attr_key in graph_attrs_keys}

    return graph

def igraph_to_img(graph, img_shape, node_color=(255, 0, 0), node_pixels_color=(255, 255, 255),
                 edge_color=(0, 0, 255), out_img=None):
    """Draw igraph graph in an image.

    Parameters
    ---------
    graph : networkx.Graph
        Graph containing node and edge positions.
    img_shape : tuple of int
        Image size to draw the network.
    node_color : tuple of int, optional
        Color to use for the center position of a node.
    node_pixels_color : tuple of int, optional
        Color to use for pixels associated with a node.
    edge_color : tuple of int, optional
        Color to use for the edges.
    out_img : ndarray, optional
        If provided, the image is drawn on this array.

    Returns
    -------
    out_img : ndarray
        The image drawn.
    """

    if out_img is None:
        out_img = np.zeros((*img_shape, 3), dtype=np.uint8)

    for pixels in graph.vs['pixels']:
        if isinstance(pixels[0], (tuple, list)):
            for pixel in pixels:
                out_img[tuple(pixel)] = node_pixels_color
        else:
            pixel = pixels
            out_img[tuple(pixel)] = node_pixels_color

    for pixel in graph.vs['center']:
        out_img[tuple(pixel)] = node_color

    for path in graph.es['path']:
        for pixel in path:
            out_img[tuple(pixel)] = edge_color

    return out_img

def match_graphs_igraph(ips, ig):
    """Map indices of nodes in graph `ig` to indices of points in `ips`. Two elements match if
    their associated pixels are identical.

    Parameters
    ----------
    ips : list of InterestPoint
        Bifurcations and terminations identified in an image.
    ig : igraph.Graph
        Graph in the igraph format.

    Returns
    -------
    igraph_idx_to_ip_idx : dict
        Map having igraph nodes as keys and associated interest points indices as values. Missing
        keys indicate that a match was not found.
    """

    igraph_idx_to_ip_idx = {}
    for igraph_idx, v in enumerate(ig.vs):
        ig_hash = hash(tuple(sorted(map(tuple, v['pixels']))))
        for ip_idx, ip in enumerate(ips):
            ip_hash = hash(tuple(sorted(ip.pixels)))
            if ig_hash==ip_hash:
                igraph_idx_to_ip_idx[igraph_idx] = ip_idx

    return igraph_idx_to_ip_idx

if __name__=='__main__':

    # Run some tests

    ig_graph = Graph(n=4, edges=[(0,0),(0,1),(0,1),(0,2),(1,2),(0,3)], graph_attrs={'file':'graph.gml'},
                     vertex_attrs={'color':['red', 'green', 'blue', 'yellow']}, edge_attrs={'length':[0, 2, 2.5, 5, 7, 10]})

    graph = igraph_to_nx(ig_graph)
    ig_graph2 = nx_to_igraph(graph)

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from([(0, {'color':'blue'}), (1, {'color':'red'}), (2, {'color':'green'})])
    nx_graph.add_edges_from([(0, 1, {'relation':'enemy'}), (0, 2, {'relation':'enemy'}), (1, 2, {'relation':'friend'})])

    graph = nx_to_igraph(nx_graph)
    nx_graph2 = igraph_to_nx(graph)