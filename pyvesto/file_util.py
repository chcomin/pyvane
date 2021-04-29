"""Utility functions for creating and manipulating files and directories."""

from pathlib import Path
from collections import deque
import natsort
import copy
import os
from shutil import copy2

class Tree:
    """Class representing a directed tree. If you want node and edge attributes, the tree should
    be build incrementally, passing None in the constructor for both `nodes` and `edges`.

    Parameters
    ----------
    nodes : list of hashable objects, optional
        List of hashable objects.
    edges : list of tuples, optional
        List of tuples indicating adjacent nodes in the form (node1, node2).
    """

    def __init__(self, nodes=None, edges=None):

        if nodes is None:
            nodes = {}
        if edges is None:
            edges = {}

        self.edges = edges
        self.nodes = nodes

    def add_node(self, node, attrs=None):
        """Add a node to the tree. A ValueError is raised if the node is already in the tree.

        Parameters
        ----------
        node : hashable
            Node do add to the graph. Usually, it is a number or a string.
        attrs : dict, optional
            Dictionary containing node attributes.
        """

        if attrs is None:
            attrs = {}

        nodes = self.nodes
        if node in nodes:
            raise ValueError('Node already in graph')

        nodes[node] = attrs

    def add_edge(self, source, target, attrs=None):
        """Add a directed edge to the tree. A ValueError is raised if the nodes are not in the tree.
        If the edge is already in the tree, the attributes are updated.

        Parameters
        ----------
        source : hashable
            Parent node.
        target : hashable
            Target node.
        attrs : dict, optional
            Dictionary containing edge attributes.
        """

        if attrs is None:
            attrs = {}

        nodes = self.nodes
        edges = self.edges

        if source not in nodes:
            raise ValueError('First node is not in the graph')
        if target not in nodes:
            raise ValueError('Second node is not in the graph')

        if source not in edges:
            edges[source] = {target:attrs}
        else:
            edges[source][target] = attrs

    def get_nodes(self, data=False):
        """Get nodes in the graph.

        Parameters
        ----------
        data : bool
            If `data=False`, returns a list containing node names, otherwise returns
            a list containing tuples in the form (node, attrs), where attrs is the
            dictionary of attributes of the node.

        Returns
        -------
        list
             List of nodes. See definition of parameter `data` for a description of the returned value.
        """

        if data:
            return list(self.nodes.items())
        else:
            return list(self.nodes.keys())

    def get_edges(self, data=False):
        """Get edges in the graph.

        Parameters
        ----------
        data : bool
            If `data=False`, returns a list of tuples containing the edges, otherwise returns
            a list containing tuples in the form (source, target, attrs), where attrs is the
            dictionary of attributes of the edge.

        Returns
        -------
        list
             List of edges. See definition of parameter `data` for a description of the returned value.
        """

        edges = self.edges
        edge_list = []
        for source, successors in edges.items():
            for target, attrs in successors.items():
                if data:
                    edge_list.append((source, target, attrs))
                else:
                    edge_list.append((source, target))

        return edge_list

    def successors(self, node):
        """Return neighbors that are successors of a node.

        Parameters
        ----------
        node : hashable
            A node in the tree.

        Returns
        -------
        list
            List of nodes that are successors of `node`.
        """

        try:
            successors = self.edges[node]
        except KeyError:
            raise ValueError('Node is not in the tree.')

        return list(successors.keys())

    def copy(self):
        """Creates a copy of the tree.

        Returns
        -------
        Tree
            A copy of the tree.
        """

        return Tree(copy.deepcopy(self.get_nodes(True)), copy.deepcopy(self.get_edges(True)))

class FileTree(Tree):
    """Class representing a file and directory tree. Each node represents a file or directory. The
    name of a node is the absolute path of the respective file or directory. The tree needs to be
    created from an existing directory in the system.

    Parameters
    ----------
    root_path : Union[str, Path]
        The path for the root of the tree.
    name_filter : func
        Function that receives the name of a file or directory and returns True if the item should be
        added to the tree or False otherwise. For directories, files and subdirectories inside each
        directory will also not be added.
    """

    def __init__(self, root_path, name_filter=None):
        super().__init__()

        if name_filter is None:
            name_filter = lambda x: True

        self.root_path = Path(root_path)
        self.name_filter = name_filter

        self._build_tree()

    def _build_tree(self):
        """Builds the tree.
        """

        root_path = self.root_path
        self.add_node(root_path)

        # Breadth-first search
        file_queue = deque()
        file_queue.append(root_path)
        while len(file_queue)>0:
            file = file_queue.popleft()
            if file.is_dir():
                for child in file.iterdir():
                    if self.name_filter(child.name):
                        file_queue.append(child)
                        self.add_node(child)
                        self.add_edge(file, child)

    def get_files(self):
        """Return absolute paths for all files in the tree (that is, directories are not included). The
        function uses method Path.is_file() from pathlib. Therefore, all files and directories in the tree
        must exist for the function to work.

        Returns
        -------
        list of Path
            Nodes in the tree representing files.
        """

        files = []
        for node in self.get_nodes():
            if node.is_file():
                files.append(node)

        return natsort.natsorted(files)

    def get_directories(self):
        """Return absolute paths for all directories in the tree (that is, files are not included). The
        function uses method Path.is_dir() from pathlib. Therefore, all files and directories in the tree
        must exist for the function to work.

        Returns
        -------
        list of Path
            Nodes in the tree representing directories.
        """

        directories = []
        for node in self.get_nodes():
            if node.is_dir():
                directories.append(node)

        return natsort.natsorted(directories)

    def get_paths_from_name(self, file_name):
        """Return absolute paths for files in the tree having name `file_name`. More than one
        path will be returned if the tree contains more than one file with the same name.

        Parameters
        ----------
        file_name : Union[str, Path]

        Returns
        -------
        list of Path
            The files in the tree having name `file_name`.
        """

        paths = []
        for node in self.get_nodes():
            if file_name==node.name:
                paths.append(node)

        return natsort.natsorted(paths)

    def change_root(self, new_root):
        """Change the absolute path of the root for a tree containing a directory structure. Returns
        a new tree of type Tree, since the files probably do not exist in the system and many methods
        from FileTree will not work.

        For instance, suppose a tree contains the following paths:

        root: /dir1/dir2/
        child1: /dir1/dir2/file1
        child2: /dir1/dir2/file2

        Calling this function with `new_root=/codes` will return a tree with nodes

        root: /codes
        child1: /codes/file1
        child2: /codes/file2

        Parameters
        ----------
        new_root : Union[str, Path]
            The new root.

        Returns
        -------
        new_file_tree : Tree
            A new tree with file and directory paths containing a new root. Note that the files and
            directories in the returned tree probably do not exist in the system and many methods
            from FileTree will not work.
        """

        new_root = Path(new_root)
        curr_root = self.root_path

        new_file_tree = Tree()
        for node, attrs in self.get_nodes(True):
            rel_path = node.relative_to(curr_root)
            new_node = new_root/rel_path
            new_file_tree.add_node(new_node, attrs)

        for node1, node2, attrs in self.get_edges(True):
            rel_path1 = node1.relative_to(curr_root)
            rel_path2 = node2.relative_to(curr_root)
            new_node1 = new_root/rel_path1
            new_node2 = new_root/rel_path2
            new_file_tree.add_edge(new_node1, new_node2, attrs)

        return new_file_tree

class Graph:
    """Initial implementation of a graph, not finished.
    """

    def __init__(self, nodes=None, edges=None, directed=False, attrs=None):

        if nodes is None:
            nodes = {}
        if edges is None:
            edges = {}
        if graph_attrs is None:
            attrs = {}

        self.edges = edges
        self.nodes = nodes
        self.directed = directed
        self.attrs = attrs

    def add_node(self, node, attrs=None):

        if attrs is None:
            attrs = {}

        nodes = self.nodes
        if node in nodes:
            raise ValueError('Node already in graph')

        nodes[node] = attrs

    def add_edge(self, node1, node2, attrs=None):

        if attrs is None:
            attrs = {}

        nodes = self.nodes
        edges = self.edges

        if node1 not in nodes:
            raise ValueError('First node is not in the graph')
        if node2 not in nodes:
            raise ValueError('Second node is not in the graph')

        if node1 in edges:
            if node2 in edges[nodes1]:
                raise ValueError('Edge already in graph')
            else:
                edges[node1][node2] = attrs
        else:
            edges[node1] = {node2:attrs}

        if not directed:
            if node2 not in edges:
                edges[node2] = {node1:attrs}
            else:
                if node1 in edges[nodes2]:
                    print(f"Warning, undirected graph is inconsistent. Edge ({node1},{node2}) was in the graph but ({node2},{node1}) wasn't.")
                edges[node2][node1] = attrs

    def get_nodes(self, data=False):

        if data:
            return list(self.nodes.items())
        else:
            return list(self.nodes.keys())

    def get_edges(self, data=False):

        edges = self.edges
        edge_list = []
        edge_set = set()
        for node1, successors in edges.items():
            for node2, attrs in successors.items():
                if self.directed or (node2, node1) not in edge_set:
                    if data:
                        edge_list.append((node1, node2, attrs))
                    else:
                        edge_list.append((node1, node2))

                    edge_set.add((node1, node2))

        return edge_list

    def successors(self, node):

        successors = self.edges[node]
        return list(successors.keys())

    def predecessors(self, node):

        predecessors = []
        for node_it, successors in self.edges.items():
            if node in successors:
                predecessors.append(node_it)

        return predecessors

    def copy(self):

        return Graph(copy.deepcopy(self.get_nodes(True)), copy.deepcopy(self.get_edges(True)), copy.deepcopy(self.attrs))

def get_files(root_path, name_filter=None):
    """Recursively get all files and directories inside `root_path`.

    Parameters
    ----------
    root_path : Union[str, Path]
        Root directory.
    name_filter : func
        Function that receives the name of a file or directory and returns True if the item should be
        added to the tree or False otherwise. For directories, files and subdirectories inside each
        directory will also not be added.

    Returns
    -------
    FileTree
        A tree representing a file and directory tree. Each node represents a file or directory. The
        name of a node is the absolute path of the respective file or directory.
    list of Path
        A list of all files inside `root_path`, excluding files in `exclude_names`.
    """

    file_tree = FileTree(root_path, name_filter)

    return file_tree, file_tree.get_files()

def directory_has_file(path):
    """Check if a directory has at least one file.

    Parameters
    ----------
    path : Union[str, Path]
        A path in the system.

    Returns
    -------
    bool
        Returns True if the directory contains at least one file. Returns False otherwise.
    """

    path = Path(path)

    for file in path.iterdir():
        if file.is_file():
            return True
    return False

def iterate_directory_path(path):
    """For path x/y/z, yields a list [x, x/y, x/y/z].

    Parameters
    ----------
    path : Union[str, Path]
        A file path

    Yields
    ------
    Path
        Sequence of paths with increasing depth.
    """

    path = Path(path)
    dirs = path.parts
    par_path = Path('.')
    for dir in dirs:
        par_path = par_path/dir
        yield par_path

def create_directory(directory):
    """Creates a directory and any parent directory that do not exist in the system. For instance,
    for the path x/y/z, directories x, x/y and x/y/z will be created if they do not exist.

    Parameters
    ----------
    directory : Union[str, Path]
        A directory.
    """

    directory = Path(directory)
    for dir in iterate_directory_path(directory):
        if not dir.exists(): dir.mkdir()

def get_file_tag(file, directory=None, sep='@', include_ext=False):

        file_parts = file.parts
        if directory is not None:
            ind = file_parts.index(directory)
            file_parts = file_parts[ind+1:]

        if include_ext:
            tag = sep.join(file_parts)
        else:
            tag = sep.join(file_parts[:-1])
            if len(tag)>0:
                tag += sep+file.stem
            else:
                tag = file.stem

        return tag

def flatten_directory(in_folder, out_folder, name_filter=None, sep='@'):

    in_folder = Path(in_folder)
    directory = in_folder.parts[-1]
    _, files = get_files(in_folder, name_filter)
    for file in files:
        file_tag = get_file_tag(file, directory, sep, include_ext=True)
        copy2(file, out_folder/file_tag)

def make_directories(file_tree, out_dir, gen_step_dirs=None, gen_subdirs=None):
    """Make directories for saving experiment data.

    Parameters
    ----------
    param : list
        Description

    Returns
    -------
    param : int
        Description
    """

    if gen_step_dirs is None:
        gen_step_dirs = []
    if gen_subdirs is None:
        gen_subdirs = []

    out_dir = Path(out_dir)
    curr_root = file_tree.root_path
    curr_root_parent = curr_root.parent

    # Generate directories for each experiment
    for directory in file_tree.get_directories():
        rel_path = directory.relative_to(curr_root_parent)
        for gen_dir in gen_step_dirs:
            # Generate subdirectories for each experiment (e.g., max proj, numpy, etc)
            if directory_has_file(directory):
                for gen_subdir in gen_subdirs:
                    new_subdir = out_dir/gen_dir/rel_path/gen_subdir
                    create_directory(new_subdir)

# Unused functions
def _find_common_parent(path1, path2):

    common_parts = []
    for part1, part2 in zip(path1.parts, path2.parts):
        if part1==part2:
            common_parts.append(part1)

    if len(common_parts)==0:
        common_parent = None
    else:
        common_parent = Path('/'.join(common_parts))

    return common_parent

def _relative_to(ref_path, other_path):

    common_parent = _find_common_parent(ref_path, other_path)
    if common_parent is None:
        return None
    else:
        rel_out_path = other_path.relative_to(common_parent)

    num_levels = len(ref_path.parts) - len(common_parent.parts)
    if ref_path.is_file():
        num_levels -= 1

    rel_path = Path('../'*num_levels)/rel_out_path

    return rel_path

if __name__=='__main__':

    # run some tests
    p = Path('E:/Dropbox/ufscar/pyvesto/pyvesto/img_io.py')
    out_dir1 = Path('E:/Dropbox/ufscar/codes/img.py')
    out_dir2 = Path('E:/Dropbox/ufscar/pyvesto/pyvesto/codes')
    out_dir3 = Path('E:/codes')
    out_dir4 = Path('C:/codes')
    out_dir5 = Path('E:/Dropbox/ufscar/pyvesto/pyvesto')

    print(_relative_to(p, out_dir1))