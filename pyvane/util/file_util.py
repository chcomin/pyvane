"""Utility functions for creating and manipulating files and directories."""

import copy
from collections import deque
from collections.abc import Callable, Generator
from pathlib import Path
from shutil import copy2
from typing import Any

import natsort


class Tree:
    """Class representing a directed tree.

    If you want node and edge attributes, the tree should be built incrementally,
    passing None in the constructor for both `nodes` and `edges`.
    """

    def __init__(self, nodes: dict | None = None, edges: dict | None = None) -> None:
        """Args:
        nodes: List of hashable objects representing the tree nodes.
        edges: List of tuples indicating adjacent nodes in the form (node1, node2).
        """

        if nodes is None:
            nodes = {}
        if edges is None:
            edges = {}

        self.edges = edges
        self.nodes = nodes

    def add_node(self, node: Any, attrs: dict | None = None) -> None:
        """Adds a node to the tree.

        Args:
            node: Node to add to the tree. Usually a number or a string.
            attrs: Dictionary containing node attributes.

        Raises:
            ValueError: If the node is already in the tree.
        """

        if attrs is None:
            attrs = {}

        nodes = self.nodes
        if node in nodes:
            raise ValueError("Node already in graph")

        nodes[node] = attrs

    def add_edge(self, source: Any, target: Any, attrs: dict | None = None) -> None:
        """Adds a directed edge to the tree.

        If the edge already exists, its attributes are updated.

        Args:
            source: Parent node.
            target: Child/target node.
            attrs: Dictionary containing edge attributes.

        Raises:
            ValueError: If either source or target node is not in the tree.
        """

        if attrs is None:
            attrs = {}

        nodes = self.nodes
        edges = self.edges

        if source not in nodes:
            raise ValueError("First node is not in the graph")
        if target not in nodes:
            raise ValueError("Second node is not in the graph")

        if source not in edges:
            edges[source] = {target:attrs}
        else:
            edges[source][target] = attrs

    def get_nodes(self, data: bool = False) -> list:
        """Returns nodes in the tree.

        Args:
            data: If False, returns a list of node names. If True, returns a list of
                (node, attrs) tuples where attrs is the node attribute dictionary.

        Returns:
            List of node names, or list of (node, attrs) tuples if `data=True`.
        """

        if data:
            return list(self.nodes.items())
        else:
            return list(self.nodes.keys())

    def get_edges(self, data: bool = False) -> list:
        """Returns edges in the tree.

        Args:
            data: If False, returns a list of (source, target) tuples. If True, returns
                a list of (source, target, attrs) tuples where attrs is the edge attribute
                dictionary.

        Returns:
            List of (source, target) tuples, or (source, target, attrs) tuples if
            `data=True`.
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

    def successors(self, node: Any) -> list:
        """Returns the successors of a node.

        Args:
            node: A node in the tree.

        Returns:
            List of nodes that are successors of `node`.

        Raises:
            ValueError: If the node is not in the tree.
        """

        try:
            successors = self.edges[node]
        except KeyError as e:
            raise ValueError("Node is not in the tree.") from e

        return list(successors.keys())

    def copy(self) -> "Tree":
        """Creates a deep copy of the tree.

        Returns:
            A new Tree instance with copies of all nodes and edges.
        """

        return Tree(copy.deepcopy(self.get_nodes(True)), copy.deepcopy(self.get_edges(True)))

class FileTree(Tree):
    """Class representing a file and directory tree.

    Each node represents a file or directory, identified by its absolute path. The
    tree is constructed from an existing directory in the file system.
    """

    def __init__(
            self,
            root_path: str | Path,
            name_filter: Callable[[str], bool] | None = None,
            ) -> None:
        """Args:
        root_path: The path of the root directory.
        name_filter: Function that receives a file or directory name and returns True
            if the item should be included. If False for a directory, its contents
            are also excluded.
        """
        super().__init__()

        if name_filter is None:
            def f(x): return True
            name_filter = f

        self.root_path = Path(root_path)
        self.name_filter = name_filter

        self._build_tree()

    def _build_tree(self) -> None:
        """Builds the tree."""

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

    def get_files(self) -> list[Path]:
        """Returns absolute paths for all files in the tree.

        Directories are excluded. Requires all paths in the tree to exist on the
        file system (uses `Path.is_file()`).

        Returns:
            Sorted list of Path objects representing files in the tree.
        """

        files = []
        for node in self.get_nodes():
            if node.is_file():
                files.append(node)

        return natsort.natsorted(files)

    def get_directories(self) -> list[Path]:
        """Returns absolute paths for all directories in the tree.

        Files are excluded. Requires all paths in the tree to exist on the
        file system (uses `Path.is_dir()`).

        Returns:
            Sorted list of Path objects representing directories in the tree.
        """

        directories = []
        for node in self.get_nodes():
            if node.is_dir():
                directories.append(node)

        return natsort.natsorted(directories)

    def get_paths_from_name(self, file_name: str | Path) -> list[Path]:
        """Returns absolute paths for files in the tree having name `file_name`.

        More than one path is returned if the tree contains multiple files with the
        same name.

        Args:
            file_name: The file name (including extension) to search for.

        Returns:
            Sorted list of Path objects matching `file_name`.
        """

        paths = []
        for node in self.get_nodes():
            if file_name==node.name:
                paths.append(node)

        return natsort.natsorted(paths)

    def change_root(self, new_root: str | Path) -> "Tree":
        """Returns a new tree with the root path replaced by `new_root`.

        Returns a plain Tree (not FileTree), since the new paths likely do not exist
        on the file system. For example, for a tree rooted at /dir1/dir2/ with children
        /dir1/dir2/file1 and /dir1/dir2/file2, calling this function with
        `new_root=/codes` returns a tree with nodes /codes, /codes/file1, /codes/file2.

        Args:
            new_root: The new root path.

        Returns:
            A new Tree whose node paths have been rebased onto `new_root`.
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

def get_files(
        root_path: str | Path,
        name_filter: Callable[[str], bool] | None = None,
        ) -> tuple[FileTree, list[Path]]:
    """Recursively discovers all files and directories inside `root_path`.

    Args:
        root_path: Root directory to search.
        name_filter: Function that receives a file or directory name and returns True
            if the item should be included. If False for a directory, its contents
            are also excluded.

    Returns:
        A tuple (file_tree, files) where file_tree is a FileTree representing the
        directory structure and files is a sorted list of all file paths found.
    """

    file_tree = FileTree(root_path, name_filter)

    return file_tree, file_tree.get_files()

def directory_has_file(path: str | Path) -> bool:
    """Checks if a directory contains at least one file.

    Args:
        path: Path to the directory to check.

    Returns:
        True if the directory contains at least one file, False otherwise.
    """

    path = Path(path)
    return any(file.is_file() for file in path.iterdir())

def iterate_directory_path(path: str | Path) -> Generator:
    """Yields each prefix of a path with increasing depth.

    For example, for path x/y/z, yields x, x/y, x/y/z in order.

    Args:
        path: A file or directory path.

    Yields:
        Each partial path from the first component up to the full path.
    """

    path = Path(path)
    dirs = path.parts
    par_path = Path(".")
    for dir in dirs:
        par_path = par_path/dir
        yield par_path

def create_directory(directory: str | Path) -> None:
    """Creates a directory and all missing parent directories.

    For example, for path x/y/z, directories x, x/y, and x/y/z will be created
    if they do not already exist.

    Args:
        directory: Path to the directory to create.
    """

    directory = Path(directory)
    for dir in iterate_directory_path(directory):
        if not dir.exists(): 
            dir.mkdir()

def get_file_tag(
        file: Path,
        directory: str | None = None,
        sep: str = "@",
        include_ext: bool = False,
        ) -> str:
    """Builds a flat tag string encoding the path of a file.

    Joins path components with `sep` to create a unique flat filename that preserves
    path hierarchy information. Useful for flattening directory structures.

    Args:
        file: The file path to encode.
        directory: If provided, only path components after this directory name are used.
        sep: Separator used to join path components.
        include_ext: If True, includes the file extension in the tag.

    Returns:
        A flat string tag encoding the file's relative path.
    """

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

def flatten_directory(
        in_folder: str | Path,
        out_folder: str | Path,
        name_filter: Callable[[str], bool] | None = None,
        sep: str = "@",
        ) -> None:
    """Copies all files from a nested directory into a single flat directory.

    Each file is renamed using `get_file_tag` so that its original path hierarchy
    is encoded in the filename using `sep` as a separator.

    Args:
        in_folder: Source directory to flatten.
        out_folder: Destination directory where flattened files are copied.
        name_filter: Optional filter function for including/excluding files by name.
        sep: Separator used to encode subdirectory names in the flat filename.
    """

    in_folder = Path(in_folder)
    directory = in_folder.parts[-1]
    _, files = get_files(in_folder, name_filter)
    for file in files:
        file_tag = get_file_tag(file, directory, sep, include_ext=True)
        copy2(file, out_folder/file_tag)

def make_directories(
        file_tree: FileTree,
        out_dir: str | Path,
        gen_step_dirs: list | None = None,
        gen_subdirs: list | None = None,
        ) -> None:
    """Creates a mirrored output directory structure from a file tree.

    For each directory in `file_tree` that contains at least one file, creates
    step subdirectories under `out_dir` and optional leaf subdirectories within
    each step directory.

    Args:
        file_tree: The source file tree whose directory structure is mirrored.
        out_dir: Root output directory where new directories are created.
        gen_step_dirs: List of step directory names to create under each mirrored
            directory. If None, no directories are created.
        gen_subdirs: List of subdirectory names to create inside each step directory.
            If None, no subdirectories are created.
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

